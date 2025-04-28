"""Player-related endpoints for the NFL Data API."""

import logging
from typing import Dict, List, Optional, Any, Literal
from fastapi import APIRouter, Query, Path, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from enum import Enum
from datetime import datetime, date
import pandas as pd
import numpy as np

from ..stats_helpers import (
    get_player_stats,
    get_player_game_log,
    get_player_career_stats,
    get_player_on_field_stats,
    get_situation_stats,
    resolve_player,
    get_position_specific_stats_from_pbp,
    get_player_headshot_url,
    calculate_age
)

from ..data_loader import (
    load_players, load_weekly_stats, load_rosters,
    load_depth_charts, load_injuries, load_pbp_data,
    extract_situational_stats_from_pbp
)

from fastapi_cache.decorator import cache

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class HistoryType(str, Enum):
    """Enumeration of player history types."""
    ROSTER = "roster"
    DEPTH = "depth"
    INJURY = "injury"

class AggregationType(str, Enum):
    """Enumeration of aggregation types for player statistics."""
    CAREER = "career"
    SEASON = "season"
    WEEK = "week"

class SituationType(str, Enum):
    """Enumeration of situation types for filtered statistics."""
    RED_ZONE = "red_zone"
    THIRD_DOWN = "third_down"
    FOURTH_DOWN = "fourth_down"
    GOAL_LINE = "goal_line"
    TWO_MINUTE_DRILL = "two_minute_drill"

@router.get("/api/player/{name}/info")
@cache(expire=43200)  # 12 hours
async def get_player_info(
    name: str = Path(..., description="Player name (format: 'First Last')"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent available)", ge=1920, le=2024)
):
    """Get basic player information including ID, name, position, team, physical attributes, etc."""
    try:
        # Resolve player first - use await since resolve_player is now async
        player, alternatives = await resolve_player(name, season)
        
        if not player and alternatives:
            # Enhanced response for LLMs with more context
            return JSONResponse(
                status_code=300,
                content={
                    "error": f"Multiple players found matching '{name}'",
                    "suggestion": "Please specify which player you mean by providing additional context like team, position, or active/retired status",
                    "matches": [{
                        **alt,
                        "active_status": "Active" if alt.get("team_abbr") else "Inactive/Retired",
                        "full_context": f"{alt.get('display_name')} ({alt.get('position')}, {alt.get('team_abbr') or 'Retired'}, {alt.get('years_of_experience', 0)} years exp.)"
                    } for alt in alternatives]
                }
            )
            
        if not player:
            raise HTTPException(status_code=404, detail=f"No player found matching '{name}'")
            
        # Get player ID and headshot URL
        player_id = player["gsis_id"]
        
        # Use the headshot URL directly from the player data
        headshot_url = player.get("headshot", "")
        
        # Build focused player info response
        response = {
            "player_id": player_id,
            "name": player["display_name"],
            "position": player.get("position", ""),
            "team": player.get("team_abbr", ""),
            "age": calculate_age(player["birth_date"]) if "birth_date" in player else None,
            "experience": player.get("years_of_experience", None),
            "college": player.get("college_name", None),
            "height": player.get("height", None),
            "weight": player.get("weight", None),
            "headshot_url": headshot_url
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting player information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting player information: {str(e)}")

@router.get("/api/player/{name}/stats")
@cache(expire=43200)  # 12 hours
async def get_player_stats_endpoint(
    name: str = Path(..., description="Player name (format: 'First Last')"),
    aggregate: AggregationType = Query(AggregationType.SEASON, description="Aggregation level for statistics"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent available)"),
    week: Optional[int] = Query(None, description="Filter by week number (only applicable when aggregate=week)"),
    season_type: Optional[str] = Query(None, description="Filter by season type (REG or POST)"),
    opponent: Optional[str] = Query(None, description="Filter by opponent team abbreviation"),
    home_away: Optional[str] = Query(None, description="Filter by home/away games", enum=["home", "away"]),
    situation: Optional[SituationType] = Query(None, description="Filter by game situation"),
    team: Optional[str] = Query(None, description="Team abbreviation to disambiguate players with the same name")
):
    """Get player statistics with flexible aggregation options and situational filters.
    
    - aggregate: Choose between career totals, season totals, or week-by-week breakdown
    - season/week/season_type: Filter by season, week, or season type
    - opponent: Filter stats against a specific opponent
    - home_away: Filter for home or away games only
    - situation: Filter for specific game situations like red zone, third down, etc.
    - team: Specify team abbreviation to disambiguate players with the same name
    """
    try:
        # Special handling for known ambiguous players
        if name.lower() == "josh allen" and not team:
            # Default to the Bills QB Josh Allen if team not specified
            name = "Josh Allen"
            team = "BUF"  # Buffalo Bills
            
        # Resolve player first
        player, alternatives = await resolve_player(name, season)
        
        # If we have team information and multiple alternatives, filter by team
        if not player and alternatives and team:
            team_matches = [alt for alt in alternatives if alt.get('team_abbr') == team.upper()]
            if len(team_matches) == 1:
                player = team_matches[0]
                alternatives = []
            elif len(team_matches) > 1:
                alternatives = team_matches
        
        if not player and alternatives:
            return JSONResponse(
                status_code=300,
                content={
                    "error": f"Multiple players found matching '{name}'",
                    "message": "Please specify which player you want by adding the team parameter",
                    "matches": [{
                        **alt,
                        "full_context": f"{alt.get('display_name')} ({alt.get('position')}, {alt.get('team_abbr') or 'Retired'})"
                    } for alt in alternatives]
                }
            )
            
        if not player:
            raise HTTPException(status_code=404, detail=f"No player found matching '{name}'")
            
        # Get player ID and position
        player_id = player["gsis_id"]
        position = player.get("position", "")
        
        # Load weekly stats data
        seasons = [season] if season else list(range(2020, 2025))
        weekly_stats = await load_weekly_stats(seasons)
        
        # Filter data for this player
        player_stats = weekly_stats[weekly_stats['player_id'] == player_id] if 'player_id' in weekly_stats.columns else pd.DataFrame()
        
        # Apply additional filters
        if season:
            player_stats = player_stats[player_stats['season'] == season]
        if week and aggregate != AggregationType.CAREER:
            player_stats = player_stats[player_stats['week'] == week]
        if season_type:
            player_stats = player_stats[player_stats['season_type'] == season_type]
        if opponent:
            player_stats = player_stats[player_stats['opponent_team'] == opponent.upper()]
        if home_away:
            player_team = player.get("team_abbr", "")
            if home_away == "home":
                player_stats = player_stats[player_stats['recent_team'] == player_stats['opponent_team'].apply(lambda x: x != player_team)]
            else:  # away
                player_stats = player_stats[player_stats['recent_team'] == player_stats['opponent_team'].apply(lambda x: x == player_team)]
            
        # Apply situational filters if available - this requires play-by-play data
        situation_applied = False
        situational_stats = {}
        
        if situation:
            try:
                logger.info(f"Loading play-by-play data for situation filtering: {situation}")
                pbp_data = load_pbp_data()
                
                # Create a list of player IDs to search for in various PBP columns
                player_id_variations = [player_id]
                if 'gsis_it_id' in player:
                    player_id_variations.append(player['gsis_it_id'])
                
                # Extract situation-specific stats directly from play-by-play data
                try:
                    situational_stats = extract_situational_stats_from_pbp(
                        pbp_data=pbp_data,
                        player_id_variations=player_id_variations,
                        situation_type=situation.value,
                        season=season,
                        week=week
                    )
                    logger.info(f"Extracted situational stats for {situation.value}: {situational_stats}")
                except Exception as e:
                    logger.error(f"Error extracting situational stats: {e}")
                    situational_stats = {"error": str(e)}
                
                # If we found situation plays, mark as applied
                if situational_stats:
                    situation_applied = True
                    logger.info(f"Successfully extracted {situation} stats: {situational_stats}")
                    
                    # Force situational_stats to have at least one entry to ensure it's included in the response
                    if len(situational_stats) == 0:
                        situational_stats['play_count'] = 0
                    
                    # We still need weekly stats for games where this player was involved in this situation
                    # This is needed for aggregate-level endpoints
                    
                    # If games info is available, filter weekly stats to just those games
                    if 'games' in situational_stats and situational_stats['games'] > 0:
                        
                        # Identify games with situation plays - we'll add this to the response
                        situational_stats['situation_games'] = situational_stats['games']
                        
                        # For now, keep weekly stats as they were - we'll use our direct situational stats
                        # for the detailed breakdown. This allows us to maintain the aggregation mechanisms
                        # established elsewhere in the code.
                            
            except Exception as e:
                logger.error(f"Error applying situation filter: {str(e)}")
                # Don't raise an exception, just log the error and continue without situation filtering
            
        # Check if we have data
        if player_stats.empty:
            return {
                "player_id": player_id,
                "name": player["display_name"],
                "position": position,
                "aggregation": aggregate,
                "filters_applied": {
                    "season": season,
                    "week": week,
                    "season_type": season_type,
                    "opponent": opponent,
                    "home_away": home_away,
                    "situation": situation.value if situation else None
                },
                "stats": []
            }
            
        # Helper for sanitizing records
        def sanitize_record(record):
            if isinstance(record, dict):
                result = {}
                for k, v in record.items():
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v) or v > 1.7976931348623157e+308 or v < -1.7976931348623157e+308):
                        result[k] = None
                    elif pd.isna(v):
                        result[k] = None
                    elif isinstance(v, dict) or isinstance(v, list):
                        # Recursively sanitize nested structures
                        result[k] = sanitize_record(v)
                    else:
                        # Convert numpy types to Python native types
                        if isinstance(v, np.integer):
                            result[k] = int(v)
                        elif isinstance(v, np.floating):
                            if np.isnan(v) or np.isinf(v) or v > 1.7976931348623157e+308 or v < -1.7976931348623157e+308:
                                result[k] = None
                            else:
                                result[k] = float(v)
                        else:
                            result[k] = v
                return result
            elif isinstance(record, list):
                return [sanitize_record(item) for item in record]
            else:
                # Handle scalar values
                if isinstance(record, float) and (np.isnan(record) or np.isinf(record) or record > 1.7976931348623157e+308 or record < -1.7976931348623157e+308):
                    return None
                elif pd.isna(record):
                    return None
                elif isinstance(record, np.integer):
                    return int(record)
                elif isinstance(record, np.floating):
                    if np.isnan(record) or np.isinf(record) or record > 1.7976931348623157e+308 or record < -1.7976931348623157e+308:
                        return None
                    else:
                        return float(record)
                else:
                    return record
            
        # Process stats based on aggregation level
        stats_data = []

        # If a situation is specified, use play-by-play data for aggregation
        if situation:
            logger.info(f"Aggregating main stats from PBP for situation: {situation}")
            pbp_data = load_pbp_data()
            player_id_variations = [player_id]
            if 'gsis_it_id' in player:
                player_id_variations.append(player['gsis_it_id'])

            # Build a mask for all plays involving this player (reuse logic from extract_situational_stats_from_pbp)
            player_cols = [
                'passer_player_id', 'receiver_player_id', 'rusher_player_id',
                'lateral_receiver_player_id', 'lateral_rusher_player_id',
                'fumbled_1_player_id', 'fumbled_2_player_id'
            ]
            valid_cols = [col for col in player_cols if col in pbp_data.columns]
            player_plays_mask = False
            for pid in player_id_variations:
                for col in valid_cols:
                    player_plays_mask = player_plays_mask | (pbp_data[col] == pid)
            player_plays = pbp_data[player_plays_mask].copy()
            if season:
                player_plays = player_plays[player_plays['season'] == season]
            if week and aggregate != AggregationType.CAREER:
                player_plays = player_plays[player_plays['week'] == week]
            if season_type:
                player_plays = player_plays[player_plays['season_type'] == season_type]
            # Apply situation filter
            if situation.value == "red_zone":
                situation_plays = player_plays[player_plays['yardline_100'] <= 20]
            elif situation.value == "third_down":
                situation_plays = player_plays[player_plays['down'] == 3]
            elif situation.value == "fourth_down":
                situation_plays = player_plays[player_plays['down'] == 4]
            elif situation.value == "goal_line":
                situation_plays = player_plays[player_plays['yardline_100'] <= 5]
            elif situation.value == "two_minute_drill":
                two_min_mask = (
                    ((player_plays['qtr'] == 2) & (player_plays['half_seconds_remaining'] <= 120)) |
                    ((player_plays['qtr'] == 4) & (player_plays['half_seconds_remaining'] <= 120))
                )
                situation_plays = player_plays[two_min_mask]
            else:
                situation_plays = player_plays

            # If no plays, return empty stats
            if situation_plays.empty:
                stats_data = []
            else:
                # Ensure 'season_type' exists for grouping
                if 'season_type' not in situation_plays.columns:
                    situation_plays['season_type'] = 'REG'
                # Aggregation logic
                if aggregate == AggregationType.CAREER:
                    # All plays together
                    stats = get_position_specific_stats_from_pbp(
                        situation_plays, position, player_id=player_id
                    )
                    # Ensure all expected keys are present
                    if position == "QB":
                        for k in ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds", "completion_percentage", "yards_per_attempt"]:
                            stats.setdefault(k, 0)
                    elif position == "RB":
                        for k in ["rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds", "targets", "yards_per_carry", "yards_per_reception"]:
                            stats.setdefault(k, 0)
                    elif position in ["WR", "TE"]:
                        for k in ["targets", "receptions", "receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds", "yards_per_reception", "yards_per_target", "catch_rate"]:
                            stats.setdefault(k, 0)
                    stats_data.append(sanitize_record(stats))
                elif aggregate == AggregationType.SEASON:
                    # Group by season and season_type
                    grouped = situation_plays.groupby(['season', 'season_type'])
                    for (season_val, season_type_val), group in grouped:
                        stats = get_position_specific_stats_from_pbp(
                            group, position, player_id=player_id
                        )
                        # Ensure all expected keys are present
                        if position == "QB":
                            for k in ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds", "completion_percentage", "yards_per_attempt"]:
                                stats.setdefault(k, 0)
                        elif position == "RB":
                            for k in ["rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds", "targets", "yards_per_carry", "yards_per_reception"]:
                                stats.setdefault(k, 0)
                        elif position in ["WR", "TE"]:
                            for k in ["targets", "receptions", "receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds", "yards_per_reception", "yards_per_target", "catch_rate"]:
                                stats.setdefault(k, 0)
                        stats["season"] = int(season_val)
                        stats["season_type"] = season_type_val
                        stats_data.append(sanitize_record(stats))
                elif aggregate == AggregationType.WEEK:
                    # Group by season, week, and season_type
                    grouped = situation_plays.groupby(['season', 'week', 'season_type'])
                    for (season_val, week_val, season_type_val), group in grouped:
                        stats = get_position_specific_stats_from_pbp(
                            group, position, player_id=player_id
                        )
                        # Ensure all expected keys are present
                        if position == "QB":
                            for k in ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds", "completion_percentage", "yards_per_attempt"]:
                                stats.setdefault(k, 0)
                        elif position == "RB":
                            for k in ["rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds", "targets", "yards_per_carry", "yards_per_reception"]:
                                stats.setdefault(k, 0)
                        elif position in ["WR", "TE"]:
                            for k in ["targets", "receptions", "receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds", "yards_per_reception", "yards_per_target", "catch_rate"]:
                                stats.setdefault(k, 0)
                        stats["season"] = int(season_val)
                        stats["week"] = int(week_val)
                        stats["season_type"] = season_type_val
                        stats_data.append(sanitize_record(stats))
                # Sort as before
                if stats_data and aggregate == AggregationType.SEASON:
                    stats_data.sort(key=lambda x: (x.get('season', 0), x.get('season_type', '')), reverse=True)
                elif stats_data and aggregate == AggregationType.WEEK:
                    stats_data.sort(key=lambda x: (x.get('season', 0), x.get('week', 0)), reverse=True)
        else:
            # Default: use weekly_stats as before
            # Process stats based on aggregation level
            stats_data = []
            
            if aggregate == AggregationType.CAREER:
                # Aggregate stats across all seasons for a career total
                if not player_stats.empty:
                    # Group by season_type to aggregate regular season and postseason separately
                    grouped = player_stats.groupby('season_type')
                    
                    for season_type_val, group in grouped:
                        career_summary = {}
                        
                        # Common stats across positions
                        career_summary["season_type"] = season_type_val
                        career_summary["games_played"] = len(group)
                        
                        # Handle position-specific aggregations
                        if position == "QB":
                            # Sum counting stats
                            career_summary["passing_yards"] = float(group['passing_yards'].sum())
                            career_summary["passing_tds"] = int(group['passing_tds'].sum())
                            career_summary["interceptions"] = int(group['interceptions'].sum())
                            career_summary["rushing_yards"] = float(group['rushing_yards'].sum())
                            career_summary["rushing_tds"] = int(group['rushing_tds'].sum())
                            completions = group['completions'].sum()
                            attempts = group['attempts'].sum()
                            
                            # Calculate derived stats
                            career_summary["completion_percentage"] = float((completions / attempts * 100) if attempts > 0 else 0)
                            career_summary["yards_per_attempt"] = float(group['passing_yards'].sum() / max(attempts, 1))
                            
                            # Add advanced metrics if available
                            if 'passing_epa' in group.columns:
                                career_summary["passing_epa_per_play"] = float(group['passing_epa'].sum() / max(attempts, 1))
                            
                        elif position == "RB":
                            # Sum counting stats
                            career_summary["rushing_yards"] = float(group['rushing_yards'].sum())
                            career_summary["rushing_tds"] = int(group['rushing_tds'].sum())
                            career_summary["receptions"] = int(group['receptions'].sum())
                            career_summary["receiving_yards"] = float(group['receiving_yards'].sum())
                            career_summary["receiving_tds"] = int(group['receiving_tds'].sum())
                            career_summary["targets"] = int(group['targets'].sum())
                            
                            # Calculate derived stats
                            carries = group['carries'].sum()
                            career_summary["yards_per_carry"] = float(group['rushing_yards'].sum() / max(carries, 1))
                            career_summary["yards_per_reception"] = float(group['receiving_yards'].sum() / max(group['receptions'].sum(), 1))
                            
                            # Add advanced metrics if available
                            if 'rushing_epa' in group.columns:
                                career_summary["rushing_epa_per_play"] = float(group['rushing_epa'].sum() / max(carries, 1))
                            
                        elif position in ["WR", "TE"]:
                            # Sum counting stats
                            career_summary["targets"] = int(group['targets'].sum())
                            career_summary["receptions"] = int(group['receptions'].sum())
                            career_summary["receiving_yards"] = float(group['receiving_yards'].sum())
                            career_summary["receiving_tds"] = int(group['receiving_tds'].sum())
                            career_summary["rushing_yards"] = float(group['rushing_yards'].sum())
                            career_summary["rushing_tds"] = int(group['rushing_tds'].sum())
                            
                            # Calculate derived stats
                            targets = group['targets'].sum()
                            career_summary["yards_per_reception"] = float(group['receiving_yards'].sum() / max(group['receptions'].sum(), 1))
                            career_summary["yards_per_target"] = float(group['receiving_yards'].sum() / max(targets, 1))
                            career_summary["catch_rate"] = float(group['receptions'].sum() / max(targets, 1) * 100)
                            
                            # Add advanced metrics if available
                            if 'receiving_air_yards' in group.columns:
                                career_summary["air_yards_per_target"] = float(group['receiving_air_yards'].sum() / max(targets, 1))
                            if 'receiving_yards_after_catch' in group.columns:
                                career_summary["yards_after_catch_per_reception"] = float(group['receiving_yards_after_catch'].sum() / max(group['receptions'].sum(), 1))
                            if 'wopr' in group.columns:
                                career_summary["average_wopr"] = float(group['wopr'].mean())
                            
                        # Add fantasy points for all positions
                        if 'fantasy_points_ppr' in group.columns:
                            career_summary["total_fantasy_points"] = float(group['fantasy_points_ppr'].sum())
                            career_summary["fantasy_points_per_game"] = float(group['fantasy_points_ppr'].mean())
                            
                        stats_data.append(sanitize_record(career_summary))
                        
            elif aggregate == AggregationType.SEASON:
                # Aggregate stats by season
                # Group by season and season_type
                grouped = player_stats.groupby(['season', 'season_type'])
                
                for (season_val, season_type_val), group in grouped:
                    season_summary = {}
                    
                    # Common stats
                    season_summary["season"] = int(season_val)
                    season_summary["season_type"] = season_type_val
                    season_summary["games_played"] = len(group)
                    
                    # Handle position-specific aggregations
                    if position == "QB":
                        # Sum counting stats
                        season_summary["passing_yards"] = float(group['passing_yards'].sum())
                        season_summary["passing_tds"] = int(group['passing_tds'].sum())
                        season_summary["interceptions"] = int(group['interceptions'].sum())
                        season_summary["rushing_yards"] = float(group['rushing_yards'].sum())
                        season_summary["rushing_tds"] = int(group['rushing_tds'].sum())
                        completions = group['completions'].sum()
                        attempts = group['attempts'].sum()
                        
                        # Calculate derived stats
                        season_summary["completion_percentage"] = float((completions / attempts * 100) if attempts > 0 else 0)
                        season_summary["yards_per_attempt"] = float(group['passing_yards'].sum() / max(attempts, 1))
                        
                        # Add advanced metrics if available
                        if 'passing_epa' in group.columns:
                            season_summary["passing_epa_per_play"] = float(group['passing_epa'].sum() / max(attempts, 1))
                        
                    elif position == "RB":
                        # Sum counting stats
                        season_summary["rushing_yards"] = float(group['rushing_yards'].sum())
                        season_summary["rushing_tds"] = int(group['rushing_tds'].sum())
                        season_summary["receptions"] = int(group['receptions'].sum())
                        season_summary["receiving_yards"] = float(group['receiving_yards'].sum())
                        season_summary["receiving_tds"] = int(group['receiving_tds'].sum())
                        season_summary["targets"] = int(group['targets'].sum())
                        
                        # Calculate derived stats
                        carries = group['carries'].sum()
                        season_summary["yards_per_carry"] = float(group['rushing_yards'].sum() / max(carries, 1))
                        season_summary["yards_per_reception"] = float(group['receiving_yards'].sum() / max(group['receptions'].sum(), 1))
                        
                        # Add advanced metrics if available
                        if 'rushing_epa' in group.columns:
                            season_summary["rushing_epa_per_play"] = float(group['rushing_epa'].sum() / max(carries, 1))
                        
                    elif position in ["WR", "TE"]:
                        # Sum counting stats
                        season_summary["targets"] = int(group['targets'].sum())
                        season_summary["receptions"] = int(group['receptions'].sum())
                        season_summary["receiving_yards"] = float(group['receiving_yards'].sum())
                        season_summary["receiving_tds"] = int(group['receiving_tds'].sum())
                        season_summary["rushing_yards"] = float(group['rushing_yards'].sum())
                        season_summary["rushing_tds"] = int(group['rushing_tds'].sum())
                        
                        # Calculate derived stats
                        targets = group['targets'].sum()
                        season_summary["yards_per_reception"] = float(group['receiving_yards'].sum() / max(group['receptions'].sum(), 1))
                        season_summary["yards_per_target"] = float(group['receiving_yards'].sum() / max(targets, 1))
                        season_summary["catch_rate"] = float(group['receptions'].sum() / max(targets, 1) * 100)
                        
                        # Add advanced metrics if available
                        if 'receiving_air_yards' in group.columns:
                            season_summary["air_yards_per_target"] = float(group['receiving_air_yards'].sum() / max(targets, 1))
                        if 'receiving_yards_after_catch' in group.columns:
                            season_summary["yards_after_catch_per_reception"] = float(group['receiving_yards_after_catch'].sum() / max(group['receptions'].sum(), 1))
                        if 'wopr' in group.columns:
                            season_summary["average_wopr"] = float(group['wopr'].mean())
                        
                    # Add fantasy points for all positions
                    if 'fantasy_points_ppr' in group.columns:
                        season_summary["total_fantasy_points"] = float(group['fantasy_points_ppr'].sum())
                        season_summary["fantasy_points_per_game"] = float(group['fantasy_points_ppr'].mean())
                        
                    stats_data.append(sanitize_record(season_summary))
                    
            elif aggregate == AggregationType.WEEK:
                # Return week-by-week stats (no aggregation)
                for _, record in player_stats.iterrows():
                    stats_data.append(sanitize_record(record.to_dict()))
        
        # Sort the data appropriately
        if stats_data and aggregate == AggregationType.SEASON:
            stats_data.sort(key=lambda x: (x.get('season', 0), x.get('season_type', '')), reverse=True)
        elif stats_data and aggregate == AggregationType.WEEK:
            stats_data.sort(key=lambda x: (x.get('season', 0), x.get('week', 0)), reverse=True)
            
        # Build response with standard stats data and situational data if available
        response = {
            "player_id": player_id,
            "name": player["display_name"],
            "position": position,
            "team": player.get("team_abbr", ""),
            "aggregation": aggregate,
            "filters_applied": {
                "season": season,
                "week": week,
                "season_type": season_type,
                "opponent": opponent,
                "home_away": home_away,
                "situation": situation.value if situation else None,
                "situation_applied": situation_applied if situation else None
            },
            "stats": stats_data
        }
        
        # Always include situational stats if a situation was requested
        if situation:
            # Force log the situational stats
            logger.info(f"MANDATORY LOG - Situation applied: {situation_applied}")
            logger.info(f"MANDATORY LOG - Situational stats: {situational_stats}")
            
            # Ensure situational_stats is a dictionary
            if not isinstance(situational_stats, dict):
                situational_stats = {}
            
            # Add a placeholder if we have no actual stats to show
            if not situational_stats:
                situational_stats["note"] = f"No {situation.value} plays found for this player in the specified timeframe"
                
            # Clean up situational stats - convert NaN to None and format numbers
            clean_situation_stats = {}
            for k, v in situational_stats.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    clean_situation_stats[k] = None
                elif isinstance(v, (np.integer, np.floating)):
                    clean_situation_stats[k] = float(v) if isinstance(v, np.floating) else int(v)
                else:
                    clean_situation_stats[k] = v
            
            # Only add derived stats if we actually have the raw stats
            if 'passing_attempts' in clean_situation_stats or 'rushing_attempts' in clean_situation_stats or 'targets' in clean_situation_stats:
                # Add derived stats for QB
                if position == "QB" and 'passing_attempts' in clean_situation_stats and clean_situation_stats['passing_attempts'] > 0:
                    # Calculate completion percentage
                    if 'passing_completions' in clean_situation_stats:
                        completions = clean_situation_stats['passing_completions']
                        attempts = clean_situation_stats['passing_attempts']
                        clean_situation_stats['completion_percentage'] = round((completions / attempts) * 100, 1)
                    
                    # Calculate yards per attempt
                    if 'passing_yards' in clean_situation_stats:
                        clean_situation_stats['yards_per_attempt'] = round(clean_situation_stats['passing_yards'] / clean_situation_stats['passing_attempts'], 1)
                    
                    # Calculate EPA per play
                    if 'passing_epa' in clean_situation_stats:
                        clean_situation_stats['passing_epa_per_play'] = round(clean_situation_stats['passing_epa'] / clean_situation_stats['passing_attempts'], 3)
                
                # Add derived stats for RB
                if position == "RB" and 'rushing_attempts' in clean_situation_stats and clean_situation_stats['rushing_attempts'] > 0:
                    # Calculate yards per carry
                    if 'rushing_yards' in clean_situation_stats:
                        clean_situation_stats['yards_per_carry'] = round(clean_situation_stats['rushing_yards'] / clean_situation_stats['rushing_attempts'], 1)
                    
                    # Calculate EPA per rush
                    if 'rushing_epa' in clean_situation_stats:
                        clean_situation_stats['rushing_epa_per_play'] = round(clean_situation_stats['rushing_epa'] / clean_situation_stats['rushing_attempts'], 3)
                
                # Add derived stats for WR/TE
                if position in ["WR", "TE"] and 'targets' in clean_situation_stats and clean_situation_stats['targets'] > 0:
                    # Calculate catch rate
                    if 'receptions' in clean_situation_stats:
                        clean_situation_stats['catch_rate'] = round((clean_situation_stats['receptions'] / clean_situation_stats['targets']) * 100, 1)
                    
                    # Calculate yards per reception
                    if 'receiving_yards' in clean_situation_stats and 'receptions' in clean_situation_stats and clean_situation_stats['receptions'] > 0:
                        clean_situation_stats['yards_per_reception'] = round(clean_situation_stats['receiving_yards'] / clean_situation_stats['receptions'], 1)
                    
                    # Calculate yards per target
                    if 'receiving_yards' in clean_situation_stats:
                        clean_situation_stats['yards_per_target'] = round(clean_situation_stats['receiving_yards'] / clean_situation_stats['targets'], 1)
            
            # Add situational stats to response
            response["situational_stats"] = clean_situation_stats
            
            # Force add at least one stat if we don't have any
            if not response["situational_stats"] and situation_applied:
                response["situational_stats"] = {"note": f"No detailed {situation.value} stats found for this player"}
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting player stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting player stats: {str(e)}")

@router.get("/api/player/{name}/history")
@cache(expire=43200)  # 12 hours
async def get_player_history(
    name: str = Path(..., description="Player name (format: 'First Last')"),
    type: HistoryType = Query(..., description="Type of history data to retrieve"),
    season: Optional[int] = Query(None, description="Filter by specific season (optional)"),
    week: Optional[int] = Query(None, description="Filter by specific week (optional, for depth charts and injuries)")
):
    """Get player history data for roster, depth chart, or injury history."""
    try:
        # Resolve player first
        player, alternatives = await resolve_player(name, season)
        
        if not player and alternatives:
            return JSONResponse(
                status_code=300,
                content={
                    "error": f"Multiple players found matching '{name}'",
                    "matches": [{
                        **alt,
                        "full_context": f"{alt.get('display_name')} ({alt.get('position')}, {alt.get('team_abbr') or 'Retired'})"
                    } for alt in alternatives]
                }
            )
            
        if not player:
            raise HTTPException(status_code=404, detail=f"No player found matching '{name}'")
            
        # Get player ID
        player_id = player["gsis_id"]
        
        # Load appropriate dataset based on history type
        seasons = [season] if season else list(range(2020, 2025))
        
        # Helper function to sanitize records
        def sanitize_record(record):
            if isinstance(record, dict):
                result = {}
                for k, v in record.items():
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v) or v > 1.7976931348623157e+308 or v < -1.7976931348623157e+308):
                        result[k] = None
                    elif pd.isna(v):
                        result[k] = None
                    elif isinstance(v, dict) or isinstance(v, list):
                        # Recursively sanitize nested structures
                        result[k] = sanitize_record(v)
                    else:
                        # Convert numpy types to Python native types
                        if isinstance(v, np.integer):
                            result[k] = int(v)
                        elif isinstance(v, np.floating):
                            if np.isnan(v) or np.isinf(v) or v > 1.7976931348623157e+308 or v < -1.7976931348623157e+308:
                                result[k] = None
                            else:
                                result[k] = float(v)
                        else:
                            result[k] = v
                return result
            elif isinstance(record, list):
                return [sanitize_record(item) for item in record]
            else:
                # Handle scalar values
                if isinstance(record, float) and (np.isnan(record) or np.isinf(record) or record > 1.7976931348623157e+308 or record < -1.7976931348623157e+308):
                    return None
                elif pd.isna(record):
                    return None
                elif isinstance(record, np.integer):
                    return int(record)
                elif isinstance(record, np.floating):
                    if np.isnan(record) or np.isinf(record) or record > 1.7976931348623157e+308 or record < -1.7976931348623157e+308:
                        return None
                    else:
                        return float(record)
                else:
                    return record
        
        # Process the requested history type
        history_data = []
        
        if type == HistoryType.ROSTER:
            # Load roster data
            rosters = await load_rosters(seasons)
            player_rosters = rosters[rosters['gsis_id'] == player_id] if 'gsis_id' in rosters.columns else pd.DataFrame()
            
            # Filter by season if specified
            if season:
                player_rosters = player_rosters[player_rosters['season'] == season]
                
            # Include only relevant fields (normalize data)
            if not player_rosters.empty:
                selected_columns = ['season', 'team', 'status', 'jersey_number', 'depth_chart_position']
                # Filter to include only columns that exist in the dataframe
                existing_columns = [col for col in selected_columns if col in player_rosters.columns]
                filtered_rosters = player_rosters[existing_columns]
                
                for record in filtered_rosters.to_dict('records'):
                    sanitized_record = sanitize_record(record)
                    history_data.append(sanitized_record)
                    
        elif type == HistoryType.DEPTH:
            # Load depth chart data
            depth_charts = await load_depth_charts(seasons)
            player_depth = depth_charts[depth_charts['gsis_id'] == player_id] if 'gsis_id' in depth_charts.columns else pd.DataFrame()
            
            # Apply filters
            if season:
                player_depth = player_depth[player_depth['season'] == season]
            if week:
                player_depth = player_depth[player_depth['week'] == week]
                
            # Include only relevant fields (normalize data)
            if not player_depth.empty:
                selected_columns = ['season', 'week', 'depth_team', 'depth_position', 'depth_order', 'status']
                # Filter to include only columns that exist in the dataframe
                existing_columns = [col for col in selected_columns if col in player_depth.columns]
                filtered_depth = player_depth[existing_columns]
                
                for record in filtered_depth.to_dict('records'):
                    sanitized_record = sanitize_record(record)
                    history_data.append(sanitized_record)
                    
        elif type == HistoryType.INJURY:
            # Load injury data
            injuries = await load_injuries(seasons)
            player_injuries = injuries[injuries['gsis_id'] == player_id] if 'gsis_id' in injuries.columns else pd.DataFrame()
            
            # Apply filters
            if season:
                player_injuries = player_injuries[player_injuries['season'] == season]
            if week:
                player_injuries = player_injuries[player_injuries['week'] == week]
                
            # Include only relevant fields (normalize data)
            if not player_injuries.empty:
                selected_columns = ['season', 'week', 'team', 'injury_type', 'practice_status', 'game_status']
                # Filter to include only columns that exist in the dataframe
                existing_columns = [col for col in selected_columns if col in player_injuries.columns]
                filtered_injuries = player_injuries[existing_columns]
                
                for record in filtered_injuries.to_dict('records'):
                    sanitized_record = sanitize_record(record)
                    history_data.append(sanitized_record)
        
        # Sort the data by appropriate fields
        if history_data:
            if 'season' in history_data[0]:
                history_data.sort(key=lambda x: (x.get('season', 0), x.get('week', 0)), reverse=True)
        
        return {
            "player_id": player_id,
            "name": player["display_name"],
            "position": player.get("position", ""),
            "history_type": type,
            "records": history_data
        }
        
    except Exception as e:
        logger.error(f"Error getting player history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting player history: {str(e)}")

@router.get("/api/player/{name}")
@cache(expire=43200)  # 12 hours
async def get_player_information(
    name: str = Path(..., description="Player name (format: 'First Last')"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent available)", ge=1920, le=2024)
):
    """Get basic player information and career stats. For detailed data, use the dedicated endpoints.
    
    Note: This is a simplified endpoint that returns only player info and career stats.
    For more detailed or filtered data, use the following dedicated endpoints:
    - /api/player/{name}/info - For basic player information
    - /api/player/{name}/history - For roster, depth chart, or injury history
    - /api/player/{name}/stats - For statistics with flexible aggregation options
    """
    try:
        # Resolve player first - use await since resolve_player is now async
        player, alternatives = await resolve_player(name, season)
        
        if not player and alternatives:
            # Enhanced response for LLMs with more context
            return JSONResponse(
                status_code=300,
                content={
                    "error": f"Multiple players found matching '{name}'",
                    "suggestion": "Please specify which player you mean by providing additional context like team, position, or active/retired status",
                    "matches": [{
                        **alt,
                        "active_status": "Active" if alt.get("team_abbr") else "Inactive/Retired",
                        "full_context": f"{alt.get('display_name')} ({alt.get('position')}, {alt.get('team_abbr') or 'Retired'}, {alt.get('years_of_experience', 0)} years exp.)"
                    } for alt in alternatives]
                }
            )
            
        if not player:
            raise HTTPException(status_code=404, detail=f"No player found matching '{name}'")
            
        # Get player ID and headshot URL
        player_id = player["gsis_id"]
        
        # Use the headshot URL directly from the player data
        headshot_url = player.get("headshot", "")
        
        # Load stats data for career stats
        seasons = [season] if season else list(range(2020, 2025))
        weekly_stats = await load_weekly_stats(seasons)
        
        # Filter data for this player
        player_stats = weekly_stats[weekly_stats['player_id'] == player_id] if 'player_id' in weekly_stats.columns else pd.DataFrame()
        
        # Process career stats - handle potential NaN values
        career_stats = {}
        position = player.get("position", "")
        if not player_stats.empty:
            fantasy_pts_per_game = player_stats['fantasy_points_ppr'].mean()
            fantasy_pts_total = player_stats['fantasy_points_ppr'].sum()
            
            career_stats = {
                "games_played": len(player_stats),
                "fantasy_points_per_game": None if pd.isna(fantasy_pts_per_game) else float(fantasy_pts_per_game),
                "total_fantasy_points": None if pd.isna(fantasy_pts_total) else float(fantasy_pts_total)
            }
            
            # Add position-specific stats with NaN handling
            if position == "QB":
                passing_yards = player_stats['passing_yards'].sum()
                passing_tds = player_stats['passing_tds'].sum()
                interceptions = player_stats['interceptions'].sum()
                completions = player_stats['completions'].sum()
                attempts = player_stats['attempts'].sum()
                rushing_yards = player_stats['rushing_yards'].sum()
                rushing_tds = player_stats['rushing_tds'].sum()
                
                completion_percentage = (completions / attempts * 100) if attempts > 0 else 0
                
                career_stats.update({
                    "passing_yards": None if pd.isna(passing_yards) else float(passing_yards),
                    "passing_tds": None if pd.isna(passing_tds) else int(passing_tds),
                    "interceptions": None if pd.isna(interceptions) else int(interceptions),
                    "completion_percentage": None if pd.isna(completion_percentage) else float(completion_percentage),
                    "rushing_yards": None if pd.isna(rushing_yards) else float(rushing_yards),
                    "rushing_tds": None if pd.isna(rushing_tds) else int(rushing_tds)
                })
            elif position in ["RB", "WR", "TE"]:
                rushing_yards = player_stats['rushing_yards'].sum()
                rushing_tds = player_stats['rushing_tds'].sum()
                receptions = player_stats['receptions'].sum()
                receiving_yards = player_stats['receiving_yards'].sum()
                receiving_tds = player_stats['receiving_tds'].sum()
                targets = player_stats['targets'].sum()
                
                career_stats.update({
                    "rushing_yards": None if pd.isna(rushing_yards) else float(rushing_yards),
                    "rushing_tds": None if pd.isna(rushing_tds) else int(rushing_tds),
                    "receptions": None if pd.isna(receptions) else int(receptions),
                    "receiving_yards": None if pd.isna(receiving_yards) else float(receiving_yards),
                    "receiving_tds": None if pd.isna(receiving_tds) else int(receiving_tds),
                    "targets": None if pd.isna(targets) else int(targets)
                })
        
        # Build simplified response with just player info and career stats
        response = {
            "player_info": {
                "player_id": player_id,
                "name": player["display_name"],
                "position": position,
                "team": player.get("team_abbr", ""),
                "age": calculate_age(player["birth_date"]) if "birth_date" in player else None,
                "experience": player.get("years_of_experience", None),
                "college": player.get("college_name", None),
                "height": player.get("height", None),
                "weight": player.get("weight", None),
                "headshot_url": headshot_url
            },
            "career_stats": career_stats,
            "available_endpoints": {
                "info": f"/api/player/{name}/info",
                "history": f"/api/player/{name}/history?type=roster|depth|injury",
                "stats": f"/api/player/{name}/stats?aggregate=career|season|week",
                "headshot": f"/api/player/{name}/headshot",
                "gamelog": f"/api/player/{name}/gamelog",
                "career": f"/api/player/{name}/career"
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting player information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting player information: {str(e)}")

@router.get("/api/player/{name}/headshot")
@cache(expire=86400)
async def get_headshot(
    name: str = Path(..., description="Player name"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent)") 
):
    """Get URL for player's headshot image."""
    try:
        # Resolve player - use await since resolve_player is now async
        player, alternatives = await resolve_player(name, season)
        
        if not player and not alternatives:
            raise HTTPException(status_code=404, detail=f"No player found matching '{name}'")
        
        if not player and alternatives:
            return JSONResponse(
                status_code=300, 
                content={
                    "error": f"Multiple players found matching '{name}'",
                    "matches": alternatives
                }
            )
        
        # Use the headshot URL directly from the player data
        headshot_url = player.get("headshot", "")
        
        return {
            "player_id": player["gsis_id"],
            "player_name": player["display_name"],
            "team": player.get("team_abbr", ""),
            "position": player["position"],
            "headshot_url": headshot_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting player headshot: {str(e)}")

@router.get("/api/player/{name}/career")
@cache(expire=43200)  # 12 hours
async def get_career_stats_endpoint(
    name: str = Path(..., description="Player name")
):
    """Get career stats for a player across all available seasons."""
    try:
        career = await get_player_career_stats(name)
        
        if "error" in career and "matches" in career:
            return JSONResponse(status_code=300, content=career)
            
        return career
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting career stats: {str(e)}")

@router.get("/api/player/{name}/gamelog")
async def get_gamelog_endpoint(
    name: str = Path(..., description="Player name"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent)")
):
    """Get game-by-game stats for a player."""
    try:
        gamelog = await get_player_game_log(name, season)
        
        if "error" in gamelog and "matches" in gamelog:
            return JSONResponse(status_code=300, content=gamelog)
            
        return gamelog
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting game log: {str(e)}")

@router.get("/api/player/{name}/situation/{situation_type}")
async def get_stats_by_situation_endpoint(
    name: str = Path(..., description="Player name"),
    situation_type: str = Path(..., description="Situation type (red_zone, third_down, fourth_down, goal_line, two_minute_drill, etc.)"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent)")
):
    """Get player stats for specific game situations."""
    try:
        stats = await get_situation_stats(name, situation_type, season)
        
        if "error" in stats and "matches" in stats:
            return JSONResponse(status_code=300, content=stats)
            
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting situation stats: {str(e)}")

@router.get("/api/player/{name}/on-field")
async def get_player_with_other_on_field_endpoint(
    name: str = Path(..., description="Primary player name"),
    other_player: str = Query(..., description="Other player to check on/off field status"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent)"),
    week: Optional[int] = Query(None, description="Filter by week number"),
    on_field: bool = Query(True, description="True for stats when other player is on field, False for off field")
):
    """Get player performance when another player is on/off the field."""
    try:
        stats = await get_player_on_field_stats(
            player_name=name,
            other_player_name=other_player,
            season=season,
            week=week,
            on_field=on_field
        )
        
        if "error" in stats and "matches" in stats:
            return JSONResponse(status_code=300, content=stats)
            
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting on-field stats: {str(e)}")