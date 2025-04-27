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
    situations: Optional[str] = Query(None, description="Comma-separated list of situation types (e.g., 'red_zone,third_down')")
):
    """Get player statistics with flexible aggregation options and situational filters.
    
    - aggregate: Choose between career totals, season totals, or week-by-week breakdown
    - season/week/season_type: Filter by season, week, or season type
    - opponent: Filter stats against a specific opponent
    - home_away: Filter for home or away games only
    - situations: Filter for specific game situations like red zone, third down, etc.
    """
    # Define sanitize_record helper function here so it's available in all code paths
    def sanitize_record(record):
        result = {}
        # Ensure input is a dictionary-like object
        if not hasattr(record, 'items'):
             logger.warning(f"Attempted to sanitize non-dictionary record: {type(record)}")
             return {} # Return empty dict if not a dict
        for k, v in record.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                result[k] = None
            elif pd.isna(v):
                result[k] = None
            else:
                # Convert numpy types to Python native types
                if isinstance(v, np.integer):
                    result[k] = int(v)
                elif isinstance(v, np.floating):
                    result[k] = float(v)
                else:
                    result[k] = v
        return result

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
                # Home game: player's team is NOT the opponent team in the weekly stats context 
                # (assuming 'opponent_team' is always the non-home team listed)
                # This needs careful verification based on how weekly_stats is structured.
                # Assuming 'opponent_team' refers to the team played *against*.
                # A player is home if their team is NOT the opponent listed for that game row.
                # This logic might be flawed if 'opponent_team' isn't consistently the away team.
                # Safer approach might be needed if game location data is available directly.
                logger.warning("Home/Away filter logic relies on 'opponent_team' structure and might be inaccurate.")
                # Example: If player is KC and opponent_team is LV, it's a home game.
                # If player is KC and opponent_team is KC, it's an away game (opponent played at home)
                # This assumes weekly stats always list the opponent relative to the player's game location.
                # Let's adjust based on a hypothetical 'game_location' column if it exists
                # Or assume opponent_team is the actual opponent, regardless of home/away.
                # If player's team = recent_team for the stat line, and opponent = opponent_team
                # Home game = game played AT player's team's home stadium. Need stadium/location info.
                # Fallback: assume 'recent_team' is the team the player played FOR that week.
                # Home game: player's team hosted. Need 'home_team' column.
                if 'home_team' in player_stats.columns:
                     player_stats = player_stats[player_stats['home_team'] == player_stats['recent_team']]
                else:
                     logger.warning("Cannot apply 'home' filter accurately without 'home_team' column.")
                     # Applying potentially incorrect logic as a fallback:
                     # player_stats = player_stats[player_stats['opponent_team'] != player_stats['recent_team']] # Highly speculative
            else:  # away
                if 'away_team' in player_stats.columns:
                    player_stats = player_stats[player_stats['away_team'] == player_stats['recent_team']]
                else:
                     logger.warning("Cannot apply 'away' filter accurately without 'away_team' column.")
                     # Applying potentially incorrect logic as fallback:
                     # player_stats = player_stats[player_stats['opponent_team'] == player_stats['recent_team']] # Highly speculative

        # Process stats based on aggregation level
        stats_data = []
        valid_situations_applied = [] # Keep track of applied situations

        # If situations are specified, use play-by-play data for aggregation
        # Otherwise, use the pre-filtered weekly stats
        if situations:
            # Manually split the comma-separated string
            situation_list = [s.strip().lower() for s in situations.split(',') if s.strip()]
            if not situation_list:
                 logger.warning("Received empty or whitespace-only situations parameter.")
                 # Treat as if situations parameter was absent
                 situations = None # Clear situations variable to fall back to weekly stats logic
            else:
                logger.info(f"Aggregating stats from PBP for situations: {situation_list}")
                try:
                    pbp_data = load_pbp_data()
                    if pbp_data.empty:
                        raise FileNotFoundError("PBP data could not be loaded or is empty.")

                    player_id_variations = [player_id]
                    if 'gsis_it_id' in player and player['gsis_it_id']:
                        player_id_variations.append(player['gsis_it_id'])

                    # Build a mask for all plays involving this player
                    player_cols = [
                        'passer_player_id', 'receiver_player_id', 'rusher_player_id',
                        'lateral_receiver_player_id', 'lateral_rusher_player_id',
                        'fumbled_1_player_id', 'fumbled_2_player_id', 'sack_player_id',
                        'pass_defense_1_player_id', 'pass_defense_2_player_id',
                        'interception_player_id', 'tackle_for_loss_1_player_id',
                        'tackle_for_loss_2_player_id', 'qb_hit_1_player_id', 'qb_hit_2_player_id'
                    ]
                    valid_player_cols = [col for col in player_cols if col in pbp_data.columns]
                    player_plays_mask = pd.Series(False, index=pbp_data.index)
                    for pid_var in player_id_variations:
                        if pid_var:
                            for col in valid_player_cols:
                                player_plays_mask |= (pbp_data[col].fillna('') == pid_var)
                    
                    if not player_plays_mask.any():
                         logger.warning(f"No PBP plays found involving player {player_id}")
                         player_plays = pd.DataFrame(columns=pbp_data.columns) # Empty DataFrame
                    else:
                         player_plays = pbp_data[player_plays_mask].copy()
                         logger.info(f"Found {len(player_plays)} total plays involving player {player_id}")

                    # Apply season/week/season_type filters to PBP data
                    if season:
                        if 'season' in player_plays.columns:
                            player_plays = player_plays[player_plays['season'] == season]
                        else:
                            logger.warning("Cannot filter PBP by season - 'season' column missing.")
                    if week and aggregate != AggregationType.CAREER:
                         if 'week' in player_plays.columns:
                             player_plays = player_plays[player_plays['week'] == week]
                         else:
                              logger.warning("Cannot filter PBP by week - 'week' column missing.")
                    if season_type:
                        if 'season_type' in player_plays.columns:
                            player_plays = player_plays[player_plays['season_type'] == season_type]
                        else:
                            logger.warning("Cannot filter PBP by season_type - 'season_type' column missing.")
                    
                    # If after basic filtering, no plays remain, set to empty
                    if player_plays.empty and player_plays_mask.any():
                        logger.info("No plays found for player after season/week/type filtering.")
                        situation_plays = pd.DataFrame(columns=player_plays.columns)
                    
                    # Apply situation filters ONLY if player_plays is not empty
                    elif not player_plays.empty:
                        combined_situation_mask = pd.Series(False, index=player_plays.index)
                        # Use SituationType enum values for validation
                        valid_situation_keys = SituationType._value2member_map_.keys() 

                        # Use the split list here
                        for situation_str in situation_list: 
                            situation_mask = pd.Series(False, index=player_plays.index)
                            # situation_key is already lower and stripped from list comprehension
                            situation_key = situation_str 
                            applied_flag = False

                            if situation_key not in valid_situation_keys:
                                logger.warning(f"Unsupported or invalid situation type requested: {situation_str}")
                                continue # Skip invalid situation

                            # Apply filters based on valid situation_key
                            if situation_key == SituationType.RED_ZONE.value:
                                if 'yardline_100' in player_plays.columns:
                                    situation_mask = player_plays['yardline_100'] <= 20
                                    applied_flag = True
                            elif situation_key == SituationType.THIRD_DOWN.value:
                                if 'down' in player_plays.columns:
                                    situation_mask = player_plays['down'] == 3
                                    applied_flag = True
                            elif situation_key == SituationType.FOURTH_DOWN.value:
                                if 'down' in player_plays.columns:
                                    situation_mask = player_plays['down'] == 4
                                    applied_flag = True
                            elif situation_key == SituationType.GOAL_LINE.value:
                                if 'yardline_100' in player_plays.columns:
                                    situation_mask = player_plays['yardline_100'] <= 5
                                    applied_flag = True
                            elif situation_key == SituationType.TWO_MINUTE_DRILL.value:
                                if 'qtr' in player_plays.columns and 'half_seconds_remaining' in player_plays.columns and 'game_seconds_remaining' in player_plays.columns:
                                    two_min_mask = (
                                        ((player_plays['qtr'] == 2) & (player_plays['half_seconds_remaining'] <= 120)) |
                                        ((player_plays['qtr'] == 4) & (player_plays['game_seconds_remaining'] <= 120))
                                    )
                                    situation_mask = two_min_mask
                                    applied_flag = True
                            
                            if applied_flag:
                                logger.info(f"Applying filter for situation: {situation_key}")
                                combined_situation_mask |= situation_mask.fillna(False)
                                if situation_str not in valid_situations_applied: # Add original string
                                    valid_situations_applied.append(situation_str)
                            else:
                                logger.warning(f"Could not apply filter for situation '{situation_key}' due to missing PBP columns.")

                        # Filter plays by the combined situation mask if any valid situations were applied
                        if not valid_situations_applied:
                            # Use situation_list in log message
                            logger.error(f"None of the requested situations could be applied or were valid: {situation_list}")
                            # Set situation_plays to empty DF, stats will be calculated as zero
                            situation_plays = pd.DataFrame(columns=player_plays.columns)
                        elif not combined_situation_mask.any():
                            logger.info(f"No plays found matching the requested valid situations: {valid_situations_applied}")
                            situation_plays = pd.DataFrame(columns=player_plays.columns) # Empty dataframe
                        else:
                            situation_plays = player_plays[combined_situation_mask]
                            logger.info(f"Found {len(situation_plays)} plays after applying situation filters: {valid_situations_applied}")
                    
                    # If PBP data was initially empty or no plays found for player, set to empty
                    else: 
                         situation_plays = pd.DataFrame(columns=pbp_data.columns if not pbp_data.empty else [])


                    # AGGREGATION LOGIC USING situation_plays
                    if situation_plays.empty:
                        stats_data = [] # Ensure empty list if no plays found
                    else:
                        # Ensure 'season_type' exists for grouping (important for PBP)
                        if 'season_type' not in situation_plays.columns:
                            # Attempt to infer season_type based on quarter/OT if possible
                            # For simplicity, defaulting to REG if missing
                            logger.warning("PBP data missing 'season_type', defaulting to 'REG' for aggregation.")
                            situation_plays['season_type'] = 'REG' 
                        
                        # Aggregation logic remains largely the same, but operates on 'situation_plays'
                        if aggregate == AggregationType.CAREER:
                            stats = get_position_specific_stats_from_pbp(
                                situation_plays, position, player_id=player_id
                            )
                            # Simplified default setting
                            stats.setdefault("plays", len(situation_plays)) 
                            stats_data.append(sanitize_record(stats))
                        elif aggregate == AggregationType.SEASON:
                            # Group by season and season_type
                            grouped = situation_plays.groupby(['season', 'season_type'])
                            for (season_val, season_type_val), group in grouped:
                                # Select only numeric columns for aggregation, excluding identifiers
                                numeric_cols = group.select_dtypes(include=np.number).columns.tolist()
                                # Remove identifiers that shouldn't be summed
                                cols_to_sum = [
                                    col for col in numeric_cols 
                                    if col not in ['season', 'week'] # Add other non-summable numeric IDs if they exist
                                ]
                                
                                # Sum the numeric columns
                                season_summary = group[cols_to_sum].sum(numeric_only=True).to_dict()
                                
                                # Add back the group identifiers
                                season_summary["season"] = int(season_val)
                                season_summary["season_type"] = season_type_val
                                
                                # Add games played count (number of weeks in the group)
                                season_summary["games_played"] = group['week'].nunique() if 'week' in group.columns else len(group)
                                
                                # Optionally, recompute rate stats if applicable (e.g., completion_percentage)
                                # Example:
                                # if 'completions' in season_summary and 'attempts' in season_summary and season_summary['attempts'] > 0:
                                #     season_summary['completion_percentage'] = (season_summary['completions'] / season_summary['attempts']) * 100
                                # else:
                                #     season_summary['completion_percentage'] = 0.0
                                
                                stats_data.append(sanitize_record(season_summary))
                        elif aggregate == AggregationType.WEEK:
                            # Group by season, week, and season_type
                            grouped = situation_plays.groupby(['season', 'week', 'season_type'])
                            for (season_val, week_val, season_type_val), group in grouped:
                                stats = get_position_specific_stats_from_pbp(
                                    group, position, player_id=player_id
                                )
                                stats["season"] = int(season_val)
                                stats["week"] = int(week_val)
                                stats["season_type"] = season_type_val
                                stats.setdefault("plays", len(group))
                                stats_data.append(sanitize_record(stats))

                except FileNotFoundError as e:
                     logger.error(f"PBP data file not found for situation processing: {e}")
                     # Return empty stats, but indicate error in logs/potentially response
                     stats_data = [] # Or potentially raise HTTPException? For now, return empty.
                except KeyError as e:
                     logger.error(f"Missing expected column in PBP data during situation processing: {e}")
                     stats_data = [] # Or raise HTTPException
                except Exception as e:
                     logger.exception(f"Error processing situations from PBP data: {e}")
                     stats_data = [] # Or raise HTTPException

        else:
            # Original logic using pre-filtered weekly_stats if no situations requested
            if player_stats.empty:
                 # If initial weekly stats filtering resulted in empty, return empty stats
                 logger.info("No weekly stats found matching non-situational filters.")
                 stats_data = []
            elif aggregate == AggregationType.CAREER:
                # Aggregate all stats together from the weekly data
                # Need to implement aggregation from weekly_stats DataFrame
                # This requires defining which columns to sum/average from weekly_stats
                logger.warning("Career aggregation from weekly_stats is not fully implemented.")
                # Placeholder: return summary of first record or empty
                career_summary = {} # TODO: Implement weekly stats aggregation
                stats_data.append(sanitize_record(career_summary))

            elif aggregate == AggregationType.SEASON:
                # Group by season and season_type from weekly data
                 if 'season_type' not in player_stats.columns:
                      player_stats['season_type'] = 'REG' # Default if missing
                 grouped = player_stats.groupby(['season', 'season_type'])
                 for (season_val, season_type_val), group in grouped:
                     # Aggregate stats within each season group
                     # Select only numeric columns for aggregation, excluding identifiers
                     numeric_cols = group.select_dtypes(include=np.number).columns.tolist()
                     # Remove identifiers that shouldn't be summed
                     cols_to_sum = [
                         col for col in numeric_cols 
                         if col not in ['season', 'week'] # Add other non-summable numeric IDs if they exist
                     ]
                     
                     # Sum the numeric columns
                     season_summary = group[cols_to_sum].sum(numeric_only=True).to_dict()
                     
                     # Add back the group identifiers
                     season_summary["season"] = int(season_val)
                     season_summary["season_type"] = season_type_val
                     
                     # Add games played count (number of weeks in the group)
                     season_summary["games_played"] = group['week'].nunique() if 'week' in group.columns else len(group)
                     
                     # Optionally, recompute rate stats if applicable (e.g., completion_percentage)
                     # Example:
                     # if 'completions' in season_summary and 'attempts' in season_summary and season_summary['attempts'] > 0:
                     #     season_summary['completion_percentage'] = (season_summary['completions'] / season_summary['attempts']) * 100
                     # else:
                     #     season_summary['completion_percentage'] = 0.0
                     
                     stats_data.append(sanitize_record(season_summary))
                     
            elif aggregate == AggregationType.WEEK:
                # Return week-by-week stats (no aggregation needed from weekly data)
                for _, record in player_stats.iterrows():
                    stats_data.append(sanitize_record(record.to_dict()))
        
        # Sort the data appropriately regardless of source (PBP or weekly)
        if stats_data and aggregate == AggregationType.SEASON:
            stats_data.sort(key=lambda x: (x.get('season', 0), x.get('season_type', '')), reverse=True)
        elif stats_data and aggregate == AggregationType.WEEK:
            stats_data.sort(key=lambda x: (x.get('season', 0), x.get('week', 0)), reverse=True)
            
        # Build response
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
                "situations_requested": situation_list if situations else None,
                "situations_applied": valid_situations_applied # List of successfully applied situations
            },
            "stats": stats_data
        }
        
        return response
        
    except Exception as e:
        logger.exception(f"Error getting player stats for {name}: {str(e)}") # Use logger.exception for traceback
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
            result = {}
            for k, v in record.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    result[k] = None
                elif pd.isna(v):
                    result[k] = None
                else:
                    # Convert numpy types to Python native types
                    if isinstance(v, np.integer):
                        result[k] = int(v)
                    elif isinstance(v, np.floating):
                        result[k] = float(v)
                    else:
                        result[k] = v
            return result
        
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
                # Define the source columns we actually need from the Parquet file
                source_columns = ['season', 'week', 'team', 'report_primary_injury', 
                                  'practice_primary_injury', 'practice_status', 'report_status']
                
                # Filter to include only columns that exist in the dataframe
                existing_source_columns = [col for col in source_columns if col in player_injuries.columns]
                filtered_injuries = player_injuries[existing_source_columns]

                for record in filtered_injuries.to_dict('records'):
                    # Start with a sanitized version of the record
                    sanitized_record = sanitize_record(record)
                    
                    # Determine injury description
                    injury_desc = sanitized_record.get('report_primary_injury')
                    if not injury_desc or pd.isna(injury_desc):
                        injury_desc = sanitized_record.get('practice_primary_injury')

                    # Create the final record structure for the API response
                    final_record = {
                        'season': sanitized_record.get('season'),
                        'week': sanitized_record.get('week'),
                        'team': sanitized_record.get('team'),
                        'injury_description': injury_desc, # Use the derived injury description
                        'practice_status': sanitized_record.get('practice_status'),
                        'game_status': sanitized_record.get('report_status') # Map report_status to game_status
                    }
                    
                    # Ensure None values for missing fields rather than NaN or missing keys
                    final_record = {k: (None if pd.isna(v) else v) for k, v in final_record.items()}
                    
                    history_data.append(final_record)
        
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