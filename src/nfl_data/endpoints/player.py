"""Player-related endpoints for the NFL Data API."""

import logging
from typing import Dict, List, Optional, Any, Literal, Union
from fastapi import APIRouter, Query, Path, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from enum import Enum
from datetime import datetime, date
import pandas as pd
import numpy as np
from pathlib import Path as PathLib
import importlib.util





from fastapi_cache.decorator import cache

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


from src.nfl_data.endpoints.player_stats import calculate_player_stats, get_top_players
from src.nfl_data.stats_helpers import resolve_player, get_player_game_log, calculate_age

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

@router.get("/api/{team}/{name}/info")
@cache(expire=43200)  # 12 hours
async def get_player_info(
    team: str = Path(..., description="Team abbreviation e.g. KC"),
    name: str = Path(..., description="Player name (format: 'First Last')"),
    season: Optional[int] = Query(None, description="Optional: NFL season year for context (defaults to most recent available)", ge=1920, le=2024)
):
    """Get basic player information including ID, name, position, team, physical attributes, etc.
    
    Uses both team and name for more accurate player resolution.
    """
    try:
        # Resolve player using name, team, and optionally season
        player, alternatives = resolve_player(name=name, team=team.upper(), season=season)
        
        # Handling alternatives when team *is* provided is slightly different
        # If the exact team/name combo didn't work, but alternatives exist, it's still ambiguous
        if not player and alternatives:
            # Suggest alternatives, emphasizing the team context might be wrong or name slightly off
            return JSONResponse(
                status_code=300,
                content={
                    "error": f"Ambiguous player: Multiple matches found for '{name}' even with team '{team.upper()}' constraint.",
                    "suggestion": "Verify the player name spelling and the team abbreviation. The player might have played for a different team in the specified season, or is currently inactive.",
                    "matches": [
                        {
                            **alt,
                            "active_status": "Active" if alt.get("team_abbr") else "Inactive/Retired",
                            "full_context": f"{alt.get('display_name')} ({alt.get('position')}, {alt.get('team_abbr') or 'Retired'}, {alt.get('years_of_experience', 0)} years exp.)"
                        } for alt in alternatives
                    ]
                }
            )
            
        if not player:
            # No player found at all for this name/team combo
            raise HTTPException(status_code=404, detail=f"No player found matching '{name}' for team '{team.upper()}'")
            
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
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like 404)
        raise http_exc
    except Exception as e:
        logger.error(f"Error getting player information for {name} ({team}): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting player information: {str(e)}")

@router.get("/api/player/{name}/stats")
@cache(expire=43200)
async def get_player_stats_endpoint(
    name: str = Path(..., description="Player name (format: 'First Last')"),
    team: Optional[str] = Query(None, description="Optional team abbreviation for disambiguation"),
    position: Optional[str] = Query(None, description="Player position override (QB, RB, WR, TE)"),
    aggregation: AggregationType = Query(AggregationType.SEASON, description="Aggregation level for statistics"),
    seasons: Optional[str] = Query(None, description="Comma-separated list of seasons e.g. '2022,2023'"),
    week: Optional[int] = Query(None, description="Filter by week number (only for aggregation = week)"),
    season_type: str = Query("REG", description="Season type filter (REG, POST, or REG+POST)"),
    redzone_only: bool = Query(False, description="If true, only include red-zone plays (inside 20-yard line)"),
    downs: Optional[str] = Query(None, description="Comma-separated list of downs to filter by e.g. '3,4'"),
    opponent_team: Optional[str] = Query(None, description="Filter by opponent team abbreviation"),
    score_differential_range: Optional[str] = Query(None, description="Min,max score differential filter e.g. '-14,0'"),
):
    """Return detailed statistics for an individual NFL player.

    The heavy-lifting is delegated to ``calculate_player_stats`` that lives in
    *player-stats.py*.  This endpoint mainly translates HTTP query parameters
    into the appropriate Python types before invoking that helper and then
    serialising the resulting pandas ``DataFrame`` into JSON-compatible output.
    """
    try:
        # Parse simple list / range query parameters
        seasons_parsed: Union[List[int], int, None]
        if seasons is None:
            seasons_parsed = None
        else:
            season_list = [int(s.strip()) for s in seasons.split(",") if s.strip()]
            if len(season_list) == 1:
                seasons_parsed = season_list[0]
            else:
                seasons_parsed = season_list

        downs_parsed: Optional[List[int]] = None
        if downs:
            downs_parsed = [int(d.strip()) for d in downs.split(",") if d.strip()]

        score_diff_parsed: Optional[List[int]] = None
        if score_differential_range and "," in score_differential_range:
            parts = [int(x.strip()) for x in score_differential_range.split(",") if x.strip()]
            if len(parts) == 2:
                score_diff_parsed = parts

        # Define PBP path and load data within the endpoint
        pbp_path = PathLib("cache/play_by_play_condensed.parquet")
        if not pbp_path.is_file():
            logger.error(f"PBP data file not found at: {pbp_path}")
            raise HTTPException(status_code=500, detail="Play-by-play data file is missing.")
        pbp_df = pd.read_parquet(pbp_path)

        stats_df = calculate_player_stats(
            pbp=pbp_df,
            player_name=name,
            team=team,
            position=position,
            aggregation_type=aggregation.value,
            seasons=seasons_parsed,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            downs=downs_parsed,
            opponent_team=opponent_team,
            score_differential_range=score_diff_parsed,
            add_player_name=True,
        )

        # Check if stats calculation returned an empty DataFrame (player not found/resolved)
        if stats_df.empty:
            # Attempt to provide more context if possible (requires async resolve_player again)
            # This adds complexity back, but improves UX for ambiguous names without team
            # For now, just return 404 if empty.
            raise HTTPException(status_code=404, detail=f"Player '{name}' not found or is ambiguous. Try providing a team.")

        # Convert DataFrame to list-of-dicts JSON
        records = [_sanitize_record(rec) for rec in stats_df.to_dict(orient="records")]
        
        # Extract resolved player info from the first record if available
        resolved_player_id = records[0].get("player_id") if records else None
        resolved_name = records[0].get("player_name", name) if records else name
        # Use the position passed into the endpoint or determined by calculate_player_stats if possible
        # However, calculate_player_stats doesn't currently return the determined position easily.
        # For now, return the position passed as a query parameter.
        # resolved_pos = position 
        # Updated: Get position from stats results
        resolved_pos = records[0].get("position") if records else position
        
        return {
            "player_id": resolved_player_id, # May be None if stats_df was empty but somehow didn't trigger 404
            "name": resolved_name,
            "position": resolved_pos,
            "team": team.upper(), # Return the provided team
            "stats": records,
        }
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly (like 404 from above)
        raise http_exc
    except Exception as e:
        logger.exception(f"Error computing stats for {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing stats: {e}")

@router.get("/api/player/{name}/headshot")
@cache(expire=86400)
async def get_headshot(
    name: str = Path(..., description="Player name"),
    season: Optional[int] = Query(None, description="NFL season year (defaults to most recent)") 
):
    """Get URL for player's headshot image."""
    try:
        # Resolve player - use await since resolve_player is now async
        player, alternatives = resolve_player(name=name, season=season)
        
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

@router.get("/api/players/top")
@cache(expire=43200)
async def get_top_players_endpoint(
    position: str = Query("QB", description="Player position to analyse"),
    n: int = Query(10, ge=1, le=100, description="Number of players to return"),
    sort_by: Optional[str] = Query(None, description="Column to sort by. If omitted a sensible default for the position is used."),
    min_threshold: Optional[str] = Query(None, description="Comma-sep key:value pairs e.g. 'carries:100' to filter players"),
    ascending: bool = Query(False, description="Sort ascending instead of descending"),
    seasons: Optional[str] = Query(None, description="Comma-separated list of seasons"),
    week: Optional[int] = Query(None, description="Filter by week (only when aggregation_type='week')"),
    season_type: str = Query("REG", description="Season type filter (REG, POST, or REG+POST)"),
    redzone_only: bool = Query(False, description="If true, only include red-zone plays"),
    aggregation_type: AggregationType = Query(AggregationType.SEASON, description="Aggregation type for underlying calculation"),
    include_player_details: bool = Query(True, description="Whether to include detailed player info columns"),
    downs: Optional[str] = Query(None, description="Comma-separated list of downs to filter by"),
    opponent_team: Optional[str] = Query(None, description="Filter by opponent team"),
    score_differential_range: Optional[str] = Query(None, description="Score diff range e.g. '-10,10'"),
):
    """Return a leaderboard of the top *N* players for a given position/metric."""
    try:
        # Parse helper parameters
        seasons_parsed: Union[List[int], int, None]
        if seasons:
            season_list = [int(s.strip()) for s in seasons.split(",") if s.strip()]
            seasons_parsed = season_list[0] if len(season_list) == 1 else season_list
        else:
            seasons_parsed = None

        downs_parsed: Optional[List[int]] = None
        if downs:
            downs_parsed = [int(d.strip()) for d in downs.split(",") if d.strip()]

        score_diff_parsed: Optional[List[int]] = None
        if score_differential_range and "," in score_differential_range:
            parts = [int(x.strip()) for x in score_differential_range.split(",") if x.strip()]
            if len(parts) == 2:
                score_diff_parsed = parts

        threshold_dict: Optional[Dict[str, int]] = None
        if min_threshold:
            threshold_dict = {}
            for pair in min_threshold.split(","):
                if ":" in pair:
                    key, val = pair.split(":", 1)
                    try:
                        threshold_dict[key.strip()] = int(val.strip())
                    except ValueError:
                        logger.warning(f"Could not parse threshold value '{pair}'. Skipping.")

        # Endpoint debug info
        print(f"TOP PLAYERS ENDPOINT: Position={position}, Aggregation={aggregation_type.value}, Week={week}, Seasons={seasons_parsed}")
        
        try:
            # Temporary debug - check parquet file
            import pandas as pd
            from pathlib import Path
            pbp_path = Path("cache/play_by_play_condensed.parquet")
            if pbp_path.exists():
                pbp_sample = pd.read_parquet(pbp_path)
                print(f"DEBUG: PBP parquet file exists with {len(pbp_sample)} rows")
                if len(pbp_sample) > 0:
                    # Get sample of seasons and weeks available
                    if 'season' in pbp_sample.columns:
                        seasons_available = sorted(pbp_sample['season'].unique().tolist())
                        print(f"DEBUG: Seasons available: {seasons_available}")
                    if 'week' in pbp_sample.columns:
                        weeks_available = sorted(pbp_sample['week'].unique().tolist())
                        print(f"DEBUG: Weeks available: {weeks_available}")
            else:
                print("DEBUG: PBP parquet file not found!")
            
            leaderboard_df = get_top_players(
                position=position,
                n=n,
                sort_by=sort_by,
                min_threshold=threshold_dict,
                ascending=ascending,
                aggregation_type=aggregation_type.value,
                seasons=seasons_parsed,
                week=week,
                season_type=season_type,
                redzone_only=redzone_only,
                include_player_details=include_player_details,
                downs=downs_parsed,
                opponent_team=opponent_team,
                score_differential_range=score_diff_parsed,
            )
        except ValueError as ve:
            # Handle specific error for unsupported position
            if "Unsupported position" in str(ve):
                logger.warning(f"Invalid position requested: {position}")
                raise HTTPException(status_code=400, detail=str(ve))
            # Handle other ValueError cases (like invalid sort_by column)
            raise HTTPException(status_code=400, detail=str(ve))

        if leaderboard_df.empty:
            return {
                "position": position,
                "leaderboard": [],
                "message": f"No players found matching the criteria for position '{position}'. Try adjusting filters."
            }
            
        # Convert to records and override any incorrect positions with the requested position
        # This ensures the position in the response matches what was requested
        records = []
        for rec in leaderboard_df.to_dict(orient="records"):
            sanitized_rec = _sanitize_record(rec)
            # Explicitly set position to match the requested position
            if "position" in sanitized_rec:
                sanitized_rec["position"] = position
            records.append(sanitized_rec)
            
        return {
            "position": position,
            "leaderboard": records,
        }
    except HTTPException as he:
        # Re-raise HTTP exceptions to preserve status code and message
        raise he
    except Exception as e:
        logger.exception(f"Error generating top players leaderboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating leaderboard: {e}")

@router.get("/api/{team}/{name}/stats")
@cache(expire=43200)
async def get_player_stats_by_team_endpoint(
    team: str = Path(..., description="Team abbreviation e.g. KC"),
    name: str = Path(..., description="Player name (format: 'First Last')"),
    position: Optional[str] = Query(None, description="Player position override (QB, RB, WR, TE)"),
    aggregation: AggregationType = Query(AggregationType.SEASON, description="Aggregation level for statistics"),
    seasons: Optional[str] = Query(None, description="Comma-separated list of seasons e.g. '2022,2023'"),
    week: Optional[int] = Query(None, description="Filter by week number (only for aggregation = week)"),
    season_type: str = Query("REG", description="Season type filter (REG, POST, or REG+POST)"),
    redzone_only: bool = Query(False, description="If true, only include red-zone plays (inside 20-yard line)"),
    downs: Optional[str] = Query(None, description="Comma-separated list of downs to filter by e.g. '3,4'"),
    opponent_team: Optional[str] = Query(None, description="Filter by opponent team abbreviation"),
    score_differential_range: Optional[str] = Query(None, description="Min,max score differential filter e.g. '-14,0'"),
):
    """Return detailed statistics for a player, using both *team* and *name* to identify them.

    Resolution is handled within ``calculate_player_stats``.
    """
    # Simplified endpoint - resolution logic moved to calculate_player_stats
    try:
        # Parse simple list / range query parameters
        seasons_parsed: Union[List[int], int, None]
        if seasons is None:
            seasons_parsed = None
        else:
            season_list = [int(s.strip()) for s in seasons.split(",") if s.strip()]
            if len(season_list) == 1:
                seasons_parsed = season_list[0]
            else:
                seasons_parsed = season_list

        downs_parsed: Optional[List[int]] = None
        if downs:
            downs_parsed = [int(d.strip()) for d in downs.split(",") if d.strip()]

        score_diff_parsed: Optional[List[int]] = None
        if score_differential_range and "," in score_differential_range:
            parts = [int(x.strip()) for x in score_differential_range.split(",") if x.strip()]
            if len(parts) == 2:
                score_diff_parsed = parts

        # Define PBP path and load data within the endpoint
        pbp_path = PathLib("cache/play_by_play_condensed.parquet")
        if not pbp_path.is_file():
            logger.error(f"PBP data file not found at: {pbp_path}")
            raise HTTPException(status_code=500, detail="Play-by-play data file is missing.")
        pbp_df = pd.read_parquet(pbp_path)

        stats_df = calculate_player_stats(
            pbp=pbp_df,
            player_name=name,
            team=team.upper(), # Pass team for resolution
            position=position,
            aggregation_type=aggregation.value,
            seasons=seasons_parsed,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            downs=downs_parsed,
            opponent_team=opponent_team,
            score_differential_range=score_diff_parsed,
            add_player_name=True,
        )

        if stats_df.empty:
             raise HTTPException(status_code=404, detail=f"Player '{name}' not found for team '{team.upper()}'.")

        # Convert DataFrame to list-of-dicts JSON
        records = [_sanitize_record(rec) for rec in stats_df.to_dict(orient="records")]

        # Extract resolved player info from the first record if available
        resolved_player_id = records[0].get("player_id") if records else None
        resolved_name = records[0].get("player_name", name) if records else name
        # Use the position passed into the endpoint or determined by calculate_player_stats if possible
        # However, calculate_player_stats doesn't currently return the determined position easily.
        # For now, return the position passed as a query parameter.
        # resolved_pos = position 
        # Updated: Get position from stats results
        resolved_pos = records[0].get("position") if records else position

        return {
            "player_id": resolved_player_id,
            "name": resolved_name,
            "position": resolved_pos,
            "team": team.upper(), # Return the provided team
            "stats": records,
        }
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly (like 404 from above)
        raise http_exc
    except Exception as e:
        logger.exception(f"Error computing stats for {name} ({team}): {e}")
        raise HTTPException(status_code=500, detail=f"Error computing stats: {e}")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sanitize_record(record):
    """Convert NumPy / pandas dtypes to vanilla Python types and remove NaN/Inf.

    This helper is used before returning data through the API to ensure the JSON
    serializer does not choke on unsupported values."""
    sanitized = {}
    for k, v in record.items():
        # Handle floats first so we can check for nan/inf early
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            sanitized[k] = None
        elif pd.isna(v):
            sanitized[k] = None
        else:
            if isinstance(v, np.integer):
                sanitized[k] = int(v)
            elif isinstance(v, np.floating):
                sanitized[k] = float(v)
            else:
                sanitized[k] = v
    return sanitized
