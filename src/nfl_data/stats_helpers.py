"""Helper functions for calculating various NFL statistics."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging
from pydantic_core import PydanticUndefined # Import the Undefined type

# Import the actual data loading function - REMOVED as file doesn't exist
# from .data_loader import load_players, load_weekly_stats, load_pbp_data, load_schedules, load_injuries, load_depth_charts 
# Assuming load_pbp_data might exist elsewhere or be synchronous if used
# Need to define or import load_pbp_data, load_weekly_stats etc. if facades are used
# For now, we focus on fixing resolve_player

# Set up logging at the module level
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
# Legacy compatibility wrappers (replacing the removed `data_import` module)
# ---------------------------------------------------------------------------------
# ``data_import.py`` has been deleted.  We provide thin facade functions here so
# that the rest of this module continues to work without large-scale refactors.
# Each wrapper delegates directly to the corresponding function in
# ``data_loader``.

# Remove or comment out the facade functions if they are no longer needed 
# or ensure they correctly call the imported functions.

# def import_pbp_data(): # Assuming this might be used synchronously
#     """Return the condensed play-by-play DataFrame (sync)."""
#     # Assuming load_pbp_data() is the correct synchronous function
#     return load_pbp_data() # Needs definition/import

# async def import_weekly_data(seasons): # Needs load_weekly_stats
#     """Async wrapper around ``load_weekly_stats`` for backward compatibility."""
#     return await load_weekly_stats(seasons)

# async def import_players(): # REMOVED - resolve_player now handles loading
#      """Async wrapper for loading players."""
#      return await load_players()

# def import_schedules(seasons): # Needs load_schedules
#     return load_schedules(seasons)

# async def import_injuries(seasons): # Needs load_injuries
#     return await load_injuries(seasons)

# async def import_depth_charts(seasons): # Needs load_depth_charts
#     return await load_depth_charts(seasons)

# Updated: Use imported functions directly where possible or fix facades.
# Example: If import_pbp_data is still used elsewhere, fix it:
def import_pbp_data():
    """Return the condensed play-by-play DataFrame (sync)."""
    # Assuming load_pbp_data() is the correct synchronous function
        # Load PBP data directly from the condensed cache file
    pbp_file_path = "cache/play_by_play_condensed.parquet"
    try:
        pbp_data = pd.read_parquet(pbp_file_path)
        logger.info(f"Loaded PBP data from {pbp_file_path}")
    except FileNotFoundError:
        logger.error(f"PBP cache file not found at: {pbp_file_path}")
        return {"error": f"Required PBP data file not found. Please run ETL refresh."}
    except Exception as load_err:
        logger.error(f"Error loading PBP data from {pbp_file_path}: {load_err}")
        return {"error": f"Error loading required PBP data."}

    if pbp_data.empty:
        return {"error": f"No play-by-play data could be loaded."}
async def import_weekly_data(seasons):
    """Async wrapper around ``load_weekly_stats`` for backward compatibility."""
    return await load_weekly_stats(seasons)

async def import_players():
     """Async wrapper for loading players."""
     return await load_players()

def import_schedules(seasons):
    return load_schedules(seasons)

async def import_injuries(seasons):
    return await load_injuries(seasons)

async def import_depth_charts(seasons):
    return await load_depth_charts(seasons)

def get_current_season() -> int:
    """Get the current NFL season year.
    During offseason/preseason, returns the upcoming season.
    During regular season/playoffs, returns the current season."""
    current_month = datetime.now().month
    current_year = datetime.now().year
    # If we're in the NFL offseason (Feb-Aug), return the upcoming season
    return current_year if current_month >= 9 else current_year - 1

def get_defensive_stats(team: str, pbp_data: Optional[pd.DataFrame] = None, weekly_stats: Optional[pd.DataFrame] = None, season: Optional[int] = None) -> Optional[Dict]:
    """Get defensive statistics for a team.
    
    Args:
        team: Team abbreviation
        pbp_data: Optional pre-loaded play-by-play data
        weekly_stats: Optional pre-loaded weekly stats data
        season: Optional season to get stats for (defaults to current season)
        
    Returns:
        Dictionary of defensive statistics or None if team not found
    """
    if season is None:
        season = get_current_season()
        
    # Load data if not provided
    if pbp_data is None:
        pbp_data = import_pbp_data()
        pbp_data = pbp_data[pbp_data['season'] == season]
    if weekly_stats is None:
        weekly_stats = import_weekly_data([season])
        
    if pbp_data.empty or weekly_stats.empty:
        return None
        
    # Filter plays where this team is on defense
    defensive_plays = pbp_data[pbp_data['defteam'] == team]
    if defensive_plays.empty:
        return None
        
    # Calculate defensive stats
    stats = {}
    
    # Points allowed per game
    points_allowed = defensive_plays.groupby('game_id')['posteam_score'].max().mean()
    stats['points_allowed_per_game'] = float(points_allowed) if not pd.isna(points_allowed) else 0
    
    # Sacks per game
    total_sacks = defensive_plays['sack'].sum()
    num_games = defensive_plays['game_id'].nunique()
    stats['sacks_per_game'] = float(total_sacks / num_games) if num_games > 0 else 0
    
    # Add more defensive stats as needed
    
    return stats

def get_historical_matchup_stats(
    player_name: str,
    opponent: str,
    seasons: Optional[List[int]] = None
) -> Dict:
    """Get historical matchup statistics for a player against a specific opponent.
    
    Args:
        player_name: Name of the player
        opponent: Opponent team abbreviation
        seasons: Optional list of seasons (defaults to last 3 seasons)
        
    Returns:
        Dictionary of historical matchup statistics
    """
    try:
        if seasons is None:
            current_season = get_current_season()
            seasons = [current_season, current_season-1, current_season-2]
            
        player, alternatives = resolve_player(player_name)
        if not player:
            return {"error": f"No player found matching '{player_name}'"} if not alternatives else \
                   {"error": f"Multiple players found matching '{player_name}'", "matches": alternatives}
                   
        player_id = player['gsis_id']
        
        # NOTE: This function is now synchronous. 
        # It requires a synchronous way to load weekly data.
        # Assuming import_weekly_data can be called synchronously or has a sync alternative.
        try:
            # Removed await
            # weekly_stats = await import_weekly_data(seasons) 
            weekly_stats = import_weekly_data(seasons) # Placeholder - needs fixing if used
        except NameError:
            logger.error("import_weekly_data is not defined or imported correctly, or has no synchronous version.")
            return {"error": "Server configuration error: Cannot load weekly stats."}
        except Exception as e:
            logger.error(f"Error loading weekly data for seasons {seasons}: {e}")
            return {"error": f"Could not load weekly stats for seasons {seasons}."}
            
        if weekly_stats is None or weekly_stats.empty:
            return {"error": f"No weekly stats data found for seasons: {seasons}"}
            
        matchup_stats = weekly_stats[
            (weekly_stats['player_id'] == player_id) &
            (weekly_stats['opponent'] == opponent)
        ]
        
        # Aggregate stats if multiple games found
        if not matchup_stats.empty:
            # Select numeric columns for aggregation (example)
            numeric_cols = matchup_stats.select_dtypes(include=np.number).columns
            aggregated_stats = matchup_stats[numeric_cols].mean().to_dict()
            # Convert numpy types to standard Python types
            aggregated_stats = {k: (v.item() if hasattr(v, 'item') else v) for k, v in aggregated_stats.items()}
            return {
                "player_name": player_name,
                "opponent": opponent,
                "seasons": seasons,
                "games_found": len(matchup_stats),
                "average_stats": aggregated_stats
            }
        else:
            return {
                "player_name": player_name,
                "opponent": opponent,
                "seasons": seasons,
                "games_found": 0,
                "message": "No matchup data found for this player against this opponent in the specified seasons."
            }
            
    except Exception as e:
        logger.exception(f"Error calculating historical stats for {player_name} vs {opponent}: {e}")
        return {"error": f"An error occurred calculating historical stats: {str(e)}"}


# Updated: Changed to synchronous function, loads data directly
def resolve_player(name: str, team: Optional[str] = None, season: Optional[int] = None) -> Tuple[Optional[Dict], List[Dict]]:
    """Find a player by name, optionally filtering by team and season.
    
    Args:
        name: Player name (e.g., "Patrick Mahomes")
        team: Optional team abbreviation (e.g., "KC")
        season: Optional season year for context (defaults to most recent)
        
    Returns:
        Tuple: (Found player dictionary or None, List of alternative matches)
    """
    logger.debug(f"Resolving player: name='{name}', team='{team}', season='{season}'")
    try:
        # Use the asynchronous loader - REMOVED
        # players_df = await load_players() 
        # Load directly from parquet file
        player_file_path = "cache/players.parquet"
        try:
            players_df = pd.read_parquet(player_file_path)
            logger.info(f"Successfully loaded player data from {player_file_path}")
        except FileNotFoundError:
             logger.error(f"Player data file not found at: {player_file_path}")
             return None, []
        except Exception as load_err:
             logger.error(f"Error loading player data from {player_file_path}: {load_err}")
             return None, []

        if players_df.empty:
            logger.error("Loaded players DataFrame is empty.")
            return None, []

        # Ensure team is uppercase if provided
        if team:
            team = team.upper()

        # Filter by season if provided (relevant for finding the player on a specific team *in that season*)
        # Note: players.parquet usually has the *current* team. For historical team context,
        # other data sources like roster history might be needed, but this is a basic filter.
        filtered_players = players_df.copy()
        # We don't filter by season directly here as players_df might not represent seasonal rosters accurately.
        # Instead, we use season context implicitly when matching team.

        # --- Search Logic --- 
        # 1. Exact match on display_name (case-insensitive)
        name_lower = name.lower()
        matches = filtered_players[filtered_players['display_name'].str.lower() == name_lower]
        logger.debug(f"Initial exact matches by name: {len(matches)}")

        # 2. Handle "Last, First" format if necessary
        if matches.empty:
            alt_name = None
            if ',' in name_lower:
                try:
                    last, first = name_lower.split(',', 1)
                    alt_name = f"{first.strip()} {last.strip()}"
                    matches = filtered_players[filtered_players['display_name'].str.lower() == alt_name]
                except ValueError:
                    pass # Ignore malformed name
            else:
                parts = name_lower.split()
                if len(parts) >= 2:
                    alt_name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                    matches = filtered_players[filtered_players['display_name'].str.lower() == alt_name]
            if not matches.empty:
                 logger.debug(f"Found {len(matches)} matches using alternative name format: '{alt_name}'")

        # 3. Fallback: Contains match (case-insensitive)
        if matches.empty:
            matches = filtered_players[filtered_players['display_name'].str.lower().str.contains(name_lower, na=False)]
            logger.debug(f"Found {len(matches)} matches using CONTAINS '{name_lower}'")
            # If still no match, try contains with the alternative name format
            if matches.empty and 'alt_name' in locals() and alt_name:
                 matches = filtered_players[filtered_players['display_name'].str.lower().str.contains(alt_name, na=False)]
                 logger.debug(f"Found {len(matches)} matches using CONTAINS '{alt_name}'")
        
        # --- Disambiguation --- 
        if matches.empty:
            logger.warning(f"No player found matching criteria: name='{name}', team='{team}'")
            return None, []
        
        if len(matches) == 1:
            player_data = matches.iloc[0]
            # If team was provided, verify it matches the player's current team or if they are inactive (no team)
            # Add detailed logging for the team comparison
            player_team_abbr = str(player_data.get('team_abbr', '')).upper()
            player_display_name = player_data.get('display_name', '[Name Missing]')
            is_team_abbr_null = pd.isna(player_data.get('team_abbr'))
            logger.debug(f"Resolve Player Check: Single match found for '{player_display_name}'. Player team from data: '{player_team_abbr}' (Is Null: {is_team_abbr_null}). Requested team: '{team}'.")
            
            if team and player_team_abbr != team and not is_team_abbr_null:
                logger.warning(f"Found unique player '{player_data['display_name']}' but team '{player_team_abbr}' does not match requested team '{team}'. Returning alternatives leading to 404.")
                # Return no primary match, but include the found player in alternatives
                return None, matches.fillna('').to_dict('records') 
            else:
                # Unique match, and either no team was specified, or the team matches, or player is inactive
                logger.info(f"Found unique player: {player_data['display_name']} (ID: {player_data['gsis_id']}) matching criteria.")
                return player_data.fillna('').to_dict(), [] # Return single match, no alternatives
        else: # Multiple matches found by name
            alternatives = matches.fillna('').to_dict('records')
            if team:
                # Try filtering the multiple matches by the specified team
                team_matches = matches[matches['team_abbr'].str.upper() == team]
                if len(team_matches) == 1:
                    player_data = team_matches.iloc[0]
                    logger.info(f"Found unique player via team disambiguation: {player_data['display_name']} (ID: {player_data['gsis_id']}) for team {team}.")
                    return player_data.fillna('').to_dict(), alternatives # Return unique match and the original alternatives
                elif len(team_matches) > 1:
                    logger.warning(f"Multiple players found for name '{name}' on team '{team}'. Returning alternatives.")
                    return None, team_matches.fillna('').to_dict('records') # Return only the team-specific alternatives
                else:
                    # No players on the list matched the specific team
                    logger.warning(f"Multiple players found for name '{name}', but none matched team '{team}'. Returning all alternatives.")
                    return None, alternatives
            else:
                # Multiple matches found, but no team was provided for disambiguation
                logger.warning(f"Multiple players found for name '{name}' and no team provided. Returning alternatives.")
                return None, alternatives

    except FileNotFoundError:
        logger.error("Error resolving player: players.parquet not found.")
        return None, []
    except Exception as e:
        logger.exception(f"Unexpected error resolving player '{name}': {e}")
        return None, []

def get_player_headshot_url(player_id: str) -> str:
    """Get URL for player's headshot image."""
    # Use the NFL CDN URL format
    return f"https://static.www.nfl.com/image/private/t_player_profile_landscape/f_auto/league/{player_id}"

async def get_player_game_log(player_name: str, season: Optional[int] = None) -> Dict:
    """Get game-by-game stats for a player."""
    # Resolve player first
    # Updated: Removed await, resolve_player is now synchronous
    player, alternatives = resolve_player(player_name)
    if not player and alternatives:
        return {
            "error": f"Multiple players found matching '{player_name}'",
            "matches": alternatives
        }
    if not player:
        return {
            "error": f"No player found matching '{player_name}'"
        }
    
    if not season:
        season = get_current_season() # Use helper to get latest completed/active season
    
    # Get weekly data - use await since this is an async function
    # NOTE: This still relies on an async import_weekly_data/load_weekly_stats
    try:
        # Assuming import_weekly_data exists and works async
        # weekly_data = await import_weekly_data([season]) # Commented out - needs fix
        weekly_data = pd.DataFrame() # Placeholder
    except NameError:
        logger.error("import_weekly_data is not defined or imported correctly.")
        return {"error": "Server configuration error: Cannot load weekly stats."} 
    except Exception as e:
        logger.error(f"Error loading weekly data for season {season}: {e}")
        return {"error": f"Could not load weekly stats for season {season}."}

    # Filter using player_id (gsis_id) for reliability instead of player_name
    if 'player_id' in weekly_data.columns:
        player_games = weekly_data[weekly_data['player_id'] == player["gsis_id"]]
    else:
        # Fallback or error if player_id column is missing (should not happen with standard data)
        logger.warning(f"'player_id' column not found in weekly_data for season {season}. Cannot filter games accurately.")
        player_games = pd.DataFrame() # Return empty DataFrame

    # Return formatted response
    return {
        "player_id": player["gsis_id"],
        "player_name": player["display_name"],
        "team": player["team_abbr"],
        "position": player["position"],
        "games": player_games.to_dict('records'),
        "season": season
    }

async def get_player_career_stats(player_name: str) -> Dict:
    """Get career stats for a player."""
    # Resolve player first
    # Updated: Removed await, resolve_player is now synchronous
    player, alternatives = resolve_player(player_name)
    if not player and alternatives:
        return {
            "error": f"Multiple players found matching '{player_name}'",
            "matches": alternatives
        }
    if not player:
        return {
            "error": f"No player found matching '{player_name}'"
        }
    
    # TODO: Implement actual career stats calculation
    # For now, return a minimal structure that matches the expected model
    return {
        "player_id": player["gsis_id"],
        "player_name": player["display_name"],
        "team": player["team_abbr"],
        "position": player["position"],
        "career_stats": {"games_played": 0, "seasons": 0},
        "seasons_played": [2024]
    }

async def get_situation_stats(player_name: str, situations: List[str], season: Optional[int] = None) -> Dict:
    """Get player stats filtered by one or more game situations."""
    # Resolve player first
    # Updated: Removed await, resolve_player is now synchronous
    player, alternatives = resolve_player(player_name, season)
    if not player and alternatives:
        return {
            "error": f"Multiple players found matching '{player_name}'",
            "matches": alternatives
        }
    if not player:
        return {
            "error": f"No player found matching '{player_name}'"
        }

    player_id = player["gsis_id"]
    position = player.get("position", "")
    logger.info(f"Calculating situation stats for {player['display_name']} ({player_id}), situations: {situations}, season: {season}")

    try:
        # Load PBP data directly from the condensed cache file
        pbp_file_path = "cache/play_by_play_condensed.parquet"
        try:
            # pbp_data = load_pbp_data() # ERROR: Not defined
            pbp_data = pd.read_parquet(pbp_file_path)
            logger.info(f"Loaded PBP data from {pbp_file_path}")
        except FileNotFoundError:
            logger.error(f"PBP cache file not found at: {pbp_file_path}")
            return {"error": f"Required PBP data file not found. Please run ETL refresh."}
        except Exception as load_err:
            logger.error(f"Error loading PBP data from {pbp_file_path}: {load_err}")
            return {"error": f"Error loading required PBP data."}
            
        if pbp_data.empty:
             return {"error": f"No play-by-play data could be loaded."}

        # Filter PBP data by season if specified
        if season:
             if 'season' in pbp_data.columns:
                 pbp_data = pbp_data[pbp_data['season'] == season]
                 if pbp_data.empty:
                     return {"error": f"No play-by-play data available for season {season}"}
                 logger.info(f"Filtered PBP data to season {season}, {len(pbp_data)} plays remaining.")
             else:
                 logger.warning(f"Could not filter PBP data by season {season}, 'season' column not found.")
                 # Proceed with unfiltered data, but log a warning

        # Filter relevant PBP columns if needed to reduce memory
        # relevant_cols = [...] 
        # pbp_data = pbp_data[relevant_cols]

        # Find plays involving the player
        player_id_variations = [player_id]
        if 'gsis_it_id' in player and player['gsis_it_id']:
             player_id_variations.append(player['gsis_it_id'])
        
        player_cols = [
            'passer_player_id', 'receiver_player_id', 'rusher_player_id',
            'lateral_receiver_player_id', 'lateral_rusher_player_id',
            'fumbled_1_player_id', 'fumbled_2_player_id', 'sack_player_id', # Add other relevant IDs
            'pass_defense_1_player_id', 'pass_defense_2_player_id',
            'interception_player_id', 'tackle_for_loss_1_player_id',
            'tackle_for_loss_2_player_id', 'qb_hit_1_player_id', 'qb_hit_2_player_id' 
        ]
        valid_player_cols = [col for col in player_cols if col in pbp_data.columns]
        player_plays_mask = pd.Series(False, index=pbp_data.index)
        for pid_var in player_id_variations:
             if pid_var: # Ensure ID is not None or empty
                 for col in valid_player_cols:
                      # Ensure comparison is robust to missing values (NaN)
                      player_plays_mask |= (pbp_data[col].fillna('') == pid_var)

        if not player_plays_mask.any():
             logger.warning(f"No plays found involving player {player_id}")
             # Return empty stats instead of error
             return {
                 "player_id": player_id,
                 "player_name": player["display_name"],
                 "team": player.get("team_abbr", ""),
                 "position": position,
                 "situations_requested": situations,
                 "season": season,
                 "stats": { "plays": 0 }
             }

        player_plays = pbp_data[player_plays_mask].copy()
        logger.info(f"Found {len(player_plays)} plays involving player {player_id}")

        # Apply situation filters
        combined_situation_mask = pd.Series(False, index=player_plays.index)
        valid_situations_applied = []

        for situation in situations:
            situation_mask = pd.Series(False, index=player_plays.index)
            situation_key = situation.lower().strip()
            applied_flag = False

            if situation_key == "red_zone":
                if 'yardline_100' in player_plays.columns:
                     situation_mask = player_plays['yardline_100'] <= 20
                     applied_flag = True
            elif situation_key == "third_down":
                if 'down' in player_plays.columns:
                     situation_mask = player_plays['down'] == 3
                     applied_flag = True
            elif situation_key == "fourth_down":
                 if 'down' in player_plays.columns:
                     situation_mask = player_plays['down'] == 4
                     applied_flag = True
            elif situation_key == "goal_line":
                if 'yardline_100' in player_plays.columns:
                     situation_mask = player_plays['yardline_100'] <= 5 # Adjust definition as needed
                     applied_flag = True
            elif situation_key == "two_minute_drill":
                 if 'qtr' in player_plays.columns and 'half_seconds_remaining' in player_plays.columns:
                     two_min_mask = (
                         ((player_plays['qtr'] == 2) & (player_plays['half_seconds_remaining'] <= 120)) |
                         ((player_plays['qtr'] == 4) & (player_plays['game_seconds_remaining'] <= 120)) # Use game_seconds for 4th qtr
                     )
                     situation_mask = two_min_mask
                     applied_flag = True
            # Add more situations here if needed
            else:
                 logger.warning(f"Unsupported situation type requested: {situation}")
                 continue # Skip unsupported situation

            if applied_flag:
                 logger.info(f"Applying filter for situation: {situation_key}")
                 combined_situation_mask |= situation_mask.fillna(False) # Combine masks with OR, handle NaNs
                 if situation not in valid_situations_applied:
                     valid_situations_applied.append(situation)
            else:
                 logger.warning(f"Could not apply filter for situation '{situation_key}' due to missing PBP columns.")
        
        # Filter plays by the combined situation mask
        if not valid_situations_applied:
             # Handle case where none of the requested situations could be applied
             logger.error(f"None of the requested situations could be applied: {situations}")
             # Optionally return an error or specific message
             # For now, return stats for all player plays if no valid situation filters applied
             situation_filtered_plays = player_plays 
        elif not combined_situation_mask.any():
            logger.info(f"No plays found matching the requested situations: {valid_situations_applied}")
            situation_filtered_plays = pd.DataFrame(columns=player_plays.columns) # Empty dataframe
        else:
            situation_filtered_plays = player_plays[combined_situation_mask]
            logger.info(f"Found {len(situation_filtered_plays)} plays after applying situation filters: {valid_situations_applied}")

        # Calculate stats using the helper function
        calculated_stats = get_position_specific_stats_from_pbp(
            situation_filtered_plays, 
            position,
            player_id=player_id
            # Pass other args to get_position_specific_stats_from_pbp if needed
        )

        # Add play count
        calculated_stats['plays'] = len(situation_filtered_plays)

        # --- Sanitize the calculated stats --- 
        sanitized_calculated_stats = {}
        for key, value in calculated_stats.items():
            if value is PydanticUndefined:
                sanitized_calculated_stats[key] = None # Replace Undefined with None
            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                 sanitized_calculated_stats[key] = None # Replace NaN/inf with None
            elif pd.isna(value):
                 sanitized_calculated_stats[key] = None # Replace Pandas NA with None
            else:
                 # Convert numpy types just in case
                 if isinstance(value, np.integer):
                     sanitized_calculated_stats[key] = int(value)
                 elif isinstance(value, np.floating):
                      sanitized_calculated_stats[key] = float(value)
                 else:
                     sanitized_calculated_stats[key] = value
        # --- End Sanitization ---

        # Format response using the sanitized stats
        return {
            "player_id": player_id,
            "player_name": player["display_name"],
            "team": player.get("team_abbr", ""),
            "position": position,
            "situations_applied": valid_situations_applied,
            "season_filter": season,
            "stats": sanitized_calculated_stats # Use the sanitized dictionary
        }

    except FileNotFoundError as e:
         logger.error(f"Data file not found: {e}")
         return {"error": f"Required data file not found for season {season}. Please ensure data is loaded."}
    except KeyError as e:
         logger.error(f"Missing expected column in PBP data: {e}")
         return {"error": f"Data processing error: Missing expected column '{e}'. The data might be incomplete or corrupted."}
    except Exception as e:
        logger.exception(f"Error calculating situation stats for {player_name}, situations {situations}: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def get_position_specific_stats_from_pbp(
    plays: pd.DataFrame,
    position: str,
    weekly_data: Optional[pd.DataFrame] = None,
    player_id: Optional[str] = None,
    situation_filters: Optional[Dict] = None,
    game_filters: Optional[Dict] = None
) -> Dict:
    """Calculate position-specific stats from play by play data.
    
    Args:
        plays: Play by play data filtered for the player
        position: Player position (QB, RB, WR, TE)
        weekly_data: Optional weekly stats data for verification
        player_id: Player's GSIS ID for weekly data lookup
        situation_filters: Optional dict of situational filters to apply:
            - down: int (1-4)
            - distance: int
            - quarter: int (1-5, where 5 is OT)
            - half: str ('first' or 'second')
            - score_differential: int
            - red_zone: bool
            - shotgun: bool
            - no_huddle: bool
            - qb_under_pressure: bool
        game_filters: Optional dict of game-specific filters:
            - last_n_games: int (get stats from only the last N games)
            - opponent: str (team abbreviation to filter games against)
            - division_games: bool (only include divisional games)
            - home_games: bool (only include home games)
            - away_games: bool (only include away games)
            - season_games: List[int] (specific weeks to include)
            - exclude_games: List[str] (game_ids to exclude)
            
    Returns:
        dict: Dictionary of position-specific statistics
    """
    if len(plays) == 0:
        return {}
        
    # Log input DataFrame columns and head
    logger.info(f"[get_pos_stats] Received plays DataFrame with shape {plays.shape} and columns: {plays.columns.tolist()}")
    # logger.debug(f"[get_pos_stats] Head of received plays:\n{plays.head().to_string()}") # DEBUG - potentially verbose

    # Apply game filters if provided
    if game_filters:
        filtered_plays = plays.copy()
        
        for key, value in game_filters.items():
            if key == 'last_n_games':
                # Get the last N unique game IDs
                last_n_games = filtered_plays.sort_values('game_date', ascending=False)['game_id'].unique()[:value]
                filtered_plays = filtered_plays[filtered_plays['game_id'].isin(last_n_games)]
                
            elif key == 'opponent':
                # Filter for games against specific opponent
                filtered_plays = filtered_plays[
                    (filtered_plays['defteam'] == value) |  # When player's team is on offense
                    (filtered_plays['posteam'] == value)    # When player's team is on defense
                ]
                
            elif key == 'division_games':
                if value:
                    # Get player's team
                    team = filtered_plays['posteam'].iloc[0]
                    # Define division mappings
                    divisions = {
                        'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
                        'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
                        'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
                        'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
                        'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
                        'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
                        'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
                        'NFC_WEST': ['ARI', 'LAR', 'SF', 'SEA']
                    }
                    # Find player's division
                    player_division = next(
                        (div for div, teams in divisions.items() if team in teams),
                        None
                    )
                    if player_division:
                        # Filter for games against division opponents
                        division_teams = divisions[player_division]
                        filtered_plays = filtered_plays[
                            (filtered_plays['defteam'].isin(division_teams)) |
                            (filtered_plays['posteam'].isin(division_teams))
                        ]
                        
            elif key == 'home_games':
                if value:
                    team = filtered_plays['posteam'].iloc[0]
                    filtered_plays = filtered_plays[filtered_plays['home_team'] == team]
                    
            elif key == 'away_games':
                if value:
                    team = filtered_plays['posteam'].iloc[0]
                    filtered_plays = filtered_plays[filtered_plays['away_team'] == team]
                    
            elif key == 'season_games':
                filtered_plays = filtered_plays[filtered_plays['week'].isin(value)]
                
            elif key == 'exclude_games':
                filtered_plays = filtered_plays[~filtered_plays['game_id'].isin(value)]
    else:
        filtered_plays = plays
        
    # Apply situational filters if provided
    if situation_filters:
        for key, value in situation_filters.items():
            if key == 'down' and value in [1, 2, 3, 4]:
                filtered_plays = filtered_plays[filtered_plays['down'] == value]
            elif key == 'distance':
                filtered_plays = filtered_plays[filtered_plays['ydstogo'] == value]
            elif key == 'quarter' and value in [1, 2, 3, 4, 5]:
                filtered_plays = filtered_plays[filtered_plays['qtr'] == value]
            elif key == 'half':
                if value == 'first':
                    filtered_plays = filtered_plays[filtered_plays['qtr'].isin([1, 2])]
                elif value == 'second':
                    filtered_plays = filtered_plays[filtered_plays['qtr'].isin([3, 4, 5])]
            elif key == 'score_differential':
                filtered_plays = filtered_plays[filtered_plays['score_differential'] == value]
            elif key == 'red_zone':
                filtered_plays = filtered_plays[filtered_plays['yardline_100'] <= 20]
            elif key == 'shotgun':
                filtered_plays = filtered_plays[filtered_plays['shotgun'] == value]
            elif key == 'no_huddle':
                filtered_plays = filtered_plays[filtered_plays['no_huddle'] == value]
            elif key == 'qb_under_pressure':
                filtered_plays = filtered_plays[filtered_plays['qb_hit'] == value]
    
    stats = {}
    
    # Log filtered DataFrame info before position logic
    logger.info(f"[get_pos_stats] Filtered plays DataFrame shape: {filtered_plays.shape}")
    # logger.debug(f"[get_pos_stats] Head of filtered plays:\n{filtered_plays.head().to_string()}") # DEBUG - potentially verbose

    if position == 'QB':
        # Calculate QB stats from filtered plays
        # Ensure required columns exist before attempting calculations
        required_qb_cols = ['pass_attempt', 'complete_pass', 'yards_gained', 'pass_touchdown', 'interception', 'sack']
        if not all(col in filtered_plays.columns for col in required_qb_cols):
            logger.warning(f"[get_pos_stats] Missing required QB columns in filtered_plays. Available: {filtered_plays.columns.tolist()}")
            return {} # Return empty if required columns are missing
        
        passing_plays = filtered_plays[filtered_plays['pass_attempt'] == 1]
        total_attempts = len(passing_plays)
        completions = passing_plays['complete_pass'].fillna(0).sum()
        # Patch: Always provide 'passing_yards' as sum of 'yards_gained' for pass attempts
        if 'passing_yards' in passing_plays.columns:
            passing_yards = passing_plays['passing_yards'].fillna(0).sum()
        else:
            passing_yards = passing_plays['yards_gained'].fillna(0).sum()
        passing_tds = passing_plays['pass_touchdown'].fillna(0).sum()
        interceptions = passing_plays['interception'].fillna(0).sum()
        sacks = filtered_plays['sack'].fillna(0).sum()
        
        # Calculate derived stats
        completion_percentage = (completions / total_attempts * 100) if total_attempts > 0 else 0
        yards_per_attempt = (passing_yards / total_attempts) if total_attempts > 0 else 0
        touchdown_percentage = (passing_tds / total_attempts * 100) if total_attempts > 0 else 0
        interception_percentage = (interceptions / total_attempts * 100) if total_attempts > 0 else 0
        sack_rate = (sacks / (total_attempts + sacks) * 100) if (total_attempts + sacks) > 0 else 0
        
        # Log calculated QB stats
        logger.info(f"[get_pos_stats QB] Attempts: {total_attempts}, Completions: {completions}, PassYds: {passing_yards}, PassTDs: {passing_tds}, INTs: {interceptions}, Sacks: {sacks}")
        # === Add detailed logging for derived stats ===
        logger.info(f"[DEBUG QB Derived] Comp%: {completion_percentage}, YPA: {yards_per_attempt}, TD%: {touchdown_percentage}, INT%: {interception_percentage}, SackRate: {sack_rate}")
        # === End detailed logging ===

        # --- Calculate QB Rushing Stats ---
        total_rushes = 0
        rushing_yards = 0.0
        rushing_tds = 0
        fumbles = 0
        yards_per_carry = 0.0

        # Check if necessary rushing columns exist
        required_rush_cols = ['rush_attempt', 'rushing_yards', 'rush_touchdown', 'fumble_lost']
        if all(col in filtered_plays.columns for col in required_rush_cols):
            rushing_plays = filtered_plays[filtered_plays['rush_attempt'] == 1]
            total_rushes = len(rushing_plays)
            if total_rushes > 0:
                rushing_yards = rushing_plays['rushing_yards'].fillna(0).sum()
                rushing_tds = rushing_plays['rush_touchdown'].fillna(0).sum()
                # Use 'fumble_lost' here
                fumbles = rushing_plays['fumble_lost'].fillna(0).sum()
                yards_per_carry = (rushing_yards / total_rushes) if total_rushes > 0 else 0.0
            # Log calculated QB rushing stats
            logger.info(f"[get_pos_stats QB] Rushes: {total_rushes}, RushYds: {rushing_yards}, RushTDs: {rushing_tds}, FumLost: {fumbles}")
            # === Add detailed logging for derived rushing stats ===
            logger.info(f"[DEBUG QB Rushing Derived] YPC: {yards_per_carry}")
            # === End detailed logging ===
        else:
            logger.warning(f"[get_pos_stats QB] Missing required Rushing columns in filtered_plays. Cannot calculate QB rushing stats. Available: {filtered_plays.columns.tolist()}")
        # --- End QB Rushing Stats ---

        # === Log final stats dict before update ===
        temp_stats_dict = {
            'total_attempts': int(total_attempts),
            'completions': int(completions),
            'passing_yards': float(passing_yards),
            'passing_tds': int(passing_tds),
            'interceptions': int(interceptions),
            'sacks': int(sacks),
            'completion_percentage': float(completion_percentage),
            'yards_per_attempt': float(yards_per_attempt),
            'touchdown_percentage': float(touchdown_percentage),
            'interception_percentage': float(interception_percentage),
            'sack_rate': float(sack_rate),
            'total_rush_attempts': int(total_rushes),
            'total_rushing_yards': float(rushing_yards),
            'rushing_tds': int(rushing_tds),
            'fumbles': int(fumbles),
            'yards_per_carry': float(yards_per_carry)
        }
        logger.info(f"[DEBUG QB Final Dict] Stats before update: {temp_stats_dict}")
        # === End log final stats dict ===

        stats.update(temp_stats_dict)

    elif position == 'RB':
        # Calculate RB stats from filtered plays
        # Ensure required columns exist before attempting calculations
        required_rb_cols = ['rush_attempt', 'rushing_yards', 'rush_touchdown', 'first_down_rush', 'fumble_lost']
        if not all(col in filtered_plays.columns for col in required_rb_cols):
            logger.warning(f"[get_pos_stats] Missing required RB columns in filtered_plays. Available: {filtered_plays.columns.tolist()}")
            return {} # Return empty if required columns are missing

        rushing_plays = filtered_plays[filtered_plays['rush_attempt'] == 1]
        total_rushes = len(rushing_plays)
        rushing_yards = rushing_plays['rushing_yards'].fillna(0).sum()
        rushing_tds = rushing_plays['rush_touchdown'].fillna(0).sum()
        first_downs = rushing_plays['first_down_rush'].fillna(0).sum()
        # Use 'fumble_lost' here
        fumbles = rushing_plays['fumble_lost'].fillna(0).sum()

        # Calculate derived stats
        yards_per_carry = (rushing_yards / total_rushes) if total_rushes > 0 else 0
        rush_touchdown_rate = (rushing_tds / total_rushes * 100) if total_rushes > 0 else 0
        first_down_rate = (first_downs / total_rushes * 100) if total_rushes > 0 else 0
        fumble_rate = (fumbles / total_rushes * 100) if total_rushes > 0 else 0

        stats.update({
            'total_rush_attempts': int(total_rushes),
            'total_rushing_yards': float(rushing_yards),
            'rushing_tds': int(rushing_tds),
            'rushing_first_downs': int(first_downs),
            # Note: This now counts fumbles_lost
            'fumbles': int(fumbles),
            'yards_per_carry': float(yards_per_carry),
            'rush_touchdown_rate': float(rush_touchdown_rate),
            'first_down_rate': float(first_down_rate),
            'fumble_rate': float(fumble_rate)
        })

        # Log calculated RB stats
        logger.info(f"[get_pos_stats RB] Rushes: {total_rushes}, RushYds: {rushing_yards}, RushTDs: {rushing_tds}, FumLost: {fumbles}")

    elif position in ['WR', 'TE']:
        # Calculate receiving stats from filtered plays
        # Ensure required columns exist
        required_rec_cols = ['pass_attempt', 'complete_pass', 'receiving_yards', 'pass_touchdown', 'first_down_pass', 'yards_after_catch', 'incomplete_pass']
        # Player ID columns checked separately
        if not all(col in filtered_plays.columns for col in required_rec_cols):
             logger.warning(f"[get_pos_stats] Missing required Receiving columns in filtered_plays. Available: {filtered_plays.columns.tolist()}")
             return {}
        
        # Check for player ID columns
        receiver_id_col = None
        if 'receiver_player_id' in filtered_plays.columns:
             receiver_id_col = 'receiver_player_id'
        elif 'receiver_id' in filtered_plays.columns:
             receiver_id_col = 'receiver_id'
        
        if not receiver_id_col:
             logger.warning("[get_pos_stats] Missing receiver ID column (receiver_player_id or receiver_id).")
             return {}

        receiving_plays = filtered_plays[
            (filtered_plays['pass_attempt'] == 1) &  # Only include actual pass plays
            (filtered_plays[receiver_id_col] == player_id) # Match on available player ID column
        ]
        
        # Log for debugging
        logger.info(f"Found {len(receiving_plays)} receiving plays for player ID {player_id} using column {receiver_id_col}")
        
        # Total targets = number of plays where this player was the intended receiver
        total_targets = len(receiving_plays)
        
        # Total receptions = sum of complete_pass
        total_receptions = receiving_plays['complete_pass'].fillna(0).sum()
        
        # Total receiving yards and touchdowns
        total_receiving_yards = receiving_plays['receiving_yards'].fillna(0).sum()
        
        # Count touchdowns - using pass_touchdown field to correctly identify receiving TDs
        receiving_tds = receiving_plays[
            (receiving_plays['complete_pass'] == 1) & 
            (receiving_plays['pass_touchdown'] == 1)
        ].shape[0]
        
        # Log touchdown plays for debugging
        td_plays = receiving_plays[
            (receiving_plays['complete_pass'] == 1) & 
            (receiving_plays['pass_touchdown'] == 1)
        ]
        if not td_plays.empty:
            logger.info(f"Found {len(td_plays)} touchdown plays:")
            # Commenting out the line causing KeyError: 'desc'
            # for _, play in td_plays.iterrows():
            #     logger.info(f"TD Play - Game: {play['game_id']}, Desc: {play['desc']}")
        
        first_downs = receiving_plays['first_down_pass'].fillna(0).sum()
        yards_after_catch = receiving_plays['yards_after_catch'].fillna(0).sum()
        incomplete_passes = receiving_plays['incomplete_pass'].fillna(0).sum()
        
        # Calculate derived stats
        catch_rate = (total_receptions / total_targets * 100) if total_targets > 0 else 0
        yards_per_reception = (total_receiving_yards / total_receptions) if total_receptions > 0 else 0
        yards_per_target = (total_receiving_yards / total_targets) if total_targets > 0 else 0
        touchdown_rate = (receiving_tds / total_targets * 100) if total_targets > 0 else 0
        first_down_rate = (first_downs / total_receptions * 100) if total_receptions > 0 else 0
        yac_per_reception = (yards_after_catch / total_receptions) if total_receptions > 0 else 0
        
        stats.update({
            'total_targets': int(total_targets),
            'total_receptions': int(total_receptions),
            'total_receiving_yards': float(total_receiving_yards),
            'receiving_tds': int(receiving_tds),
            'receiving_first_downs': int(first_downs),
            'yards_after_catch': float(yards_after_catch),
            'incomplete_passes': int(incomplete_passes),
            'catch_rate': float(catch_rate),
            'yards_per_reception': float(yards_per_reception),
            'yards_per_target': float(yards_per_target),
            'touchdown_rate': float(touchdown_rate),
            'first_down_rate': float(first_down_rate),
            'yac_per_reception': float(yac_per_reception)
        })
        
        # Log calculated Receiving stats
        logger.info(f"[get_pos_stats Rec] Targets: {total_targets}, Rec: {total_receptions}, RecYds: {total_receiving_yards}, RecTDs: {receiving_tds}")

        # Verify with weekly data if available and no situation filters are applied
        if weekly_data is not None and player_id is not None and not situation_filters:
            player_weekly = weekly_data[weekly_data['player_id'] == player_id]
            if not player_weekly.empty:
                weekly_totals = player_weekly.sum()
                
                # Verify key stats against weekly data
                # Allow 2% variance due to potential data discrepancies
                for pbp_stat, weekly_stat in [
                    ('total_receiving_yards', 'receiving_yards'),
                    ('total_receptions', 'receptions'),
                    ('total_targets', 'targets')
                ]:
                    if weekly_stat in weekly_totals.index:
                        weekly_value = float(weekly_totals[weekly_stat])
                        pbp_value = float(stats[pbp_stat])
                        
                        # If values differ by more than 2%, use weekly value
                        if weekly_value > 0 and abs(pbp_value - weekly_value) / weekly_value > 0.02:
                            stats[pbp_stat] = weekly_value
                            # Recalculate derived stats
                            if pbp_stat == 'total_receiving_yards':
                                stats['yards_per_reception'] = weekly_value / max(stats['total_receptions'], 1)
                                stats['yards_per_target'] = weekly_value / max(stats['total_targets'], 1)
                            elif pbp_stat == 'total_receptions':
                                stats['catch_rate'] = (weekly_value / max(stats['total_targets'], 1)) * 100
                                stats['yards_per_reception'] = stats['total_receiving_yards'] / max(weekly_value, 1)
                            elif pbp_stat == 'total_targets':
                                stats['catch_rate'] = (stats['total_receptions'] / max(weekly_value, 1)) * 100
                                stats['yards_per_target'] = stats['total_receiving_yards'] / max(weekly_value, 1)
    
    # Log final stats dictionary before returning
    logger.info(f"[get_pos_stats] Returning stats: {stats}")
    return stats

# Position-specific stat mappings
POSITION_STATS = {
    'QB': [
        'passing_yards', 'passing_tds', 'interceptions', 'completion_percentage',
        'passer_rating', 'rushing_yards', 'rushing_tds', 'sacks', 'qb_hits',
        'pressure_rate', 'time_to_throw', 'air_yards_per_attempt'
    ],
    'RB': [
        'rushing_yards', 'rushing_tds', 'yards_per_carry', 'broken_tackles',
        'receiving_yards', 'receptions', 'targets', 'yards_after_catch',
        'red_zone_touches', 'first_downs'
    ],
    'WR': [
        'receiving_yards', 'receiving_tds', 'receptions', 'targets',
        'yards_per_reception', 'yards_after_catch', 'drops', 'contested_catches',
        'red_zone_targets', 'routes_run', 'average_separation'
    ],
    'TE': [
        'receiving_yards', 'receiving_tds', 'receptions', 'targets',
        'yards_per_reception', 'yards_after_catch', 'drops', 'contested_catches',
        'red_zone_targets', 'blocking_grade'
    ],
    'DEF': [
        'tackles', 'sacks', 'interceptions', 'passes_defended',
        'forced_fumbles', 'fumble_recoveries', 'tackles_for_loss',
        'qb_hits', 'pressures', 'missed_tackles'
    ]
}

def get_player_info(player_name: str) -> Dict:
    """Get basic player information including position, team, etc."""
    players_df = import_players()
    player = players_df[players_df['display_name'].str.lower() == player_name.lower()].iloc[0]
    return {
        'player_id': player['gsis_id'],
        'position': player['position'],
        'team': player['team_abbr'],
        'age': calculate_age(player['birth_date']) if 'birth_date' in player else None,
        'experience': player['years_of_experience']
    }

def calculate_age(birth_date_str: Optional[str]) -> Optional[int]:
    """Calculate age from birth date string.
    
    Args:
        birth_date_str: Birth date in YYYY-MM-DD format
        
    Returns:
        Age in years or None if birth_date_str is invalid
    """
    if not birth_date_str or pd.isna(birth_date_str):
        return None
    try:
        birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except (ValueError, TypeError):
        return None

def get_player_position(player_id: str) -> Optional[str]:
    """Get position for a player given their GSIS ID."""
    if not player_id:
        return None
    try:
        player_file_path = "/Users/shaanchanchani/dev/nfl-data-api/cache/players.parquet"
        players_df = pd.read_parquet(player_file_path)
        player_match = players_df[players_df['gsis_id'] == player_id]
        if not player_match.empty:
             position = player_match.iloc[0].get('position')
             return position if pd.notna(position) else None
        else:
             return None
    except FileNotFoundError:
        logger.error(f"Player data file not found at: {player_file_path} in get_player_position")
        return None
    except Exception as e:
        logger.error(f"Error in get_player_position for {player_id}: {e}")
        return None 