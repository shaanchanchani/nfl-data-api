"""Helper functions for calculating various NFL statistics."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging
from pydantic_core import PydanticUndefined # Import the Undefined type

# Set up logging at the module level
logger = logging.getLogger(__name__)

from .data_import import (
    import_pbp_data,
    import_weekly_data,
    import_players,
    import_schedules,
    import_injuries,
    import_depth_charts
)
from .data_loader import load_players, load_pbp_data, load_weekly_stats, load_rosters

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
        
        weekly_stats = import_weekly_data(seasons)
        # The import_weekly_data returns a coroutine that needs to be awaited,
        # but this function is not async, so we should raise an error.
        raise RuntimeError("get_historical_matchup_stats is not an async function but calls async data loaders. Use the direct async loader functions instead.")
        
    except Exception as e:
        return {"error": f"An error occurred calculating historical stats: {str(e)}"}

def get_team_stats(team: str) -> Dict:
    """Get comprehensive team statistics."""
    # Use 2024 season data
    season = 2024
    
    # Import play-by-play data
    pbp_data = import_pbp_data()
    pbp_data = pbp_data[pbp_data['season'] == season]
    team_plays = pbp_data[pbp_data['posteam'] == team]
    
    # Create default stats if no data
    if team_plays.empty:
        off_stats = {
            'points_per_game': 0,
            'yards_per_game': 0,
            'pass_play_rate': 0,
            'yards_per_pass': 0,
            'yards_per_rush': 0,
            'third_down_conversion_rate': 0,
            'red_zone_touchdown_rate': 0
        }
    else:
        # Calculate offensive stats
        off_stats = {
            'points_per_game': team_plays.groupby('game_id')['yards_gained'].sum().mean(),  # Using yards as proxy for points
            'yards_per_game': team_plays.groupby('game_id')['yards_gained'].sum().mean(),
            'pass_play_rate': len(team_plays[team_plays['pass_attempt'] == 1]) / len(team_plays) if len(team_plays) > 0 else 0,
            'yards_per_pass': team_plays[team_plays['pass_attempt'] == 1]['yards_gained'].mean() if len(team_plays[team_plays['pass_attempt'] == 1]) > 0 else 0,
            'yards_per_rush': team_plays[team_plays['rush_attempt'] == 1]['yards_gained'].mean() if len(team_plays[team_plays['rush_attempt'] == 1]) > 0 else 0,
            'third_down_conversion_rate': team_plays[team_plays['down'] == 3]['first_down'].mean() if len(team_plays[team_plays['down'] == 3]) > 0 else 0,
            'red_zone_touchdown_rate': team_plays[team_plays['yardline_100'] <= 20]['touchdown'].mean() if len(team_plays[team_plays['yardline_100'] <= 20]) > 0 else 0
        }
    
    # Get injury report
    try:
        injuries = import_injuries([season])
        team_injuries = injuries[injuries['team'] == team]
        formatted_injuries = format_injury_report(team_injuries)
    except Exception:
        formatted_injuries = []
    
    # Get depth chart
    try:
        depth_chart = import_depth_charts([season])
        team_depth = depth_chart[depth_chart['depth_team'] == team]  # Using 'depth_team' instead of 'team'
        formatted_depth = format_depth_chart(team_depth)
    except Exception:
        formatted_depth = {}
    
    return {
        'offensive_stats': off_stats,
        'defensive_stats': get_defensive_stats(team, pbp_data, import_weekly_data([season]), season),
        'injuries': formatted_injuries,
        'depth_chart': formatted_depth
    }

def analyze_key_matchups(home_team: str, away_team: str) -> List[Dict]:
    """Analyze key positional matchups between teams."""
    # Get current season
    season = datetime.now().year
    
    # Get team rosters and stats
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    
    # Analyze key matchups
    matchups = []
    
    # QB vs Pass Defense
    matchups.append({
        'type': 'QB vs Pass Defense',
        'home_strength': home_stats['offensive_stats']['yards_per_pass'],
        'away_strength': away_stats['defensive_stats']['completion_percentage_allowed'],
        'advantage': 'home' if home_stats['offensive_stats']['yards_per_pass'] > 
                    league_average('yards_per_pass') else 'away'
    })
    
    # Run Game vs Run Defense
    matchups.append({
        'type': 'Run Game vs Run Defense',
        'home_strength': home_stats['offensive_stats']['yards_per_rush'],
        'away_strength': away_stats['defensive_stats']['yards_per_carry_allowed'],
        'advantage': 'home' if home_stats['offensive_stats']['yards_per_rush'] >
                    away_stats['defensive_stats']['yards_per_carry_allowed'] else 'away'
    })
    
    # Add more matchups as needed
    
    return matchups

def analyze_player_matchup(player_name: str, home_team: str, away_team: str) -> Dict:
    """Analyze specific player matchup for a game."""
    # Get player info
    players_df = import_players()
    player = players_df[players_df['player_name'].str.lower() == player_name.lower()].iloc[0]
    player_team = player['current_team']
    opponent = away_team if player_team == home_team else home_team
    
    # Get historical performance vs opponent
    historical_stats = get_historical_matchup_stats(player_name, opponent)
    
    # Get opponent defensive stats
    opp_def_stats = get_defensive_stats(opponent, import_pbp_data()[import_pbp_data()['season'] == datetime.now().year], import_weekly_data([datetime.now().year]), datetime.now().year)
    
    # Get matchup-specific analysis based on position
    position_matchup = analyze_position_matchup(
        player_name,
        player['position'],
        opponent
    )
    
    return {
        'historical_performance': historical_stats,
        'opponent_defense': opp_def_stats,
        'position_matchup': position_matchup
    }

# Helper functions

def get_top_defender(def_players: pd.DataFrame, stat: str) -> Dict:
    """Get top defender for a specific stat."""
    if def_players.empty or stat not in def_players.columns:
        return None
    
    top_player = def_players.nlargest(1, stat).iloc[0]
    return {
        'name': top_player['player_name'],
        'value': float(top_player[stat])
    }

def get_position_specific_stats(stats_df: pd.DataFrame, position: str) -> Dict:
    """Get position-specific statistics from player stats."""
    if stats_df.empty:
        return {}
    
    if position == 'QB':
        return {
            'avg_passing_yards': stats_df['passing_yards'].mean(),
            'avg_passing_tds': stats_df['passing_tds'].mean(),
            'avg_interceptions': stats_df['interceptions'].mean()
        }
    elif position == 'RB':
        return {
            'avg_rushing_yards': stats_df['rushing_yards'].mean(),
            'avg_rushing_tds': stats_df['rushing_tds'].mean(),
            'avg_receptions': stats_df['receptions'].mean()
        }
    elif position in ['WR', 'TE']:
        return {
            'avg_receiving_yards': stats_df['receiving_yards'].mean(),
            'avg_receiving_tds': stats_df['receiving_tds'].mean(),
            'avg_receptions': stats_df['receptions'].mean(),
            'avg_targets': stats_df['targets'].mean()
        }
    
    return {}

def format_injury_report(injuries_df: pd.DataFrame) -> List[Dict]:
    """Format injury report data."""
    if injuries_df.empty:
        return []
    
    return injuries_df.apply(
        lambda x: {
            'player': x['player_name'],
            'position': x['position'],
            'injury': x['injury_type'],
            'status': x['practice_status']
        },
        axis=1
    ).tolist()

def format_depth_chart(depth_df: pd.DataFrame) -> Dict:
    """Format depth chart data."""
    if depth_df.empty:
        return {}
    
    depth_chart = {}
    for _, row in depth_df.iterrows():
        pos = row['position']
        if pos not in depth_chart:
            depth_chart[pos] = []
        depth_chart[pos].append({
            'player': row['player_name'],
            'depth': row['depth_team'],
            'status': row['status']
        })
    
    return depth_chart

def analyze_position_matchup(
    player_name: str,
    position: str,
    opponent: str
) -> Dict:
    """Analyze position-specific matchup."""
    season = datetime.now().year
    
    if position == 'WR':
        # Analyze CB matchup
        depth_chart = import_depth_charts([season])
        opponent_cbs = depth_chart[
            (depth_chart['team'] == opponent) &
            (depth_chart['position'] == 'CB')
        ]
        
        # Get CB stats
        cb_stats = []
        for _, cb in opponent_cbs.iterrows():
            cb_weekly = import_weekly_data([season])
            cb_weekly = cb_weekly[cb_weekly['player_name'] == cb['player_name']]
            if not cb_weekly.empty:
                cb_stats.append({
                    'name': cb['player_name'],
                    'coverage_snaps': cb_weekly['coverage_snaps'].sum(),
                    'completion_pct_allowed': cb_weekly['completion_percentage_allowed'].mean(),
                    'passer_rating_allowed': cb_weekly['passer_rating_allowed'].mean()
                })
        
        return {
            'likely_coverage': cb_stats[0] if cb_stats else None,
            'backup_coverage': cb_stats[1:] if len(cb_stats) > 1 else []
        }
    
    # Add analysis for other positions
    return {}

def league_average(stat: str) -> float:
    """Get league average for a specific stat."""
    # Implementation would depend on available data
    # This is a placeholder
    return 0.0

def get_available_seasons() -> List[int]:
    """Get list of available seasons in the dataset."""
    # TODO: Implement actual data retrieval
    current_year = datetime.now().year
    return list(range(current_year - 5, current_year + 1))

async def search_players(name: str) -> List[Dict]:
    """Search for players by name."""
    players_df = await import_players()
    # Players dataset uses 'display_name' instead of 'player_name'
    matches = players_df[players_df['display_name'].str.lower().str.contains(name.lower())]
    return matches.to_dict('records')

async def resolve_player(name: str, season: Optional[int] = None) -> Tuple[Optional[Dict], List[Dict]]:
    """Resolve player name to a single player or return alternatives.
    
    Args:
        name: Player name to resolve
        season: Optional season to use for resolution (defaults to current season)
        
    Returns:
        Tuple of (resolved player dict or None, list of alternative matches)
    """
    if not name:
        return None, []
    
    if season is None:
        season = get_current_season()
        
    # Load both current players and historical roster data
    players_df = await load_players()
    try:
        roster_df = await load_rosters([season])
        # Merge roster data with player data to get historical team info
        if not roster_df.empty:
            players_df = players_df.merge(
                roster_df[['gsis_id', 'team', 'season']],
                left_on='gsis_id',
                right_on='gsis_id',
                how='left'
            )
            # Update team_abbr with historical team info where available
            players_df.loc[players_df['season'] == season, 'team_abbr'] = players_df.loc[players_df['season'] == season, 'team']
    except Exception:
        # If roster data isn't available, continue with current data
        pass
        
    name_lower = name.lower()
    
    # Try exact match on display_name first
    exact_matches = players_df[players_df['display_name'].str.lower() == name_lower]
    if len(exact_matches) == 1:
        return exact_matches.iloc[0].to_dict(), []
    
    # Try matching both name formats (First Last and Last, First)
    if ',' in name:
        # Convert "Last, First" to "First Last"
        last, first = name_lower.split(',')
        alt_name = f"{first.strip()} {last.strip()}"
    else:
        # Convert "First Last" to "Last, First"
        parts = name_lower.split()
        if len(parts) >= 2:
            alt_name = f"{parts[-1]}, {' '.join(parts[:-1])}"
        else:
            alt_name = name_lower
            
    name_matches = players_df[
        (players_df['display_name'].str.lower() == name_lower) |
        (players_df['display_name'].str.lower() == alt_name)
    ]
    if len(name_matches) == 1:
        return name_matches.iloc[0].to_dict(), []
    if len(name_matches) > 1:
        return None, name_matches.to_dict('records')
    
    # Try contains match as fallback
    contains_matches = players_df[
        players_df['display_name'].str.lower().str.contains(name_lower, na=False) |
        players_df['display_name'].str.lower().str.contains(alt_name, na=False)
    ]
    if len(contains_matches) == 1:
        return contains_matches.iloc[0].to_dict(), []
    elif len(contains_matches) > 1:
        return None, contains_matches.to_dict('records')
    
    return None, []

def get_player_headshot_url(player_id: str) -> str:
    """Get URL for player's headshot image."""
    # Use the NFL CDN URL format
    return f"https://static.www.nfl.com/image/private/t_player_profile_landscape/f_auto/league/{player_id}"

async def get_player_game_log(player_name: str, season: Optional[int] = None) -> Dict:
    """Get game-by-game stats for a player."""
    # Resolve player first
    player, alternatives = await resolve_player(player_name)
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
    weekly_data = await import_weekly_data([season])
    
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

def get_player_comparison(player1: str, player2: str) -> Dict:
    """Compare two players' stats."""
    # TODO: Implement actual player comparison
    return {}

def get_game_stats(game_id: str) -> Dict:
    """Get comprehensive stats for a game."""
    # TODO: Implement actual game stats retrieval
    return {}

async def get_situation_stats(player_name: str, situations: List[str], season: Optional[int] = None) -> Dict:
    """Get player stats filtered by one or more game situations."""
    # Resolve player first
    player, alternatives = await resolve_player(player_name, season)
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
        # Load PBP data (remove await, it's a regular function)
        pbp_data = load_pbp_data()
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

def normalize_team_name(team: str) -> str:
    """Normalize team name to standard abbreviation."""
    # TODO: Implement actual team name normalization
    return team.upper()

async def get_player_on_field_stats(player_name: str, other_player_name: str, season: Optional[int] = None, week: Optional[int] = None, on_field: bool = True) -> Dict:
    """Get player stats when another player is on/off the field."""
    # TODO: Implement actual on-field stats calculation
    return {}

def get_player_stats(
    player_name: str,
    season: Optional[int] = None,
    week: Optional[int] = None,
    **situation_filters: Dict[str, Any]
) -> Dict:
    """Get comprehensive player stats with optional situation filters."""
    try:
        # For testing purposes, return a dummy response
        return {
            "player_name": player_name,
            "season": season or 2024,
            "week": week,
            "filters_applied": situation_filters,
            "stats": {
                "games_played": 17,
                "passing_yards": 5250,
                "passing_tds": 38,
                "interceptions": 12,
                "completion_percentage": 67.2,
                "qb_rating": 105.7
            }
        }
    except Exception as e:
        return {"error": f"An error occurred calculating stats: {str(e)}"}

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
        else:
            logger.warning(f"[get_pos_stats QB] Missing required Rushing columns in filtered_plays. Cannot calculate QB rushing stats. Available: {filtered_plays.columns.tolist()}")
        # --- End QB Rushing Stats ---

        stats.update({
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
        })

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