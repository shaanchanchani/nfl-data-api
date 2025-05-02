"""
Enhanced NFL Player Stats Calculator

A Python implementation of nflverse's calculate_stats function
with added flexibility for position-specific stats and various aggregation methods.
Supports QB, RB, WR, and other position-specific statistics.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
import logging # Added logging import
import sys # For function name detection

# Set up logger
logger = logging.getLogger(__name__)

# Debug helper function
def debug_player_positions():
    """Print debug information about player positions in the database"""
    try:
        players_df = pd.read_parquet(Path("cache/players.parquet"))
        unique_positions = sorted(players_df['position'].dropna().unique().tolist())
        print(f"DEBUG: Unique positions in players database: {unique_positions}")
        
        # Count by position
        pos_counts = players_df['position'].value_counts().to_dict()
        print(f"DEBUG: Position counts: {pos_counts}")
        
        # Check for ambiguous positions
        for pos in ['WR', 'TE']:
            players = players_df[players_df['position'] == pos]['display_name'].tolist()[:5]
            print(f"DEBUG: Sample {pos} players: {players}")
    except Exception as e:
        print(f"DEBUG: Error analyzing player positions: {e}")

# Import the consolidated resolve_player from stats_helpers
from src.nfl_data.stats_helpers import resolve_player, get_player_position
from src.nfl_data.qb_helpers import calculate_qb_stats
from src.nfl_data.rb_helpers import calculate_rb_stats
from src.nfl_data.wr_helpers import calculate_wr_stats

# Removed streak dependency

# --- Synchronous Player Resolution Helper --- 

def get_player_ids(player_names: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Get the GSIS ID(s) for one or more players given their display name(s).
    
    Args:
        player_names: Single player name as string or list of player names
        
    Returns:
        Single GSIS ID as string or list of GSIS IDs corresponding to input names.
        Returns None for any players not found.
    """
    players_df = pd.read_parquet(Path("cache/players.parquet"))
    # Handle single name case
    if isinstance(player_names, str):
        player_match = players_df[players_df['display_name'] == player_names]
        return player_match.iloc[0]['gsis_id'] if len(player_match) > 0 else None
        
    # Handle list of names case
    player_ids = []
    for name in player_names:
        player_match = players_df[players_df['display_name'] == name]
        player_ids.append(player_match.iloc[0]['gsis_id'] if len(player_match) > 0 else None)
    
    return player_ids

def get_player_position(player_id: str) -> Optional[str]:
    """
    Get position for a player given their GSIS ID.
    
    Args:
        player_id: GSIS ID for the player
        
    Returns:
        Position as string or None if player not found
    """
    players_df = pd.read_parquet(Path("cache/players.parquet"))
    player_match = players_df[players_df['gsis_id'] == player_id]
    return player_match.iloc[0]['position'] if len(player_match) > 0 else None

def filter_by_situation(
    pbp: pd.DataFrame,
    downs: Optional[List[int]] = None,
    yards_to_go: Optional[List[int]] = None,
    field_position: Optional[str] = None,
    score_differential: Optional[List[int]] = None,
    formation: Optional[str] = None,
    personnel_package: Optional[str] = None,
    defense_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter play-by-play data by various situational criteria.
    
    Parameters:
    -----------
    pbp : pd.DataFrame
        Play-by-play data
    downs : list of int, optional
        List of downs to filter by (e.g., [3] for 3rd down only)
    yards_to_go : list of int, optional
        Range of yards to go (e.g., [1, 2] for short yardage situations)
    field_position : str, optional
        Field position description ('redzone' for inside 20, 'own_half', 'opp_half', etc.)
    score_differential : list of int, optional
        Range of score differential to filter by, e.g., [-100, 0] for trailing
    formation : str, optional
        Offensive formation to filter by
    personnel_package : str, optional
        Offensive personnel package to filter by (e.g., '11' for 1 RB, 1 TE, 3 WR)
    defense_type : str, optional
        Defensive scheme to filter by
        
    Returns:
    --------
    pd.DataFrame
        Filtered play-by-play data
    """
    filtered_pbp = pbp.copy()
    
    # Filter by downs
    if downs is not None:
        filtered_pbp = filtered_pbp[filtered_pbp['down'].isin(downs)]
        
    # Filter by yards to go
    if yards_to_go is not None:
        if len(yards_to_go) == 2:
            filtered_pbp = filtered_pbp[
                (filtered_pbp['ydstogo'] >= yards_to_go[0]) & 
                (filtered_pbp['ydstogo'] <= yards_to_go[1])
            ]
        else:
            filtered_pbp = filtered_pbp[filtered_pbp['ydstogo'].isin(yards_to_go)]
    
    # Filter by field position
    if field_position is not None:
        if field_position.lower() == 'redzone':
            filtered_pbp = filtered_pbp[filtered_pbp['yardline_100'] <= 20]
        elif field_position.lower() == 'own_half':
            filtered_pbp = filtered_pbp[filtered_pbp['yardline_100'] > 50]
        elif field_position.lower() == 'opp_half':
            filtered_pbp = filtered_pbp[filtered_pbp['yardline_100'] <= 50]
        # Add more field position filters as needed
    
    # Filter by score differential
    if score_differential is not None:
        if len(score_differential) == 2:
            filtered_pbp = filtered_pbp[
                (filtered_pbp['score_differential'] >= score_differential[0]) & 
                (filtered_pbp['score_differential'] <= score_differential[1])
            ]
    
    # Filter by formation, personnel package, and defense type
    # These filters would depend on the specific columns in your dataset
    if formation is not None and 'formation' in filtered_pbp.columns:
        filtered_pbp = filtered_pbp[filtered_pbp['formation'] == formation]
        
    if personnel_package is not None and 'personnel_package' in filtered_pbp.columns:
        filtered_pbp = filtered_pbp[filtered_pbp['personnel_package'] == personnel_package]
        
    if defense_type is not None and 'defense_type' in filtered_pbp.columns:
        filtered_pbp = filtered_pbp[filtered_pbp['defense_type'] == defense_type]
    
    return filtered_pbp

def calculate_player_stats(
    pbp: pd.DataFrame,
    player_name: Optional[str] = None, # Changed from player_id
    team: Optional[str] = None,         # Added team for resolution
    position: Optional[str] = None,
    aggregation_type: str = "season",
    seasons: Union[List[int], int, None] = None,
    week: Optional[int] = None,
    season_type: str = "REG",
    redzone_only: bool = False,
    add_player_name: bool = True,
    downs: Optional[List[int]] = None,
    opponent_team: Optional[str] = None,
    score_differential_range: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Universal function to compute stats for any player position based on NFL play-by-play data.
    
    Parameters:
    -----------
    pbp : pd.DataFrame
        Play-by-play data
    player_name : str, optional
        The display name of the player.
    team : str, optional
        Optional team abbreviation (e.g., "KC") to filter matches.
    position : str, optional
        Player position ('QB', 'RB', 'WR', 'TE'). If not provided, will try to detect based on player_id
    aggregation_type : str
        One of 'season', 'week', or 'career'
    seasons : int or list of int, optional
        Season(s) to include in the analysis
    week : int, optional
        Specific week to filter for (when aggregation_type is 'week')
    season_type : str
        One of 'REG', 'POST', or 'REG+POST'
    redzone_only : bool, default False
        If True, only include plays where yardline_100 <= 20 (redzone plays)
    downs : list of int, optional
        Downs to filter by (e.g., [3] for third down only)
    opponent_team : str, optional
        Filter stats to plays against a specific opponent team
    score_differential_range : list of int, optional
        Range of score differential to filter by, e.g., [-100, 0] for trailing
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing player stats according to the specified parameters and position
    """
    # Input validation
    if aggregation_type not in ["season", "week", "career"]:
        raise ValueError("aggregation_type must be one of 'season', 'week', or 'career'")
    
    if season_type not in ["REG", "POST", "REG+POST"]:
        raise ValueError("season_type must be one of 'REG', 'POST', or 'REG+POST'")
    
    resolved_player_id: Optional[str] = None
    if player_name:
        # Use the imported resolve_player from stats_helpers
        player_data, _ = resolve_player(name=player_name, team=team)
        if player_data:
             resolved_player_id = player_data.get('gsis_id')
        
        if resolved_player_id is None:
             # Log instead of print
             logger.warning(f"Could not resolve player '{player_name}' (Team: {team}). Returning empty DataFrame.")
             # Consider raising a specific exception instead?
             return pd.DataFrame() # Player not found or ambiguous

    # If player_id is provided but position is not, try to detect position
    # Use resolved_player_id here
    if resolved_player_id is not None and position is None:
        # Use our helper function to get the position
        # Updated: Directly use the imported get_player_position
        position = get_player_position(resolved_player_id)
        if position:
            print(f"Detected position for player {resolved_player_id}: {position}")
        else:
            print(f"Could not detect position automatically for player ID: {resolved_player_id}")
            print("Please provide a position parameter ('QB', 'RB', 'WR', 'TE')")
    
    if position is None:
        # If still no position, try to detect based on play involvement in filtered data
        filtered_pbp = pbp.copy()
        
        # Handle seasons parameter
        if seasons is not None:
            if isinstance(seasons, int):
                seasons = [seasons]
            filtered_pbp = filtered_pbp[filtered_pbp['season'].isin(seasons)]
        
        # Use resolved_player_id here
        if resolved_player_id is not None:
            # Count occurrences in different player ID columns
            passer_count = filtered_pbp[filtered_pbp['passer_player_id'] == resolved_player_id].shape[0]
            rusher_count = filtered_pbp[filtered_pbp['rusher_player_id'] == resolved_player_id].shape[0]
            receiver_count = filtered_pbp[filtered_pbp['receiver_player_id'] == resolved_player_id].shape[0]
            
            # Determine position based on most frequent role
            if passer_count > rusher_count and passer_count > receiver_count:
                position = 'QB'
                print(f"Detected position for player {resolved_player_id} based on play role: QB")
            elif rusher_count > passer_count and rusher_count > receiver_count:
                position = 'RB'
                print(f"Detected position for player {resolved_player_id} based on play role: RB")
            elif receiver_count > 0:
                position = 'WR'  # Default to WR if they have any receptions
                print(f"Detected position for player {resolved_player_id} based on play role: WR")
            else:
                raise ValueError("Could not detect position for player. Please provide a position parameter.")
        else:
            raise ValueError("Either player_id or position must be provided.")
    
    # Now call the appropriate function based on position
    if position.upper() in ['QB', 'QUARTERBACK']:
        return calculate_qb_stats(
            pbp=pbp,
            player_id=resolved_player_id, # Pass resolved ID
            aggregation_type=aggregation_type,
            seasons=seasons,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            add_player_name=add_player_name,
            downs=downs,
            opponent_team=opponent_team,
            score_differential_range=score_differential_range
        )
    elif position.upper() in ['RB', 'HB', 'FB', 'RUNNINGBACK', 'HALFBACK', 'FULLBACK']:
        return calculate_rb_stats(
            pbp=pbp,
            player_id=resolved_player_id, # Pass resolved ID
            aggregation_type=aggregation_type,
            seasons=seasons,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            add_player_name=add_player_name,
            downs=downs,
            opponent_team=opponent_team,
            score_differential_range=score_differential_range
        )
    elif position.upper() in ['WR', 'TE', 'WIDERECEIVER', 'TIGHTEND']:
        return calculate_wr_stats(
            pbp=pbp,
            player_id=resolved_player_id, # Pass resolved ID
            aggregation_type=aggregation_type,
            seasons=seasons,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            add_player_name=add_player_name,
            downs=downs,
            opponent_team=opponent_team,
            score_differential_range=score_differential_range
        )
    else:
        raise ValueError(f"Unsupported position: {position}. Supported positions are QB, RB, WR, and TE.")

def get_top_players(
    pbp_path: str = 'cache/play_by_play_condensed.parquet',
    position: str = 'QB',
    n: int = 10,
    sort_by: str = None,
    min_threshold: Dict[str, int] = None,
    ascending: bool = False,
    aggregation_type: str = "season",
    seasons: Union[List[int], int, None] = None,
    week: Optional[int] = None,
    season_type: str = "REG",
    redzone_only: bool = False,
    include_player_details: bool = True,
    downs: Optional[List[int]] = None,
    opponent_team: Optional[str] = None,
    score_differential_range: Optional[List[int]] = None
) -> pd.DataFrame:
    # Debug player positions at the beginning
    debug_player_positions()
    """
    Get top N players of a specified position sorted by a statistical category.
    
    Parameters:
    -----------
    pbp_path : str
        Path to the play-by-play parquet file
    position : str
        Player position to analyze ('QB', 'RB', 'WR', 'TE')
    n : int, default 10
        Number of records to return
    sort_by : str, default None
        Column to sort by. If None, will use a position-appropriate default:
        - QB: 'epa_per_dropback'
        - RB: 'rushing_yards'
        - WR/TE: 'receiving_yards'
    min_threshold : dict, optional
        Dictionary with minimum values for columns to filter players
        Example: {'attempts': 100} for QBs with at least 100 pass attempts
    ascending : bool, default False
        Sort order. False means highest values first (descending)
    aggregation_type : str
        One of 'season', 'week', or 'career'
    seasons : int or list of int, optional
        Season(s) to include in the analysis
    week : int, optional
        Specific week to filter for (when aggregation_type is 'week')
    season_type : str
        One of 'REG', 'POST', or 'REG+POST'
    redzone_only : bool, default False
        If True, only include plays where yardline_100 <= 20 (redzone plays)
    include_player_details : bool, default True
        If True, will add additional player details (full name, team, etc.)
    downs : list of int, optional
        Downs to filter by (e.g., [3] for third down only)
    opponent_team : str, optional
        Filter stats to plays against a specific opponent team
    score_differential_range : list of int, optional
        Range of score differential to filter by, e.g., [-100, 0] for trailing
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing top N players sorted by the specified column
    """
    # Load play-by-play data
    pbp = pd.read_parquet(pbp_path)
    
    # Define valid positions and their mappings
    QB_POSITIONS = ['QB', 'QUARTERBACK']
    RB_POSITIONS = ['RB', 'HB', 'FB', 'RUNNINGBACK', 'HALFBACK', 'FULLBACK']
    WR_POSITIONS = ['WR', 'WIDERECEIVER']
    TE_POSITIONS = ['TE', 'TIGHTEND']
    position_upper = position.upper()
    
    print(f"DEBUG: Position provided: {position} (uppercase: {position_upper})")
    print(f"DEBUG: Position type: {type(position)}")
    
    # Default sort columns by position if not specified
    if sort_by is None:
        if position_upper in QB_POSITIONS:
            sort_by = 'epa_per_dropback'
        elif position_upper in RB_POSITIONS:
            sort_by = 'rushing_yards'
        elif position_upper in WR_POSITIONS or position_upper in TE_POSITIONS:
            sort_by = 'receiving_yards'
    
    # Default minimum thresholds by position if not specified
    if min_threshold is None:
        # Use different thresholds based on aggregation type
        if aggregation_type == "week":
            # Use lower thresholds for game-level stats
            if position_upper in QB_POSITIONS:
                min_threshold = {'qb_dropback': 10}  # At least 10 dropbacks in a game
            elif position_upper in RB_POSITIONS:
                min_threshold = {'carries': 5}       # At least 5 carries in a game
            elif position_upper in WR_POSITIONS or position_upper in TE_POSITIONS:
                min_threshold = {'targets': 3}       # At least 3 targets in a game
        else:
            # Use higher thresholds for season or career aggregation
            if position_upper in QB_POSITIONS:
                min_threshold = {'qb_dropback': 100}
            elif position_upper in RB_POSITIONS:
                min_threshold = {'carries': 50}
            elif position_upper in WR_POSITIONS or position_upper in TE_POSITIONS:
                min_threshold = {'targets': 30}
    
    # Calculate stats for the specific position
    position_stats = None
    
    # Standardized position for return
    standardized_position = position
    
    if position_upper in QB_POSITIONS:
        # IMPORTANT: For QBs specifically, ensure all the parameters are defined correctly
        print(f"DEBUG: Getting QB stats with position={position_upper}")
        position_stats = calculate_qb_stats(
            pbp=pbp,
            aggregation_type=aggregation_type,
            seasons=seasons,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            downs=downs,
            opponent_team=opponent_team,
            score_differential_range=score_differential_range
        )
        
        # Load the players dataframe to get accurate positions
        players_df = pd.read_parquet(Path("cache/players.parquet"))
        
        # Create a mapping of player IDs to their actual positions and position groups
        player_position_map = dict(zip(players_df['gsis_id'], players_df['position']))
        player_position_group_map = dict(zip(players_df['gsis_id'], players_df['position_group']))
        
        # Add the actual position column from the database
        if 'player_id' in position_stats.columns:
            position_stats.loc[:, 'actual_position'] = position_stats['player_id'].map(
                lambda x: player_position_map.get(x, "UNKNOWN")
            )
            position_stats.loc[:, 'actual_position_group'] = position_stats['player_id'].map(
                lambda x: player_position_group_map.get(x, "UNKNOWN")
            )
            
            # Print debug info about position groups before filtering
            if len(position_stats) > 0:
                position_group_counts = position_stats['actual_position_group'].value_counts().to_dict()
                print(f"DEBUG: Position group counts before filtering: {position_group_counts}")
                
            # Filter ONLY to include QBs by position group
            position_stats = position_stats[
                position_stats['actual_position_group'].str.upper() == "QB"
            ]
            
            # Now set the display position
            position_stats.loc[:, 'position'] = "QB"  # Explicitly set position for all players in this result
        else:
            position_stats.loc[:, 'position'] = "QB"
            
        standardized_position = "QB"
        
        # Always add QB rushing stats, regardless of sort_by
        if len(position_stats) > 0:
            print(f"DEBUG: Adding QB rushing stats for all QBs")
            
            # Make a fresh copy of the PBP data
            rushing_pbp = pbp.copy()
            
            # Apply the same filters
            if seasons is not None:
                if isinstance(seasons, int):
                    seasons = [seasons]
                rushing_pbp = rushing_pbp[rushing_pbp['season'].isin(seasons)]
            
            if season_type in ["REG", "POST"]:
                rushing_pbp = rushing_pbp[rushing_pbp['season_type'] == season_type]
            
            if week is not None and aggregation_type == "week":
                rushing_pbp = rushing_pbp[rushing_pbp['week'] == week]
            
            if redzone_only:
                rushing_pbp = rushing_pbp[rushing_pbp['yardline_100'] <= 20]
                
            if downs is not None:
                rushing_pbp = rushing_pbp[rushing_pbp['down'].isin(downs)]
            
            if opponent_team is not None:
                rushing_pbp = rushing_pbp[rushing_pbp['defteam'] == opponent_team]
                
            if score_differential_range is not None:
                min_diff, max_diff = score_differential_range
                rushing_pbp = rushing_pbp[(rushing_pbp['score_differential'] >= min_diff) & 
                                         (rushing_pbp['score_differential'] <= max_diff)]
            
            # Set grouping variables based on aggregation_type
            if aggregation_type == "season":
                rusher_group_cols = ['season', 'rusher_player_id', 'posteam']
            elif aggregation_type == "week":
                rusher_group_cols = ['season', 'week', 'rusher_player_id', 'posteam']
            else:  # career
                rusher_group_cols = ['rusher_player_id', 'posteam']
            
            # Filter for QB rusher plays
            qb_ids = position_stats['player_id'].tolist()
            rusher_filter = ~rushing_pbp['rusher_player_id'].isna() & rushing_pbp['rusher_player_id'].isin(qb_ids) & (rushing_pbp['play_type'] == 'run')
            
            print(f"DEBUG: Found {rusher_filter.sum()} QB rushing plays")
            
            if rusher_filter.sum() > 0:
                # Calculate rushing stats for QBs
                rushing_stats = (
                    rushing_pbp[rusher_filter]
                    .groupby(rusher_group_cols)
                    .agg({
                        'yards_gained': 'sum',
                        'touchdown': 'sum',
                        'rush_touchdown': 'sum',
                        'first_down_rush': 'sum',
                        'fumble': 'sum',
                        'fumble_lost': 'sum',
                        'epa': lambda x: x.sum(skipna=True)
                    })
                    .reset_index()
                    .rename(columns={
                        'rusher_player_id': 'player_id',
                        'posteam': 'team',
                        'yards_gained': 'rushing_yards',
                        'rush_touchdown': 'rushing_tds',
                        'first_down_rush': 'rushing_first_downs',
                        'fumble': 'rushing_fumbles',
                        'fumble_lost': 'rushing_fumbles_lost',
                        'epa': 'rushing_epa'
                    })
                )
                
                # Calculate carries
                carries_count = (
                    rushing_pbp[rusher_filter]
                    .groupby(rusher_group_cols)
                    .size()
                    .reset_index()
                    .rename(columns={0: 'carries'})
                )
                
                # Rename columns to match the rushing_stats DataFrame
                rename_dict = {}
                if 'rusher_player_id' in carries_count.columns:
                    rename_dict['rusher_player_id'] = 'player_id'
                if 'posteam' in carries_count.columns:
                    rename_dict['posteam'] = 'team'
                carries_count = carries_count.rename(columns=rename_dict)
                
                # Determine merge columns
                merge_cols = []
                for col in rusher_group_cols:
                    if col == 'rusher_player_id':
                        merge_cols.append('player_id')
                    elif col == 'posteam':
                        merge_cols.append('team')
                    else:
                        merge_cols.append(col)
                
                # Merge carries with rushing stats
                rushing_stats = rushing_stats.merge(
                    carries_count,
                    on=merge_cols,
                    how='left'
                )
                
                # Add yards_per_carry
                rushing_stats['yards_per_carry'] = np.where(
                    rushing_stats['carries'] > 0,
                    rushing_stats['rushing_yards'] / rushing_stats['carries'],
                    np.nan
                )
                
                # Merge rushing stats with position_stats
                merge_cols = ['player_id']
                if 'season' in position_stats.columns and 'season' in rushing_stats.columns:
                    merge_cols.append('season')
                if 'week' in position_stats.columns and 'week' in rushing_stats.columns:
                    merge_cols.append('week')
                if 'team' in position_stats.columns and 'team' in rushing_stats.columns:
                    merge_cols.append('team')
                
                # Merge with position_stats using suffixes to avoid column duplicates
                position_stats = position_stats.merge(
                    rushing_stats,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_rushing')
                )
                
                # Handle duplicate columns (with _rushing suffix)
                rushing_cols = [col for col in position_stats.columns if col.endswith('_rushing')]
                for col in rushing_cols:
                    base_col = col.replace('_rushing', '')
                    # Use the rushing data and remove the suffix column
                    position_stats.loc[:, base_col] = position_stats[col]
                    position_stats = position_stats.drop(columns=[col])
                    print(f"DEBUG: Using data from {col} for {base_col}")
                
                # Fill NAs with 0
                for col in rushing_stats.columns:
                    if col not in merge_cols and col in position_stats.columns:
                        position_stats.loc[:, col] = position_stats[col].fillna(0)
                
                # Ensure rushing columns exist before calculating fantasy points
                rushing_columns = [
                    'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions'
                ]
                
                # Initialize any missing rushing columns needed for fantasy points with zeros
                for col in rushing_columns:
                    if col not in position_stats.columns:
                        position_stats.loc[:, col] = 0
                        print(f"DEBUG: Added missing column {col} for fantasy points calculation")
                
                # Update fantasy points to include rushing
                position_stats.loc[:, 'fantasy_points'] = (
                    (1/25) * position_stats['passing_yards'] +
                    4 * position_stats['passing_tds'] +
                    -2 * position_stats['passing_interceptions'] +
                    -2 * position_stats['passing_fumbles_lost'] +
                    2 * position_stats['passing_2pt_conversions'] +
                    # Add rushing points
                    (1/10) * position_stats['rushing_yards'] +
                    6 * position_stats['rushing_tds'] +
                    -2 * position_stats['rushing_fumbles_lost'] +
                    2 * position_stats.get('rushing_2pt_conversions', 0)
                )
            else:
                # Add empty rushing columns with zeros
                rushing_columns = [
                    'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
                    'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
                    'carries', 'yards_per_carry', 'rushing_2pt_conversions'
                ]
                
                # Ensure all rushing columns exist
                for col in rushing_columns:
                    if col not in position_stats.columns:
                        position_stats.loc[:, col] = 0
                        print(f"DEBUG: Added missing column {col} in else branch")
                
                # Also add fantasy points calculation for this branch
                position_stats.loc[:, 'fantasy_points'] = (
                    (1/25) * position_stats['passing_yards'] +
                    4 * position_stats['passing_tds'] +
                    -2 * position_stats['passing_interceptions'] +
                    -2 * position_stats['passing_fumbles_lost'] +
                    2 * position_stats['passing_2pt_conversions'] +
                    # Add rushing points (all zeros in this case)
                    (1/10) * position_stats['rushing_yards'] +
                    6 * position_stats['rushing_tds'] +
                    -2 * position_stats['rushing_fumbles_lost'] +
                    2 * position_stats.get('rushing_2pt_conversions', 0)
                )
                
            # Ensure all rushing columns exist at this point regardless of flow path
            rushing_columns = [
                'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
                'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
                'carries', 'yards_per_carry', 'rushing_2pt_conversions'
            ]
            
            # Initialize any missing rushing columns with zeros
            for col in rushing_columns:
                if col not in position_stats.columns:
                    position_stats.loc[:, col] = 0
            
            # Clean up any duplicate columns with _x or _y suffixes
            for col in rushing_columns:
                # Check for _x and _y versions of the column
                if f"{col}_x" in position_stats.columns:
                    # If we have a _y version, use that (it's from the rushing stats)
                    if f"{col}_y" in position_stats.columns:
                        position_stats.loc[:, col] = position_stats[f"{col}_y"]
                        position_stats = position_stats.drop(columns=[f"{col}_x", f"{col}_y"])
                        print(f"DEBUG: Fixed duplicate column {col} by using _y value and dropping both _x and _y")
                    else:
                        # If we only have _x, rename it
                        position_stats.loc[:, col] = position_stats[f"{col}_x"]
                        position_stats = position_stats.drop(columns=[f"{col}_x"])
                        print(f"DEBUG: Fixed duplicate column {col} by removing _x suffix")
                # If we only have _y, rename it
                elif f"{col}_y" in position_stats.columns:
                    position_stats.loc[:, col] = position_stats[f"{col}_y"]
                    position_stats = position_stats.drop(columns=[f"{col}_y"])
                    print(f"DEBUG: Fixed duplicate column {col} by removing _y suffix")
            
            # Debugging to verify rushing columns are created
            print(f"DEBUG: Rushing columns after initialization: {[col for col in rushing_columns if col in position_stats.columns]}")
            
            # Final check - make sure no columns with _x or _y suffixes remain for our rushing columns
            for col in rushing_columns:
                assert f"{col}_x" not in position_stats.columns, f"Column {col}_x still exists"
                assert f"{col}_y" not in position_stats.columns, f"Column {col}_y still exists"
                    
            print(f"DEBUG: After adding QB rushing stats, position_stats columns: {position_stats.columns.tolist()}")
    elif position_upper in RB_POSITIONS:
        position_stats = calculate_rb_stats(
            pbp=pbp,
            aggregation_type=aggregation_type,
            seasons=seasons,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            downs=downs,
            opponent_team=opponent_team,
            score_differential_range=score_differential_range
        )
        
        # Load the players dataframe to get accurate positions
        players_df = pd.read_parquet(Path("cache/players.parquet"))
        
        # Create a mapping of player IDs to their actual positions and position groups
        player_position_map = dict(zip(players_df['gsis_id'], players_df['position']))
        player_position_group_map = dict(zip(players_df['gsis_id'], players_df['position_group']))
        
        # Add the actual position column from the database
        if 'player_id' in position_stats.columns:
            position_stats.loc[:, 'actual_position'] = position_stats['player_id'].map(
                lambda x: player_position_map.get(x, "UNKNOWN")
            )
            position_stats.loc[:, 'actual_position_group'] = position_stats['player_id'].map(
                lambda x: player_position_group_map.get(x, "UNKNOWN")
            )
            
            # Print debug info about position groups before filtering
            if len(position_stats) > 0:
                position_group_counts = position_stats['actual_position_group'].value_counts().to_dict()
                print(f"DEBUG: Position group counts before filtering: {position_group_counts}")
                
            # Filter ONLY to include RBs by position group
            position_stats = position_stats[
                position_stats['actual_position_group'].str.upper() == "RB"
            ]
            
            # Now set the display position
            position_stats.loc[:, 'position'] = "RB"  # Explicitly set position for all players in this result
        else:
            position_stats.loc[:, 'position'] = "RB"
            
        standardized_position = "RB"
    elif position_upper in WR_POSITIONS:
        print(f"DEBUG: Getting WR stats with position={position_upper}")
        position_stats = calculate_wr_stats(
            pbp=pbp,
            aggregation_type=aggregation_type,
            seasons=seasons,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            downs=downs,
            opponent_team=opponent_team,
            score_differential_range=score_differential_range
        )
        
        # Load the players dataframe to get accurate positions
        players_df = pd.read_parquet(Path("cache/players.parquet"))
        
        # Create a mapping of player IDs to their actual positions and position groups
        player_position_map = dict(zip(players_df['gsis_id'], players_df['position']))
        player_position_group_map = dict(zip(players_df['gsis_id'], players_df['position_group']))
        
        # Add the actual position column from the database
        if 'player_id' in position_stats.columns:
            position_stats.loc[:, 'actual_position'] = position_stats['player_id'].map(
                lambda x: player_position_map.get(x, "UNKNOWN")
            )
            position_stats.loc[:, 'actual_position_group'] = position_stats['player_id'].map(
                lambda x: player_position_group_map.get(x, "UNKNOWN")
            )
            
            # Print debug info about position groups before filtering
            if len(position_stats) > 0:
                position_group_counts = position_stats['actual_position_group'].value_counts().to_dict()
                print(f"DEBUG: Position group counts before filtering: {position_group_counts}")
                
            # Filter ONLY to include WRs by position group
            position_stats = position_stats[
                position_stats['actual_position_group'].str.upper() == "WR"
            ]
            print(f"DEBUG: After strict WR filtering (excluding TEs): {len(position_stats)} rows")
            
            # Now set the display position
            position_stats.loc[:, 'position'] = "WR"  # Explicitly set position for all players in this result
        else:
            position_stats.loc[:, 'position'] = "WR"
            
        standardized_position = "WR"
        print(f"DEBUG: After WR stats calculation, columns: {position_stats.columns.tolist()}")
    elif position_upper in TE_POSITIONS:
        # For TEs, we use the same function as WRs but may apply different thresholds
        position_stats = calculate_wr_stats(
            pbp=pbp,
            aggregation_type=aggregation_type,
            seasons=seasons,
            week=week,
            season_type=season_type,
            redzone_only=redzone_only,
            downs=downs,
            opponent_team=opponent_team,
            score_differential_range=score_differential_range
        )
        
        # Load the players dataframe to get accurate positions
        players_df = pd.read_parquet(Path("cache/players.parquet"))
        
        # Create a mapping of player IDs to their actual positions and position groups
        player_position_map = dict(zip(players_df['gsis_id'], players_df['position']))
        player_position_group_map = dict(zip(players_df['gsis_id'], players_df['position_group']))
        
        # Add the actual position column from the database
        if 'player_id' in position_stats.columns:
            position_stats.loc[:, 'actual_position'] = position_stats['player_id'].map(
                lambda x: player_position_map.get(x, "UNKNOWN")
            )
            position_stats.loc[:, 'actual_position_group'] = position_stats['player_id'].map(
                lambda x: player_position_group_map.get(x, "UNKNOWN")
            )
            
            # Print debug info about position groups before filtering
            if len(position_stats) > 0:
                position_group_counts = position_stats['actual_position_group'].value_counts().to_dict()
                print(f"DEBUG: Position group counts before filtering: {position_group_counts}")
                
            # Filter ONLY to include TEs by position group 
            position_stats = position_stats[
                position_stats['actual_position_group'].str.upper() == "TE"
            ]
            print(f"DEBUG: After strict TE filtering (excluding WRs): {len(position_stats)} rows")
            
            # Now set the display position
            position_stats.loc[:, 'position'] = "TE"  # Explicitly set position for all players in this result
        else:
            position_stats.loc[:, 'position'] = "TE"
            
        standardized_position = "TE"
    else:
        # If position is not recognized, raise a meaningful error
        valid_positions = QB_POSITIONS + RB_POSITIONS + WR_POSITIONS + TE_POSITIONS
        raise ValueError(f"Unsupported position: '{position}'. Supported positions are: {', '.join(sorted(set(valid_positions)))}")
    
    if position_stats is None:
        print(f"No stats found for position: {position} - position_stats is None")
        return pd.DataFrame()
    elif len(position_stats) == 0:
        print(f"No stats found for position: {position} - position_stats is empty DataFrame")
        print(f"Aggregation type: {aggregation_type}, Season: {seasons}, Week: {week}")
        return pd.DataFrame()
    
    # For QBs, handle rushing-specific sort columns
    if position_upper in QB_POSITIONS and sort_by.startswith('rushing_'):
        # If the sort column doesn't exist, we need to calculate QB rushing stats and merge them
        if sort_by not in position_stats.columns:
            # Get QB rushing stats for the same players
            print(f"DEBUG: Calculating QB rushing stats for sort column '{sort_by}'")
            
            # Make a fresh copy of the PBP data
            rushing_pbp = pbp.copy()
            
            # Apply the same filters
            if seasons is not None:
                if isinstance(seasons, int):
                    seasons = [seasons]
                rushing_pbp = rushing_pbp[rushing_pbp['season'].isin(seasons)]
            
            if season_type in ["REG", "POST"]:
                rushing_pbp = rushing_pbp[rushing_pbp['season_type'] == season_type]
            
            if week is not None and aggregation_type == "week":
                rushing_pbp = rushing_pbp[rushing_pbp['week'] == week]
            
            if redzone_only:
                rushing_pbp = rushing_pbp[rushing_pbp['yardline_100'] <= 20]
                
            if downs is not None:
                rushing_pbp = rushing_pbp[rushing_pbp['down'].isin(downs)]
            
            if opponent_team is not None:
                rushing_pbp = rushing_pbp[rushing_pbp['defteam'] == opponent_team]
                
            if score_differential_range is not None:
                min_diff, max_diff = score_differential_range
                rushing_pbp = rushing_pbp[(rushing_pbp['score_differential'] >= min_diff) & 
                                         (rushing_pbp['score_differential'] <= max_diff)]
            
            # Set grouping variables based on aggregation_type
            if aggregation_type == "season":
                rusher_group_cols = ['season', 'rusher_player_id', 'posteam']
            elif aggregation_type == "week":
                rusher_group_cols = ['season', 'week', 'rusher_player_id', 'posteam']
            else:  # career
                rusher_group_cols = ['rusher_player_id', 'posteam']
            
            # Filter for QB rusher plays
            qb_ids = position_stats['player_id'].tolist()
            rusher_filter = ~rushing_pbp['rusher_player_id'].isna() & rushing_pbp['rusher_player_id'].isin(qb_ids) & (rushing_pbp['play_type'] == 'run')
            
            print(f"DEBUG: Found {rusher_filter.sum()} QB rushing plays")
            
            if rusher_filter.sum() > 0:
                # Calculate rushing stats for QBs
                rushing_stats = (
                    rushing_pbp[rusher_filter]
                    .groupby(rusher_group_cols)
                    .agg({
                        'yards_gained': 'sum',
                        'touchdown': 'sum',
                        'rush_touchdown': 'sum',
                        'first_down_rush': 'sum',
                        'fumble': 'sum',
                        'fumble_lost': 'sum',
                        'epa': lambda x: x.sum(skipna=True)
                    })
                    .reset_index()
                    .rename(columns={
                        'rusher_player_id': 'player_id',
                        'posteam': 'team',
                        'yards_gained': 'rushing_yards',
                        'rush_touchdown': 'rushing_tds',
                        'first_down_rush': 'rushing_first_downs',
                        'fumble': 'rushing_fumbles',
                        'fumble_lost': 'rushing_fumbles_lost',
                        'epa': 'rushing_epa'
                    })
                )
                
                # Calculate carries
                carries_count = (
                    rushing_pbp[rusher_filter]
                    .groupby(rusher_group_cols)
                    .size()
                    .reset_index()
                    .rename(columns={0: 'carries'})
                )
                
                # Rename columns to match the rushing_stats DataFrame
                rename_dict = {}
                if 'rusher_player_id' in carries_count.columns:
                    rename_dict['rusher_player_id'] = 'player_id'
                if 'posteam' in carries_count.columns:
                    rename_dict['posteam'] = 'team'
                carries_count = carries_count.rename(columns=rename_dict)
                
                # Determine merge columns
                merge_cols = []
                for col in rusher_group_cols:
                    if col == 'rusher_player_id':
                        merge_cols.append('player_id')
                    elif col == 'posteam':
                        merge_cols.append('team')
                    else:
                        merge_cols.append(col)
                
                # Merge carries with rushing stats
                rushing_stats = rushing_stats.merge(
                    carries_count,
                    on=merge_cols,
                    how='left'
                )
                
                # Merge rushing stats with position_stats
                merge_cols = ['player_id']
                if 'season' in position_stats.columns and 'season' in rushing_stats.columns:
                    merge_cols.append('season')
                if 'week' in position_stats.columns and 'week' in rushing_stats.columns:
                    merge_cols.append('week')
                if 'team' in position_stats.columns and 'team' in rushing_stats.columns:
                    merge_cols.append('team')
                
                # Merge with position_stats using suffixes to avoid column duplicates
                position_stats = position_stats.merge(
                    rushing_stats,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_rushing')
                )
                
                # Handle duplicate columns (with _rushing suffix)
                rushing_cols = [col for col in position_stats.columns if col.endswith('_rushing')]
                for col in rushing_cols:
                    base_col = col.replace('_rushing', '')
                    # Use the rushing data and remove the suffix column
                    position_stats.loc[:, base_col] = position_stats[col]
                    position_stats = position_stats.drop(columns=[col])
                    print(f"DEBUG: Using data from {col} for {base_col}")
                
                # Fill NAs with 0
                for col in rushing_stats.columns:
                    if col not in merge_cols and col in position_stats.columns:
                        position_stats.loc[:, col] = position_stats[col].fillna(0)
            else:
                # If no rushing plays found, add all rushing columns with zeros to avoid KeyError
                position_stats.loc[:, 'rushing_yards'] = 0
                position_stats.loc[:, 'rushing_tds'] = 0
                position_stats.loc[:, 'rushing_first_downs'] = 0
                position_stats.loc[:, 'rushing_fumbles'] = 0
                position_stats.loc[:, 'rushing_fumbles_lost'] = 0
                position_stats.loc[:, 'rushing_epa'] = 0
                position_stats.loc[:, 'carries'] = 0
                position_stats.loc[:, 'yards_per_carry'] = 0
                
                # Also add the specific sort_by column if needed
                if sort_by not in position_stats.columns:
                    position_stats.loc[:, sort_by] = 0
                
            print(f"DEBUG: After adding QB rushing stats, position_stats columns: {position_stats.columns.tolist()}")
    
    # Validate sort_by column exists
    if sort_by not in position_stats.columns:
        raise ValueError(f"sort_by column '{sort_by}' not found in {position} stats. Available columns: {position_stats.columns.tolist()}")
    
    # Apply minimum thresholds
    filtered_stats = position_stats.copy()
    if min_threshold:
        for col, threshold in min_threshold.items():
            if col in filtered_stats.columns:
                filtered_stats = filtered_stats[filtered_stats[col] >= threshold]
            else:
                print(f"Warning: Column '{col}' not found in {position} stats for threshold filtering.")
    
    if len(filtered_stats) == 0:
        thresholds_str = ", ".join([f"{col} >= {val}" for col, val in min_threshold.items()])
        print(f"No {position}s found meeting the minimum thresholds: {thresholds_str}. Try lowering the thresholds.")
        return pd.DataFrame()
    
    # Debug: Print filtered stats information 
    print(f"DEBUG: After applying thresholds:")
    print(f"  - {len(filtered_stats)} rows remaining")
    if len(filtered_stats) > 0:
        print(f"  - Columns: {filtered_stats.columns.tolist()}")
        print(f"  - Sample sort values for {sort_by}: {filtered_stats[sort_by].head(3).tolist()}")
    
    # Special handling for week aggregation type
    if aggregation_type == "week" and week is None:
        # When no specific week is requested, find the top N individual game performances
        print(f"DEBUG: Using special week aggregation handling (no specific week)")
        
        # Make sure we're tracking the player ID column
        id_cols = [col for col in filtered_stats.columns if col.endswith('player_id')]
        if id_cols:
            player_id_col = id_cols[0]
            print(f"DEBUG: Found player ID column: {player_id_col}")
            
            # First drop duplicates if necessary
            if 'player_id' in filtered_stats.columns and 'season' in filtered_stats.columns and 'week' in filtered_stats.columns:
                filtered_stats = filtered_stats.drop_duplicates(subset=['player_id', 'season', 'week'])
                print(f"DEBUG: Removed duplicates based on player_id, season, and week, {len(filtered_stats)} rows remaining")
                
            # Sort all individual games (week-level stats) and get top N
            top_players = filtered_stats.sort_values(
                by=sort_by,
                ascending=ascending,
                na_position='last'
            ).head(n)
            print(f"DEBUG: After sorting, got {len(top_players)} top players for week aggregation")
        else:
            # If we can't identify player ID column, return empty DataFrame
            print("Error: Can't find player ID column for week aggregation")
            return pd.DataFrame()
    else:
        # For season, career, or specific week aggregation, sort and get top N
        # Drop duplicates based on appropriate fields for the aggregation type
        if 'player_id' in filtered_stats.columns:
            if 'season' in filtered_stats.columns and aggregation_type == 'season':
                filtered_stats = filtered_stats.drop_duplicates(subset=['player_id', 'season'])
                print(f"DEBUG: Removed duplicates based on player_id and season, {len(filtered_stats)} rows remaining")
            elif aggregation_type == 'career':
                filtered_stats = filtered_stats.drop_duplicates(subset=['player_id'])
                print(f"DEBUG: Removed duplicates based on player_id for career stats, {len(filtered_stats)} rows remaining")
        
        # Sort the data by the specified column
        sorted_stats = filtered_stats.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position='last'
        )
        
        # Position filtering is now done at source when calculating the stats
        # We've already ensured each position_stats dataframe only contains players with the correct position
        # No need for additional filtering here
        
        # Get the top N players after all filtering
        top_players = sorted_stats.head(n)
        print(f"DEBUG: After sorting, got {len(top_players)} top players for {aggregation_type} aggregation")
    
    # Always add player names from players.parquet (regardless of include_player_details setting)
    if not top_players.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            
            # Create a mapping of player IDs to display names 
            player_name_map = dict(zip(players_df['gsis_id'], players_df['display_name']))
            
            # Add the display name as player_name column
            top_players.loc[:, 'player_name'] = top_players['player_id'].map(
                lambda x: player_name_map.get(x, f"Unknown ({x})")
            )
            
            # Add more detailed player information if requested
            if include_player_details:
                # Create a mapping of player IDs to full player information
                player_info = players_df[
                    players_df['gsis_id'].isin(top_players['player_id'])
                ].set_index('gsis_id')
                
                # Add relevant columns from player_info
                detail_columns = ['first_name', 'last_name', 'position_group',
                                 'college_name', 'height', 'weight', 'birth_date', 
                                 'draft_club', 'draft_number', 'team_abbr', 'headshot']
                
                # Only include columns that exist in the players dataframe
                available_columns = [col for col in detail_columns if col in player_info.columns]
                
                for col in available_columns:
                    top_players.loc[:, f'player_{col}'] = top_players['player_id'].map(
                        lambda x: player_info.loc[x, col] if x in player_info.index else None
                    )
                
                # Add database position as player_position but preserve the original position
                if 'position' in player_info.columns:
                    top_players.loc[:, 'player_position'] = top_players['player_id'].map(
                        lambda x: player_info.loc[x, 'position'] if x in player_info.index else None
                    )
                
        except Exception as e:
            print(f"Warning: Could not add player details - {e}")
    
    return top_players

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load play-by-play data
    pbp = pd.read_parquet('cache/play_by_play_condensed.parquet')
    
    print("===== Player Identification Examples =====")
    
    # Example 1: Get player ID by name
    mahomes_id = get_player_ids("Patrick Mahomes")
    print(f"Patrick Mahomes ID: {mahomes_id}")
    
    # Example 2: Get player position from ID
    mahomes_position = get_player_position(mahomes_id)
    print(f"Patrick Mahomes Position: {mahomes_position}")
    
    # Example 3: Get multiple player IDs
    player_names = ["Josh Allen", "Travis Kelce", "Tyreek Hill"]
    player_ids = get_player_ids(player_names)
    print(f"Multiple Player IDs:")
    for name, pid in zip(player_names, player_ids):
        print(f"{name}: {pid}")
    
    print("\n===== QB Stats Examples =====")
    
    # Example 4: QB - Get stats by player name instead of ID
    mahomes_id = get_player_ids("Patrick Mahomes")
    mahomes_season = calculate_player_stats(
        pbp=pbp,
        player_name="Patrick Mahomes",
        team="KC",
        position="QB",
        aggregation_type="season",
        seasons=[2023]
    )
    print("\nMahomes 2023 Season Stats:")
    if not mahomes_season.empty:
        # Print columns to check what's available
        print(f"Available columns: {mahomes_season.columns.tolist()}")
        # Try to print stats without player_name column
        print(mahomes_season[['attempts', 'completions', 'passing_yards', 'passing_tds', 'epa_per_dropback']])
    else:
        print("No data found for Patrick Mahomes. Check if the player ID is correct.")
    
    # Example 5: Get top 5 QBs by EPA per dropback in 2023
    top_qbs_2023 = get_top_players(
        position="QB",
        n=5,
        sort_by="epa_per_dropback",
        seasons=2023,
        min_threshold={"qb_dropback": 200}  # Minimum 200 dropbacks to qualify
    )
    print("\nTop 5 QBs by EPA per dropback in 2023:")
    print(f"Available columns: {top_qbs_2023.columns.tolist()}")
    # Print stats using available columns for names
    display_cols = ['epa_per_dropback', 'qb_dropback', 'passing_yards']
    if 'player_name' in top_qbs_2023.columns:
        display_cols.insert(0, 'player_name')
    print(top_qbs_2023[display_cols])
    
    print("\n===== RB Stats Examples =====")
    
    # Example 6: RB - Get stats by player name
    mccaffrey_id = get_player_ids("Christian McCaffrey")
    if mccaffrey_id:
        mccaffrey_season = calculate_player_stats(
            pbp=pbp,
            player_name="Christian McCaffrey",
            team="CAR",
            position="RB",
            aggregation_type="season",
            seasons=[2023]
        )
        print("\nChristian McCaffrey 2023 Season Stats:")
        if not mccaffrey_season.empty:
            print(f"Available columns: {mccaffrey_season.columns.tolist()}")
            print(mccaffrey_season[['carries', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards']])
        else:
            print("No data found for Christian McCaffrey. Check season and filters.")
    else:
        print("Player 'Christian McCaffrey' not found in the players database.")
    
    # Example 7: Get top 5 RBs by rushing yards in 2023 with detailed player info
    top_rbs_2023 = get_top_players(
        position="RB",
        n=5,
        sort_by="rushing_yards",
        seasons=2023,
        min_threshold={"carries": 100},  # Minimum 100 carries to qualify
        include_player_details=True
    )
    print("\nTop 5 RBs by rushing yards in 2023 (with player details):")
    print(f"Available columns: {top_rbs_2023.columns.tolist()}")
    
    # Create a list of columns that we know should exist
    display_cols = ['rushing_yards', 'carries', 'rushing_tds', 'receptions', 'receiving_yards']
    
    # Add player name column if it exists
    if 'player_name' in top_rbs_2023.columns:
        display_cols.insert(0, 'player_name')
            
    # Add college if it exists
    if 'player_college' in top_rbs_2023.columns:
        display_cols.append('player_college')
        
    print(top_rbs_2023[display_cols])
    
    print("\n===== WR Stats Examples =====")
    
    # Example 8: WR - Get stats by player name
    hill_id = get_player_ids("Tyreek Hill")
    if hill_id:
        hill_season = calculate_player_stats(
            pbp=pbp,
            player_name="Tyreek Hill",
            team="KC",
            position="WR",
            aggregation_type="season",
            seasons=[2023]
        )
        print("\nTyreek Hill 2023 Season Stats:")
        if not hill_season.empty:
            print(hill_season[['player_name', 'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'target_share']])
        else:
            print("No data found for Tyreek Hill. Check season and filters.")
    else:
        print("Player 'Tyreek Hill' not found in the players database.")
    
    # Example 9: Get top 5 WRs by receiving yards in 2023
    top_wrs_2023 = get_top_players(
        position="WR",
        n=5,
        sort_by="receiving_yards",
        seasons=2023,
        min_threshold={"targets": 50}  # Minimum 50 targets to qualify
    )
    print("\nTop 5 WRs by receiving yards in 2023:")
    print(f"Available columns: {top_wrs_2023.columns.tolist()}")
    print(top_wrs_2023[['player_name', 'receiving_yards', 'receptions', 'targets', 'receiving_tds', 'target_share']])
    
    # Example 10: Situational stats - Red zone performance by player name
    # First get a top RB from our previous results
    if not top_rbs_2023.empty:
        top_rb_id = top_rbs_2023.iloc[0]['player_id']
        top_rb_name = top_rbs_2023.iloc[0]['player_name']
        
        rb_redzone = calculate_player_stats(
            pbp=pbp,
            player_id=top_rb_id,
            position="RB",
            aggregation_type="season",
            seasons=[2023],
            redzone_only=True
        )
        print(f"\n{top_rb_name} Red Zone Performance in 2023:")
        if not rb_redzone.empty:
            print(rb_redzone[['player_name', 'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']])
        else:
            print("No red zone data found for this player. Check season and filters.")
    else:
        print("No top RBs found to show red zone performance.")