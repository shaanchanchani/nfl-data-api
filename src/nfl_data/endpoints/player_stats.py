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

# Import the consolidated resolve_player from stats_helpers
from src.nfl_data.stats_helpers import resolve_player, get_player_position

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

def calculate_qb_stats(
    pbp: pd.DataFrame,
    player_id: Optional[str] = None,
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
    # DEBUG: Add trace info for week-based aggregation debugging
    print(f"DEBUG: calculate_qb_stats input:")
    print(f"  - Input pbp rows: {len(pbp)}")
    print(f"  - Aggregation: {aggregation_type}, Seasons: {seasons}, Week: {week}")
    """
    Compute quarterback stats based on NFL play-by-play data.
    
    Parameters:
    -----------
    pbp : pd.DataFrame
        Play-by-play data
    player_id : str, optional
        Specific player ID to filter results for
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
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing QB stats according to specified parameters
    """
    # Add import for logging inside function or ensure it's at module level
    import logging 
    logger = logging.getLogger(__name__) # Get logger instance for this module
    
    # Input validation
    if aggregation_type not in ["season", "week", "career"]:
        raise ValueError("aggregation_type must be one of 'season', 'week', or 'career'")
    
    if season_type not in ["REG", "POST", "REG+POST"]:
        raise ValueError("season_type must be one of 'REG', 'POST', or 'REG+POST'")

    logger.debug(f"Calculating QB stats with filters - player_id: {player_id}, aggregation: {aggregation_type}, seasons: {seasons}, week: {week}, season_type: {season_type}, redzone: {redzone_only}, downs: {downs}, opponent: {opponent_team}, score_diff: {score_differential_range}")

    # Make a copy to avoid modifying the original
    filtered_pbp = pbp.copy()
    logger.debug(f"Initial PBP shape: {filtered_pbp.shape}")

    # Handle seasons parameter
    if seasons is not None:
        local_seasons = seasons
        if isinstance(local_seasons, int):
            local_seasons = [local_seasons]
        # Check data type of season column if possible
        if 'season' in filtered_pbp.columns and not filtered_pbp.empty:
            logger.debug(f"Filtering seasons: using {local_seasons} (type: {type(local_seasons[0]) if local_seasons else 'N/A'}). PBP season column type: {filtered_pbp['season'].dtype}")
        
        # Ensure seasons in PBP are compatible type before filtering
        if 'season' in filtered_pbp.columns and pd.api.types.is_numeric_dtype(filtered_pbp['season'].dtype) and local_seasons:
             # Assuming local_seasons contains integers as parsed by the endpoint
             filtered_pbp = filtered_pbp[filtered_pbp['season'].isin(local_seasons)]
        elif 'season' in filtered_pbp.columns and local_seasons: # Fallback if types might mismatch (e.g., string vs int)
             logger.warning("Attempting season filter with potentially mixed types.")
             filtered_pbp = filtered_pbp[filtered_pbp['season'].astype(str).isin([str(s) for s in local_seasons])]
        else:
             logger.warning("Could not apply season filter - 'season' column missing or empty data.")
             
        logger.debug(f"Shape after season filter: {filtered_pbp.shape}")

    # Filter by season_type if needed
    if season_type in ["REG", "POST"]:
        if 'season_type' in filtered_pbp.columns:
             original_count = len(filtered_pbp)
             filtered_pbp = filtered_pbp[filtered_pbp['season_type'] == season_type]
             logger.debug(f"Shape after season_type ('{season_type}') filter: {filtered_pbp.shape} (removed {original_count - len(filtered_pbp)} rows)")
             if len(filtered_pbp) == 0 and original_count > 0:
                 print(f"Filtering to season_type = {season_type} resulted in 0 rows. Returning empty DataFrame.")
                 logger.warning(f"Filtering to season_type = {season_type} resulted in 0 rows.")
                 return pd.DataFrame()
        else:
             logger.warning("Could not apply season_type filter - 'season_type' column missing.")

    # Filter by player_id if specified
    if player_id is not None:
         if 'passer_player_id' in filtered_pbp.columns:
             original_count = len(filtered_pbp)
             filtered_pbp = filtered_pbp[filtered_pbp['passer_player_id'] == player_id]
             logger.debug(f"Shape after player_id ('{player_id}') filter: {filtered_pbp.shape} (removed {original_count - len(filtered_pbp)} rows)")
             if len(filtered_pbp) == 0 and original_count > 0:
                 print(f"Filtering to player_id = {player_id} resulted in 0 rows. Returning empty DataFrame.")
                 logger.warning(f"Filtering to player_id = {player_id} resulted in 0 rows.")
                 return pd.DataFrame()
         else:
             logger.warning("Could not apply player_id filter - 'passer_player_id' column missing.")
             return pd.DataFrame() # Cannot calculate stats for a specific QB without ID column

    # Filter by week if specified and relevant
    if week is not None and aggregation_type == "week":
        if 'week' in filtered_pbp.columns:
             original_count = len(filtered_pbp)
             filtered_pbp = filtered_pbp[filtered_pbp['week'] == week]
             logger.debug(f"Shape after week ('{week}') filter: {filtered_pbp.shape} (removed {original_count - len(filtered_pbp)} rows)")
             if len(filtered_pbp) == 0 and original_count > 0:
                 print(f"Filtering to week = {week} resulted in 0 rows. Returning empty DataFrame.")
                 logger.warning(f"Filtering to week = {week} resulted in 0 rows.")
                 return pd.DataFrame()
        else:
             logger.warning("Could not apply week filter - 'week' column missing.")

    # Filter for redzone plays if specified
    if redzone_only:
        if 'yardline_100' in filtered_pbp.columns:
             original_count = len(filtered_pbp)
             filtered_pbp = filtered_pbp[filtered_pbp['yardline_100'] <= 20]
             logger.debug(f"Shape after redzone filter: {filtered_pbp.shape} (removed {original_count - len(filtered_pbp)} rows)")
             if len(filtered_pbp) == 0 and original_count > 0:
                 print("Filtering to redzone plays resulted in 0 rows. Returning empty DataFrame.")
                 logger.warning("Filtering to redzone plays resulted in 0 rows.")
                 return pd.DataFrame()
        else:
             logger.warning("Could not apply redzone filter - 'yardline_100' column missing.")

    # Filter by downs if specified
    if downs is not None:
        if 'down' in filtered_pbp.columns:
             original_count = len(filtered_pbp)
             filtered_pbp = filtered_pbp[filtered_pbp['down'].isin(downs)]
             logger.debug(f"Shape after downs ('{downs}') filter: {filtered_pbp.shape} (removed {original_count - len(filtered_pbp)} rows)")
             if len(filtered_pbp) == 0 and original_count > 0:
                 downs_str = ", ".join(str(d) for d in downs)
                 print(f"Filtering to downs ({downs_str}) resulted in 0 rows. Returning empty DataFrame.")
                 logger.warning(f"Filtering to downs ({downs_str}) resulted in 0 rows.")
                 return pd.DataFrame()
        else:
             logger.warning("Could not apply downs filter - 'down' column missing.")

    # Filter by opponent team if specified
    if opponent_team is not None:
        if 'defteam' in filtered_pbp.columns:
             original_count = len(filtered_pbp)
             filtered_pbp = filtered_pbp[filtered_pbp['defteam'] == opponent_team]
             logger.debug(f"Shape after opponent ('{opponent_team}') filter: {filtered_pbp.shape} (removed {original_count - len(filtered_pbp)} rows)")
             if len(filtered_pbp) == 0 and original_count > 0:
                 print(f"Filtering to opponent_team = {opponent_team} resulted in 0 rows. Returning empty DataFrame.")
                 logger.warning(f"Filtering to opponent_team = {opponent_team} resulted in 0 rows.")
                 return pd.DataFrame()
        else:
             logger.warning("Could not apply opponent team filter - 'defteam' column missing.")

    # Filter by score differential range if specified
    if score_differential_range is not None:
        if 'score_differential' in filtered_pbp.columns:
             min_diff, max_diff = score_differential_range
             original_count = len(filtered_pbp)
             filtered_pbp = filtered_pbp[(filtered_pbp['score_differential'] >= min_diff) & 
                                         (filtered_pbp['score_differential'] <= max_diff)]
             logger.debug(f"Shape after score_differential ({min_diff} to {max_diff}) filter: {filtered_pbp.shape} (removed {original_count - len(filtered_pbp)} rows)")
             if len(filtered_pbp) == 0 and original_count > 0:
                 print(f"Filtering to score_differential between {min_diff} and {max_diff} resulted in 0 rows. Returning empty DataFrame.")
                 logger.warning(f"Filtering to score_differential between {min_diff} and {max_diff} resulted in 0 rows.")
                 return pd.DataFrame()
        else:
              logger.warning("Could not apply score differential filter - 'score_differential' column missing.")

    # Set grouping variables based on aggregation_type
    if aggregation_type == "season":
        group_cols = ['season', 'passer_player_id', 'posteam']
    elif aggregation_type == "week":
        group_cols = ['season', 'week', 'passer_player_id', 'posteam']
    else:  # career - aggregate across all seasons for the player
        group_cols = ['passer_player_id', 'posteam']
    
    # Ensure all group columns exist
    missing_group_cols = [col for col in group_cols if col not in filtered_pbp.columns]
    if missing_group_cols:
        logger.error(f"Cannot group stats - missing required columns: {missing_group_cols}")
        raise ValueError(f"PBP data is missing required columns for grouping: {missing_group_cols}")

    # QB-specific passing stats
    passing_filter = filtered_pbp['play_type'].isin(['pass', 'qb_spike'])
    
    # Check if any passing plays remain after filtering
    if not passing_filter.any():
         logger.warning("No passing plays found after applying filters. Returning empty DataFrame.")
         return pd.DataFrame()
         
    logger.debug(f"Aggregating {passing_filter.sum()} passing plays using groups: {group_cols}")

    # Extract QB passing stats
    passing_stats = (
        filtered_pbp[passing_filter]
        .groupby(group_cols)
        .agg({
            'complete_pass': 'sum',
            'incomplete_pass': 'sum', 
            'interception': 'sum',
            'yards_gained': 'sum',
            'touchdown': 'sum',
            'pass_touchdown': 'sum',
            'sack': 'sum',
            'fumble': 'sum',
            'fumble_lost': 'sum',
            'qb_epa': lambda x: x.sum(skipna=True),
            'cpoe': lambda x: x.mean(skipna=True) if any(~x.isna()) else np.nan,
            'air_yards': 'sum',
            'yards_after_catch': 'sum',
            'first_down_pass': 'sum',
            'game_id': 'nunique'  # For game count
        })
        .reset_index()
        .rename(columns={
            'passer_player_id': 'player_id',
            'posteam': 'team',
            'qb_epa': 'qb_epa',
            'cpoe': 'passing_cpoe',
            'yards_gained': 'passing_yards', 
            'pass_touchdown': 'passing_tds',
            'interception': 'passing_interceptions',
            'sack': 'sacks_suffered',
            'fumble': 'passing_fumbles',
            'fumble_lost': 'passing_fumbles_lost',
            'complete_pass': 'completions',
            'incomplete_pass': 'incompletions',
            'air_yards': 'passing_air_yards',
            'yards_after_catch': 'passing_yards_after_catch',
            'first_down_pass': 'passing_first_downs',
            'game_id': 'games_played'
        })
    )
    
    # Calculate attempts as sum of completions, incompletions, and interceptions
    passing_stats['attempts'] = (
        passing_stats['completions'] + 
        passing_stats['incompletions'] + 
        passing_stats['passing_interceptions']
    )
    
    # Calculate 2-point conversions
    two_point_filter = (filtered_pbp['play_type'] == 'pass') & (filtered_pbp['two_point_conv_result'] == 'success')
    
    # Set grouping variables for 2-point conversions
    if aggregation_type == "season":
        two_point_groups = ['season', 'passer_player_id']
    elif aggregation_type == "week":
        two_point_groups = ['season', 'week', 'passer_player_id']
    else:  # career - aggregate all seasons together
        two_point_groups = ['passer_player_id']
    
    two_point_stats = (
        filtered_pbp[two_point_filter]
        .groupby(two_point_groups)
        .size()
        .reset_index()
        .rename(columns={
            0: 'passing_2pt_conversions',
            'passer_player_id': 'player_id'  # Rename to match passing_stats
        })
    )
    
    # Join with the main stats
    if two_point_stats.shape[0] > 0:
        # Determine merge columns based on aggregation_type
        if aggregation_type == "season":
            merge_cols = ['season', 'player_id']
        elif aggregation_type == "week":
            merge_cols = ['season', 'week', 'player_id']
        else:  # career - just merge on player_id to aggregate across seasons
            merge_cols = ['player_id']
            
        passing_stats = passing_stats.merge(
            two_point_stats,
            on=merge_cols,
            how='left'
        )
    else:
        passing_stats['passing_2pt_conversions'] = 0
    
    # Calculate PACR (Passing Air Conversion Ratio)
    passing_stats['pacr'] = np.where(
        passing_stats['passing_air_yards'] > 0,
        passing_stats['passing_yards'] / passing_stats['passing_air_yards'],
        np.nan
    )
    
    # Calculate completion percentage
    passing_stats['completion_pct'] = np.where(
        passing_stats['attempts'] > 0,
        100 * passing_stats['completions'] / passing_stats['attempts'],
        np.nan
    )
    
    # Calculate epa_per_dropback
    passing_stats['epa_per_dropback'] = np.where(
        (passing_stats['attempts'] + passing_stats['sacks_suffered']) > 0,
        passing_stats['qb_epa'] / (passing_stats['attempts'] + passing_stats['sacks_suffered']),
        np.nan
    )
    
    # Calculate qb_dropback
    passing_stats['qb_dropback'] = passing_stats['attempts'] + passing_stats['sacks_suffered']
    
    # Now let's get the rushing stats for QBs
    # For rushing stats, we need to get a fresh copy of the data
    rushing_pbp = pbp.copy()
    
    # Apply the same filters we applied to the passing data
    # Handle seasons parameter
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        rushing_pbp = rushing_pbp[rushing_pbp['season'].isin(seasons)]
    
    # Filter by season_type if needed
    if season_type in ["REG", "POST"]:
        rushing_pbp = rushing_pbp[rushing_pbp['season_type'] == season_type]
    
    # Filter by week if specified and relevant
    if week is not None and aggregation_type == "week":
        rushing_pbp = rushing_pbp[rushing_pbp['week'] == week]
    
    # Filter for redzone plays if specified
    if redzone_only:
        rushing_pbp = rushing_pbp[rushing_pbp['yardline_100'] <= 20]
        
    # Filter by downs if specified
    if downs is not None:
        rushing_pbp = rushing_pbp[rushing_pbp['down'].isin(downs)]
    
    # Filter by opponent team if specified
    if opponent_team is not None:
        rushing_pbp = rushing_pbp[rushing_pbp['defteam'] == opponent_team]
        
    # Filter by score differential range if specified
    if score_differential_range is not None:
        min_diff, max_diff = score_differential_range
        rushing_pbp = rushing_pbp[(rushing_pbp['score_differential'] >= min_diff) & 
                                 (rushing_pbp['score_differential'] <= max_diff)]
    
    # Set grouping variables for rushing stats
    if aggregation_type == "season":
        rusher_group_cols = ['season', 'rusher_player_id', 'posteam']
    elif aggregation_type == "week":
        rusher_group_cols = ['season', 'week', 'rusher_player_id', 'posteam']
    else:  # career
        rusher_group_cols = ['rusher_player_id', 'posteam']
    
    # Filter for rushing plays where the QB is the rusher
    if player_id is not None:
        # If we're looking for a specific QB, filter for them as a rusher
        rusher_filter = ~rushing_pbp['rusher_player_id'].isna() & (rushing_pbp['rusher_player_id'] == player_id) & (rushing_pbp['play_type'] == 'run')
    else:
        # Get all QBs from the passing stats and find their rushing plays
        rusher_filter = ~rushing_pbp['rusher_player_id'].isna() & rushing_pbp['rusher_player_id'].isin(passing_stats['player_id']) & (rushing_pbp['play_type'] == 'run')
    
    # Get rushing stats
    rushing_stats = None
    
    if rusher_filter.sum() > 0:
        # Extract features for rushes by QBs
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
        
        # Merge the carries count with rushing stats
        rushing_stats = rushing_stats.merge(
            carries_count,
            on=merge_cols,
            how='left'
        )
        
        # Calculate yards per carry
        rushing_stats['yards_per_carry'] = np.where(
            rushing_stats['carries'] > 0,
            rushing_stats['rushing_yards'] / rushing_stats['carries'],
            np.nan
        )
        
        # Calculate EPA per carry
        rushing_stats['epa_per_carry'] = np.where(
            rushing_stats['carries'] > 0,
            rushing_stats['rushing_epa'] / rushing_stats['carries'],
            np.nan
        )
        
        # Calculate 2-point conversions for rushing
        rush_2pt_filter = (rushing_pbp['play_type'] == 'run') & (rushing_pbp['two_point_conv_result'] == 'success')
        if player_id is not None:
            rush_2pt_filter &= (rushing_pbp['rusher_player_id'] == player_id)
        else:
            rush_2pt_filter &= rushing_pbp['rusher_player_id'].isin(passing_stats['player_id'])
        
        # Set grouping variables for 2-point conversions
        if aggregation_type == "season":
            rush_2pt_groups = ['season', 'rusher_player_id']
        elif aggregation_type == "week":
            rush_2pt_groups = ['season', 'week', 'rusher_player_id']
        else:  # career - aggregate all seasons together
            rush_2pt_groups = ['rusher_player_id']
        
        if rush_2pt_filter.sum() > 0:
            rush_2pt_stats = (
                rushing_pbp[rush_2pt_filter]
                .groupby(rush_2pt_groups)
                .size()
                .reset_index()
                .rename(columns={
                    0: 'rushing_2pt_conversions',
                    'rusher_player_id': 'player_id'  # Rename to match rushing_stats
                })
            )
            
            # Join with the rushing stats
            if rush_2pt_stats.shape[0] > 0:
                # Determine merge columns based on aggregation_type
                if aggregation_type == "season":
                    merge_cols = ['season', 'player_id']
                elif aggregation_type == "week":
                    merge_cols = ['season', 'week', 'player_id']
                else:  # career - just merge on player_id to aggregate across seasons
                    merge_cols = ['player_id']
                    
                rushing_stats = rushing_stats.merge(
                    rush_2pt_stats,
                    on=merge_cols,
                    how='left'
                )
            else:
                rushing_stats['rushing_2pt_conversions'] = 0
        else:
            rushing_stats['rushing_2pt_conversions'] = 0
        
        # Now merge rushing stats with passing stats
        if aggregation_type == "season":
            merge_cols = ['season', 'player_id', 'team']
        elif aggregation_type == "week":
            merge_cols = ['season', 'week', 'player_id', 'team']
        else:  # career
            merge_cols = ['player_id', 'team']
        
        # Merge rushing stats with passing stats
        if len(rushing_stats) > 0:
            passing_stats = passing_stats.merge(
                rushing_stats,
                on=merge_cols,
                how='left'
            )
            
            # Fill NaN values in rushing columns
            rushing_cols = [
                'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
                'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
                'carries', 'yards_per_carry', 'epa_per_carry', 'rushing_2pt_conversions'
            ]
            for col in rushing_cols:
                if col in passing_stats.columns:
                    passing_stats[col] = passing_stats[col].fillna(0)
    
    # Ensure all required fantasy columns exist and fill NaNs with 0 before calculation
    fantasy_required_cols = [
        'passing_yards', 'passing_tds', 'passing_interceptions',
        'passing_fumbles_lost', 'passing_2pt_conversions'
    ]
    for col in fantasy_required_cols:
        if col not in passing_stats.columns:
            logger.warning(f"Fantasy point calculation (QB): Adding missing column '{col}' with value 0.")
            passing_stats[col] = 0
        else:
            # Fill NaNs if the column already exists
            passing_stats[col] = passing_stats[col].fillna(0)
    
    # Add rushing stats to fantasy points if available
    if rushing_stats is not None and len(rushing_stats) > 0:
        # Make sure all required rushing fantasy columns exist
        rushing_fantasy_cols = [
            'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions'
        ]
        for col in rushing_fantasy_cols:
            if col not in passing_stats.columns:
                logger.warning(f"Fantasy point calculation (QB): Adding missing rushing column '{col}' with value 0.")
                passing_stats[col] = 0
            else:
                # Fill NaNs if the column already exists
                passing_stats[col] = passing_stats[col].fillna(0)
        
        # Update fantasy points to include rushing
        passing_stats['fantasy_points'] = (
            (1/25) * passing_stats['passing_yards'] +
            4 * passing_stats['passing_tds'] +
            -2 * passing_stats['passing_interceptions'] +
            -2 * passing_stats['passing_fumbles_lost'] +
            2 * passing_stats['passing_2pt_conversions'] +
            # Add rushing points
            (1/10) * passing_stats['rushing_yards'] +
            6 * passing_stats['rushing_tds'] +
            -2 * passing_stats['rushing_fumbles_lost'] +
            2 * passing_stats['rushing_2pt_conversions']
        )
    else:
        # Initialize rushing columns with zeros before calculating fantasy points
        rushing_fantasy_cols = [
            'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions'
        ]
        for col in rushing_fantasy_cols:
            if col not in passing_stats.columns:
                passing_stats[col] = 0
            else:
                passing_stats[col] = passing_stats[col].fillna(0)
                
        # Calculate fantasy points with zero rushing stats
        passing_stats['fantasy_points'] = (
            (1/25) * passing_stats['passing_yards'] +
            4 * passing_stats['passing_tds'] +
            -2 * passing_stats['passing_interceptions'] +
            -2 * passing_stats['passing_fumbles_lost'] +
            2 * passing_stats['passing_2pt_conversions'] +
            # Add rushing points (which will be zeros)
            (1/10) * passing_stats['rushing_yards'] +
            6 * passing_stats['rushing_tds'] +
            -2 * passing_stats['rushing_fumbles_lost'] +
            2 * passing_stats['rushing_2pt_conversions']
        )
    
    # Set nulls for columns with no data
    passing_stats['passing_2pt_conversions'] = passing_stats['passing_2pt_conversions'].fillna(0)
    
    # Ensure qb_dropback column exists
    if 'qb_dropback' not in passing_stats.columns:
        passing_stats['qb_dropback'] = passing_stats['attempts'] + passing_stats['sacks_suffered']
    else:
        passing_stats['qb_dropback'] = passing_stats['qb_dropback'].fillna(0)
    
    # Add default rushing stats columns
    rushing_columns = [
        'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
        'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
        'carries', 'yards_per_carry', 'epa_per_carry', 'rushing_2pt_conversions'
    ]
    
    # Initialize all rushing columns with zeros to ensure they always exist
    for col in rushing_columns:
        if col not in passing_stats.columns:
            passing_stats[col] = 0
    
    # Add rushing stats if we have player IDs
    if len(passing_stats) > 0:
        # Make a copy of the original PBP data for rushing stats
        rushing_pbp = pbp.copy()
        
        # Apply the same filters as for passing
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
        
        # Get QB IDs from passing stats
        qb_ids = passing_stats['player_id'].tolist()
        
        # Filter for QB rushing plays
        if player_id is not None:
            # If we're looking for a specific QB, filter for them as a rusher
            rusher_filter = ~rushing_pbp['rusher_player_id'].isna() & (rushing_pbp['rusher_player_id'] == player_id) & (rushing_pbp['play_type'] == 'run')
            print(f"DEBUG: Filtering QB rushing plays for QB player_id={player_id}, found {rusher_filter.sum()} plays")
        else:
            # Get all QBs from the passing stats and find their rushing plays
            rusher_filter = ~rushing_pbp['rusher_player_id'].isna() & rushing_pbp['rusher_player_id'].isin(qb_ids) & (rushing_pbp['play_type'] == 'run')
            print(f"DEBUG: Filtering QB rushing plays for {len(qb_ids)} QBs, found {rusher_filter.sum()} plays")
        
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
            
            # Calculate yards per carry
            rushing_stats['yards_per_carry'] = np.where(
                rushing_stats['carries'] > 0,
                rushing_stats['rushing_yards'] / rushing_stats['carries'],
                np.nan
            )
            
            # Calculate EPA per carry
            rushing_stats['epa_per_carry'] = np.where(
                rushing_stats['carries'] > 0,
                rushing_stats['rushing_epa'] / rushing_stats['carries'],
                np.nan
            )
            
            # Calculate 2-point conversions for rushing
            rush_2pt_filter = (rushing_pbp['play_type'] == 'run') & (rushing_pbp['two_point_conv_result'] == 'success')
            if player_id is not None:
                rush_2pt_filter &= (rushing_pbp['rusher_player_id'] == player_id)
            else:
                rush_2pt_filter &= rushing_pbp['rusher_player_id'].isin(qb_ids)
            
            # Set grouping variables for 2-point conversions
            if aggregation_type == "season":
                rush_2pt_groups = ['season', 'rusher_player_id']
            elif aggregation_type == "week":
                rush_2pt_groups = ['season', 'week', 'rusher_player_id']
            else:  # career - aggregate all seasons together
                rush_2pt_groups = ['rusher_player_id']
            
            if rush_2pt_filter.sum() > 0:
                rush_2pt_stats = (
                    rushing_pbp[rush_2pt_filter]
                    .groupby(rush_2pt_groups)
                    .size()
                    .reset_index()
                    .rename(columns={
                        0: 'rushing_2pt_conversions',
                        'rusher_player_id': 'player_id'  # Rename to match rushing_stats
                    })
                )
                
                # Join with the rushing stats
                if rush_2pt_stats.shape[0] > 0:
                    # Determine merge columns based on aggregation_type
                    if aggregation_type == "season":
                        merge_cols = ['season', 'player_id']
                    elif aggregation_type == "week":
                        merge_cols = ['season', 'week', 'player_id']
                    else:  # career - just merge on player_id to aggregate across seasons
                        merge_cols = ['player_id']
                        
                    rushing_stats = rushing_stats.merge(
                        rush_2pt_stats,
                        on=merge_cols,
                        how='left'
                    )
                else:
                    rushing_stats['rushing_2pt_conversions'] = 0
            else:
                rushing_stats['rushing_2pt_conversions'] = 0
            
            # Now merge rushing stats with passing stats
            if aggregation_type == "season":
                merge_cols = ['season', 'player_id', 'team']
            elif aggregation_type == "week":
                merge_cols = ['season', 'week', 'player_id', 'team']
            else:  # career
                merge_cols = ['player_id', 'team']
            
            # Keep only merge columns that exist in both DataFrames
            merge_cols = [col for col in merge_cols if col in passing_stats.columns and col in rushing_stats.columns]
            
            # Merge rushing stats with passing stats
            if len(merge_cols) > 0:
                passing_stats = passing_stats.merge(
                    rushing_stats,
                    on=merge_cols,
                    how='left'
                )
                
                # Fill NaN values in rushing columns
                rushing_cols = [
                    'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
                    'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
                    'carries', 'yards_per_carry', 'epa_per_carry', 'rushing_2pt_conversions'
                ]
                for col in rushing_cols:
                    if col in passing_stats.columns:
                        passing_stats[col] = passing_stats[col].fillna(0)
                
                # Ensure all required rushing columns exist before fantasy points calculation
                rushing_fantasy_cols = [
                    'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions'
                ]
                for col in rushing_fantasy_cols:
                    if col not in passing_stats.columns:
                        # Create the column if it doesn't exist
                        passing_stats[col] = 0
                    else:
                        # Fill NaNs if it exists
                        passing_stats[col] = passing_stats[col].fillna(0)
                
                # Update fantasy points to include rushing stats
                passing_stats['fantasy_points'] = (
                    (1/25) * passing_stats['passing_yards'] +
                    4 * passing_stats['passing_tds'] +
                    -2 * passing_stats['passing_interceptions'] +
                    -2 * passing_stats['passing_fumbles_lost'] +
                    2 * passing_stats['passing_2pt_conversions'] +
                    # Add rushing points
                    (1/10) * passing_stats['rushing_yards'] +
                    6 * passing_stats['rushing_tds'] +
                    -2 * passing_stats['rushing_fumbles_lost'] +
                    2 * passing_stats['rushing_2pt_conversions']
                )
            else:
                # Couldn't merge, so add empty rushing columns
                rushing_cols = [
                    'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
                    'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
                    'carries', 'yards_per_carry', 'epa_per_carry', 'rushing_2pt_conversions'
                ]
                for col in rushing_cols:
                    passing_stats[col] = 0
                
                # Update fantasy points to include rushing stats (all zeros)
                passing_stats['fantasy_points'] = (
                    (1/25) * passing_stats['passing_yards'] +
                    4 * passing_stats['passing_tds'] +
                    -2 * passing_stats['passing_interceptions'] +
                    -2 * passing_stats['passing_fumbles_lost'] +
                    2 * passing_stats['passing_2pt_conversions'] +
                    # Add rushing points (zeros)
                    0  # Since all rushing stats are zero
                )
        else:
            # No rushing plays found, add empty columns
            rushing_cols = [
                'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
                'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
                'carries', 'yards_per_carry', 'epa_per_carry', 'rushing_2pt_conversions'
            ]
            for col in rushing_cols:
                passing_stats[col] = 0
                
            # Calculate fantasy points with zero rushing stats
            passing_stats['fantasy_points'] = (
                (1/25) * passing_stats['passing_yards'] +
                4 * passing_stats['passing_tds'] +
                -2 * passing_stats['passing_interceptions'] +
                -2 * passing_stats['passing_fumbles_lost'] +
                2 * passing_stats['passing_2pt_conversions']
                # No rushing points since all rushing stats are zero
            )
                
        print(f"DEBUG: After adding QB rushing stats, columns: {passing_stats.columns.tolist()}")
    
    # Add player name if available in the play-by-play data
    if 'passer_player_name' in filtered_pbp.columns:
        player_names = (
            filtered_pbp[['passer_player_id', 'passer_player_name']]
            .drop_duplicates()
            .rename(columns={
                'passer_player_id': 'player_id',
                'passer_player_name': 'player_name'
            })
        )
        passing_stats = passing_stats.merge(player_names, on='player_id', how='left')
    
    # For consistency with the R version, include season_type for non-career aggregations
    if 'season_type' in filtered_pbp.columns and aggregation_type != "career":
        if aggregation_type == "season":
            # For season summary, we use a concatenated string of unique season types
            season_types = (
                filtered_pbp[['season', 'season_type']]
                .drop_duplicates()
                .groupby('season')
                .agg({'season_type': lambda x: '+'.join(sorted(set(x)))})
                .reset_index()
            )
            passing_stats = passing_stats.merge(season_types, on='season', how='left')
        else:  # week
            # For weekly summaries, we just take the season type for that week
            season_types = (
                filtered_pbp[['season', 'week', 'season_type']]
                .drop_duplicates()
                .groupby(['season', 'week'])
                .agg({'season_type': 'first'})
                .reset_index()
            )
            passing_stats = passing_stats.merge(season_types, on=['season', 'week'], how='left')
    elif aggregation_type == "career" and 'season_type' in filtered_pbp.columns:
        # For career stats, add the season_type as supplied in the function parameter
        passing_stats['season_type'] = season_type
    
    # Count QB dropbacks from the original data where qb_dropback is True
    # This assumes qb_dropback is a boolean column in the original data
    if 'qb_dropback' in filtered_pbp.columns:
        dropback_counts = (
            filtered_pbp[filtered_pbp['qb_dropback'] == True]
            .groupby(group_cols)
            .size()
            .reset_index()
            .rename(columns={0: 'qb_dropback'})
        )
        
        # Rename columns to match the naming in passing_stats
        rename_dict = {}
        if 'passer_player_id' in dropback_counts.columns:
            rename_dict['passer_player_id'] = 'player_id'
        if 'posteam' in dropback_counts.columns:
            rename_dict['posteam'] = 'team'
        dropback_counts = dropback_counts.rename(columns=rename_dict)
        
        # Determine merge columns
        merge_cols = []
        for col in group_cols:
            if col == 'passer_player_id':
                merge_cols.append('player_id')
            elif col == 'posteam':
                merge_cols.append('team')  # posteam was renamed to team in passing_stats
            else:
                merge_cols.append(col)
        
        # Merge the dropback counts with our stats
        passing_stats = passing_stats.merge(
            dropback_counts,
            on=merge_cols,
            how='left'
        )

    
    # Ensure qb_dropback exists and is filled with appropriate values
    if 'qb_dropback' not in passing_stats.columns:
        # If missing completely, calculate it from attempts and sacks
        if 'attempts' in passing_stats.columns and 'sacks_suffered' in passing_stats.columns:
            passing_stats['qb_dropback'] = passing_stats['attempts'] + passing_stats['sacks_suffered']
        else:
            # If we don't have the component columns, set to 0
            passing_stats['qb_dropback'] = 0
    else:
        # If it exists, fill any NA values with 0
        passing_stats['qb_dropback'] = passing_stats['qb_dropback'].fillna(0)
    
    # Ensure qb_epa exists and has appropriate values
    if 'qb_epa' not in passing_stats.columns:
        # If missing, set to 0 or calculate from play-by-play data if available
        passing_stats['qb_epa'] = 0
    else:
        # Fill NAs with 0
        passing_stats['qb_epa'] = passing_stats['qb_epa'].fillna(0)

    # Calculate EPA per dropback
    passing_stats['epa_per_dropback'] = np.where(
        passing_stats['qb_dropback'] > 0,
        passing_stats['qb_epa'] / passing_stats['qb_dropback'],
        np.nan
    )
    
    # Calculate total touchdowns (passing + rushing)
    if 'passing_tds' in passing_stats.columns and 'rushing_tds' in passing_stats.columns:
        # Fill NAs with 0 before calculation
        passing_stats['passing_tds'] = passing_stats['passing_tds'].fillna(0)
        passing_stats['rushing_tds'] = passing_stats['rushing_tds'].fillna(0)
        # Calculate total touchdowns
        passing_stats['total_tds'] = passing_stats['passing_tds'] + passing_stats['rushing_tds']
    elif 'passing_tds' in passing_stats.columns:
        # Only passing TDs are available
        passing_stats['total_tds'] = passing_stats['passing_tds']
    elif 'rushing_tds' in passing_stats.columns:
        # Only rushing TDs are available
        passing_stats['total_tds'] = passing_stats['rushing_tds']
    else:
        # No TD data available
        passing_stats['total_tds'] = 0
    
    # Determine final columns based on aggregation_type
    base_cols = [
        'player_id', 'team', 
        'games_played', 'attempts', 'completions', 'completion_pct',
        'passing_yards', 'passing_tds', 'passing_interceptions',
        'rushing_yards', 'rushing_tds', 'carries',
        'total_tds',  # Combined passing + rushing TDs
        'qb_epa', 'passing_cpoe', 'epa_per_dropback',
        'passing_air_yards', 'passing_yards_after_catch', 'pacr',
        'passing_first_downs', 'passing_2pt_conversions',
        'sacks_suffered', 'passing_fumbles', 'passing_fumbles_lost', 
        'qb_dropback', 'fantasy_points'
    ]
    
    # Add player_name to base_cols only if it exists in the DataFrame
    if 'player_name' in passing_stats.columns:
        base_cols.insert(1, 'player_name')
    
    if aggregation_type == "season":
        result_cols = ['season'] + base_cols
    elif aggregation_type == "week":
        result_cols = ['season', 'week'] + base_cols
    else:  # career
        result_cols = base_cols
        
    # Add season_type if available
    if 'season_type' in passing_stats.columns:
        result_cols.append('season_type')
        
    # Add player name if requested
    result = passing_stats[result_cols]
    
    # Add player name and position from players dataset if requested
    if add_player_name and 'player_id' in result.columns and not result.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            player_info = players_df[players_df['gsis_id'].isin(result['player_id'])].set_index('gsis_id')
            
            # Add player name if not already present
            if 'player_name' not in result.columns:
                result['player_name'] = result['player_id'].map(
                    lambda x: player_info.loc[x, 'display_name'] if x in player_info.index else f"Unknown ({x})"
                )
            
            # Always use the correct position based on the function type, don't use database position 
            # Database position may be incorrect (that's causing our issue)
            if 'calculate_qb_stats' in sys._getframe().f_code.co_name:
                result['position'] = "QB"
            elif 'calculate_rb_stats' in sys._getframe().f_code.co_name:
                result['position'] = "RB"
            elif 'calculate_wr_stats' in sys._getframe().f_code.co_name:
                result['position'] = "WR"
            # Add additional position functions as needed

        except Exception as e:
            print(f"Could not add player name/position: {e}")
    
    # Add player names for all players if requested
    elif add_player_name and 'player_name' not in result.columns and not result.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            # Create a mapping of player IDs to display names
            player_name_map = dict(zip(players_df['gsis_id'], players_df['display_name']))
            # Create a mapping for positions
            player_pos_map = dict(zip(players_df['gsis_id'], players_df['position']))
            
            # Add the display name as player_name column
            result['player_name'] = result['player_id'].map(
                lambda x: player_name_map.get(x, f"Unknown ({x})")
            )
            # For QB stats always set position to QB, don't use player_pos_map
            # This ensures correct position regardless of database values
            result['position'] = "QB"
        except Exception as e:
            print(f"Could not add player names/positions: {e}")
    
    # Ensure 'position' column is included in final selection if added
    if 'position' in result.columns and 'position' not in result_cols:
        # Try inserting after 'team' or 'player_name' if they exist
        try:
            insert_idx = result_cols.index('team') + 1
        except ValueError:
            try:
                insert_idx = result_cols.index('player_name') + 1
            except ValueError:
                insert_idx = 2 # Fallback index
        result_cols.insert(insert_idx, 'position')
    
    # Re-select columns to ensure order and inclusion of potentially added 'position'
    final_result = result[[col for col in result_cols if col in result.columns]]
    
    # Explicitly add/update position column using player_id just before returning
    if 'player_id' in final_result.columns and not final_result.empty:
        # Assuming position is consistent for a player within the group
        first_player_id = final_result['player_id'].iloc[0] 
        if first_player_id:
            # ALWAYS override with the correct position based on calculation function
            # This is the QB stats function, so use QB regardless of what's in the database
            final_result['position'] = "QB"
            
            # Only add position if it somehow isn't in columns list
            if 'position' not in final_result.columns.tolist():
                 cols = final_result.columns.tolist()
                 try:
                    insert_idx = cols.index('team') + 1
                 except ValueError:
                    try:
                        insert_idx = cols.index('player_name') + 1
                    except ValueError:
                        insert_idx = 2 # Fallback
                 cols.insert(insert_idx, 'position')
                 final_result = final_result[cols] # Recreate DataFrame with correct column order


    # Add the detected position back if it wasn't retrieved from players.parquet
    # This covers cases where add_player_name was False or player wasn't in players.parquet
    # if 'position' not in final_result.columns and position: # REMOVED - Handled above now
    #     final_result['position'] = position
    #     # Reorder to place position near name/team if possible
    #     cols = final_result.columns.tolist()
    #     if 'position' in cols:
    #          cols.remove('position')
    #          try:
    #              insert_idx = cols.index('team') + 1
    #          except ValueError:
    #              try:
    #                  insert_idx = cols.index('player_name') + 1
    #              except ValueError:
    #                  insert_idx = 2 # Fallback
    #          cols.insert(insert_idx, 'position')
    #          final_result = final_result[cols]

    # Ensure the position column exists and is populated with the correct position
    # based on which stats calculation function was called
    if 'position' not in final_result.columns:
        if 'calculate_qb_stats' in sys._getframe().f_code.co_name:
            final_result['position'] = "QB"
        elif 'calculate_rb_stats' in sys._getframe().f_code.co_name:
            final_result['position'] = "RB"  
        elif 'calculate_wr_stats' in sys._getframe().f_code.co_name:
            final_result['position'] = "WR"
        # Add other function mappings as needed


    return final_result

def calculate_rb_stats(
    pbp: pd.DataFrame,
    player_id: Optional[str] = None,
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
    Compute running back stats based on NFL play-by-play data.
    
    Parameters:
    -----------
    pbp : pd.DataFrame
        Play-by-play data
    player_id : str, optional
        Specific player ID to filter results for
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
    opponent_on_streak : bool, default False
        If True, only include plays against opponents on win streaks
    opponent_streak_length : int, default 3
        Minimum consecutive wins for opponent to be considered on a streak
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing RB stats according to specified parameters
    """
    # Input validation
    if aggregation_type not in ["season", "week", "career"]:
        raise ValueError("aggregation_type must be one of 'season', 'week', or 'career'")
    
    if season_type not in ["REG", "POST", "REG+POST"]:
        raise ValueError("season_type must be one of 'REG', 'POST', or 'REG+POST'")
    
    # Make a copy to avoid modifying the original
    filtered_pbp = pbp.copy()
    
    # Handle seasons parameter
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        filtered_pbp = filtered_pbp[filtered_pbp['season'].isin(seasons)]
    
    # Filter by season_type if needed
    if season_type in ["REG", "POST"]:
        filtered_pbp = filtered_pbp[filtered_pbp['season_type'] == season_type]
        if len(filtered_pbp) == 0:
            print(f"Filtering to season_type = {season_type} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by player_id if specified (for rushing stats)
    if player_id is not None:
        filtered_pbp = filtered_pbp[filtered_pbp['rusher_player_id'] == player_id]
        if len(filtered_pbp) == 0:
            print(f"Filtering to player_id = {player_id} resulted in 0 rows for rushing stats. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by week if specified and relevant
    if week is not None and aggregation_type == "week":
        filtered_pbp = filtered_pbp[filtered_pbp['week'] == week]
        if len(filtered_pbp) == 0:
            print(f"Filtering to week = {week} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter for redzone plays if specified
    if redzone_only:
        filtered_pbp = filtered_pbp[filtered_pbp['yardline_100'] <= 20]
        if len(filtered_pbp) == 0:
            print("Filtering to redzone plays resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by downs if specified
    if downs is not None:
        filtered_pbp = filtered_pbp[filtered_pbp['down'].isin(downs)]
        if len(filtered_pbp) == 0:
            downs_str = ", ".join(str(d) for d in downs)
            print(f"Filtering to downs ({downs_str}) resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by opponent team if specified
    if opponent_team is not None:
        # The opponent of the posteam (possession team) is defteam
        filtered_pbp = filtered_pbp[filtered_pbp['defteam'] == opponent_team]
        if len(filtered_pbp) == 0:
            print(f"Filtering to opponent_team = {opponent_team} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by score differential range if specified
    if score_differential_range is not None:
        min_diff, max_diff = score_differential_range
        filtered_pbp = filtered_pbp[(filtered_pbp['score_differential'] >= min_diff) & 
                                    (filtered_pbp['score_differential'] <= max_diff)]
        if len(filtered_pbp) == 0:
            print(f"Filtering to score_differential between {min_diff} and {max_diff} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Set grouping variables based on aggregation_type
    if aggregation_type == "season":
        group_cols = ['season', 'rusher_player_id', 'posteam']
    elif aggregation_type == "week":
        group_cols = ['season', 'week', 'rusher_player_id', 'posteam']
    else:  # career - aggregate across all seasons for the player
        group_cols = ['rusher_player_id', 'posteam']
    
    # RB-specific rushing stats
    rushing_filter = filtered_pbp['play_type'].isin(['run', 'qb_kneel'])
    
    # Extract RB rushing stats
    rushing_stats = (
        filtered_pbp[rushing_filter]
        .groupby(group_cols)
        .agg({
            'yards_gained': 'sum',
            'touchdown': 'sum',
            'rush_touchdown': 'sum',
            'first_down_rush': 'sum',
            'fumble': 'sum',
            'fumble_lost': 'sum',
            'epa': lambda x: x.sum(skipna=True),
            'game_id': 'nunique'  # For game count
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
            'epa': 'rushing_epa',
            'game_id': 'games_played'
        })
    )
    
    # Calculate carries by counting the number of rushing plays
    carries_count = (
        filtered_pbp[rushing_filter]
        .groupby(group_cols)
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
    for col in group_cols:
        if col == 'rusher_player_id':
            merge_cols.append('player_id')
        elif col == 'posteam':
            merge_cols.append('team')
        else:
            merge_cols.append(col)
    
    # Merge the carries count with rushing stats
    rushing_stats = rushing_stats.merge(
        carries_count,
        on=merge_cols,
        how='left'
    )
    
    # Calculate 2-point conversions
    two_point_filter = (filtered_pbp['play_type'] == 'run') & (filtered_pbp['two_point_conv_result'] == 'success')
    
    # Set grouping variables for 2-point conversions
    if aggregation_type == "season":
        two_point_groups = ['season', 'rusher_player_id']
    elif aggregation_type == "week":
        two_point_groups = ['season', 'week', 'rusher_player_id']
    else:  # career - aggregate all seasons together
        two_point_groups = ['rusher_player_id']
    
    two_point_stats = (
        filtered_pbp[two_point_filter]
        .groupby(two_point_groups)
        .size()
        .reset_index()
        .rename(columns={
            0: 'rushing_2pt_conversions',
            'rusher_player_id': 'player_id'  # Rename to match rushing_stats
        })
    )
    
    # Join with the main stats
    if two_point_stats.shape[0] > 0:
        # Determine merge columns based on aggregation_type
        if aggregation_type == "season":
            merge_cols = ['season', 'player_id']
        elif aggregation_type == "week":
            merge_cols = ['season', 'week', 'player_id']
        else:  # career - just merge on player_id to aggregate across seasons
            merge_cols = ['player_id']
            
        rushing_stats = rushing_stats.merge(
            two_point_stats,
            on=merge_cols,
            how='left'
        )
    else:
        rushing_stats['rushing_2pt_conversions'] = 0
    
    # Calculate yards per carry
    rushing_stats['yards_per_carry'] = np.where(
        rushing_stats['carries'] > 0,
        rushing_stats['rushing_yards'] / rushing_stats['carries'],
        np.nan
    )
    
    # Calculate EPA per carry
    rushing_stats['epa_per_carry'] = np.where(
        rushing_stats['carries'] > 0,
        rushing_stats['rushing_epa'] / rushing_stats['carries'],
        np.nan
    )
    
    # For receiving stats, we need to get a fresh copy of the data that isn't already filtered for rushing plays
    # We'll start from the original dataset and apply all filters except the rusher_player_id filter
    receiving_pbp = pbp.copy()
    
    # Apply the same filters we applied to the rushing data
    # Handle seasons parameter
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        receiving_pbp = receiving_pbp[receiving_pbp['season'].isin(seasons)]
    
    # Filter by season_type if needed
    if season_type in ["REG", "POST"]:
        receiving_pbp = receiving_pbp[receiving_pbp['season_type'] == season_type]
    
    # Filter by week if specified and relevant
    if week is not None and aggregation_type == "week":
        receiving_pbp = receiving_pbp[receiving_pbp['week'] == week]
    
    # Filter for redzone plays if specified
    if redzone_only:
        receiving_pbp = receiving_pbp[receiving_pbp['yardline_100'] <= 20]
        
    # Filter by downs if specified
    if downs is not None:
        receiving_pbp = receiving_pbp[receiving_pbp['down'].isin(downs)]
    
    # Filter by opponent team if specified
    if opponent_team is not None:
        # The opponent of the posteam (possession team) is defteam
        receiving_pbp = receiving_pbp[receiving_pbp['defteam'] == opponent_team]
        
    # Filter by score differential range if specified
    if score_differential_range is not None:
        min_diff, max_diff = score_differential_range
        receiving_pbp = receiving_pbp[(receiving_pbp['score_differential'] >= min_diff) & 
                                     (receiving_pbp['score_differential'] <= max_diff)]
    
    # Need to create new group cols for the receiver stats
    if aggregation_type == "season":
        receiver_group_cols = ['season', 'receiver_player_id', 'posteam']
    elif aggregation_type == "week":
        receiver_group_cols = ['season', 'week', 'receiver_player_id', 'posteam']
    else:  # career
        receiver_group_cols = ['receiver_player_id', 'posteam']
    
    # Filter for receiving plays where the RB is the receiver
    if player_id is not None:
        # If we're looking for a specific player, filter for them as a receiver
        receiver_filter = ~receiving_pbp['receiver_player_id'].isna() & (receiving_pbp['receiver_player_id'] == player_id)
    else:
        # Otherwise, just get all receiving plays
        receiver_filter = ~receiving_pbp['receiver_player_id'].isna()
    
    # Get receiving stats
    receiving_stats = None
    
    if receiver_filter.sum() > 0:
        receiving_stats = (
            receiving_pbp[receiver_filter]
            .groupby(receiver_group_cols)
            .agg({
                'complete_pass': 'sum',  # Receptions
                'yards_gained': 'sum',
                'touchdown': 'sum',
                'pass_touchdown': 'sum',
                'first_down_pass': 'sum',
                'fumble': 'sum',
                'fumble_lost': 'sum',
                'air_yards': 'sum',
                'yards_after_catch': 'sum',
                'epa': lambda x: x.sum(skipna=True)
            })
            .reset_index()
            .rename(columns={
                'receiver_player_id': 'player_id',
                'posteam': 'team',
                'complete_pass': 'receptions',
                'yards_gained': 'receiving_yards',
                'pass_touchdown': 'receiving_tds',
                'first_down_pass': 'receiving_first_downs',
                'fumble': 'receiving_fumbles',
                'fumble_lost': 'receiving_fumbles_lost',
                'air_yards': 'receiving_air_yards',
                'yards_after_catch': 'receiving_yards_after_catch',
                'epa': 'receiving_epa'
            })
        )
        
        # Calculate targets from pbp data
        targets_filter = ~receiving_pbp['receiver_player_id'].isna()
        if player_id is not None:
            targets_filter &= (receiving_pbp['receiver_player_id'] == player_id)
        
        # Count targets (any time the player was targeted, even if incomplete)
        targets_count = (
            receiving_pbp[targets_filter]
            .groupby(receiver_group_cols)
            .size()
            .reset_index()
            .rename(columns={0: 'targets'})
        )
        
        # Rename columns to match receiving_stats
        rename_dict = {}
        if 'receiver_player_id' in targets_count.columns:
            rename_dict['receiver_player_id'] = 'player_id'
        if 'posteam' in targets_count.columns:
            rename_dict['posteam'] = 'team'
        targets_count = targets_count.rename(columns=rename_dict)
        
        # Determine merge columns for targets
        target_merge_cols = []
        for col in receiver_group_cols:
            if col == 'receiver_player_id':
                target_merge_cols.append('player_id')
            elif col == 'posteam':
                target_merge_cols.append('team')
            else:
                target_merge_cols.append(col)
        
        # Merge the target counts with receiving stats
        receiving_stats = receiving_stats.merge(
            targets_count,
            on=target_merge_cols,
            how='left'
        )
        
        # Calculate receiving 2-point conversions
        rec_2pt_filter = (receiving_pbp['play_type'] == 'pass') & (receiving_pbp['two_point_conv_result'] == 'success')
        if player_id is not None:
            rec_2pt_filter &= (receiving_pbp['receiver_player_id'] == player_id)
        
        # Set grouping variables for 2-point conversions (receiving)
        if aggregation_type == "season":
            rec_2pt_groups = ['season', 'receiver_player_id']
        elif aggregation_type == "week":
            rec_2pt_groups = ['season', 'week', 'receiver_player_id']
        else:  # career - aggregate all seasons together
            rec_2pt_groups = ['receiver_player_id']
        
        # Check if we have any receiving 2-point conversions
        if rec_2pt_filter.sum() > 0:
            rec_2pt_stats = (
                receiving_pbp[rec_2pt_filter]
                .groupby(rec_2pt_groups)
                .size()
                .reset_index()
                .rename(columns={
                    0: 'receiving_2pt_conversions',
                    'receiver_player_id': 'player_id'
                })
            )
            
            # Join with receiving stats if we have any 2-point conversions
            if rec_2pt_stats.shape[0] > 0:
                # Determine merge columns based on aggregation_type
                if aggregation_type == "season":
                    rec_2pt_merge_cols = ['season', 'player_id']
                elif aggregation_type == "week":
                    rec_2pt_merge_cols = ['season', 'week', 'player_id']
                else:  # career
                    rec_2pt_merge_cols = ['player_id']
                
                receiving_stats = receiving_stats.merge(
                    rec_2pt_stats,
                    on=rec_2pt_merge_cols,
                    how='left'
                )
            else:
                receiving_stats['receiving_2pt_conversions'] = 0
        else:
            receiving_stats['receiving_2pt_conversions'] = 0
        
        # Calculate RACR (Receiving Air Conversion Ratio)
        receiving_stats['racr'] = np.where(
            receiving_stats['receiving_air_yards'] != 0,
            receiving_stats['receiving_yards'] / receiving_stats['receiving_air_yards'],
            np.nan
        )
        
        # Calculate yards per reception
        receiving_stats['yards_per_reception'] = np.where(
            receiving_stats['receptions'] > 0,
            receiving_stats['receiving_yards'] / receiving_stats['receptions'],
            np.nan
        )
        
        # Calculate catch rate
        receiving_stats['catch_rate'] = np.where(
            receiving_stats['targets'] > 0,
            receiving_stats['receptions'] / receiving_stats['targets'],
            np.nan
        )
    
    # Now merge rushing and receiving stats (if we have receiving stats)
    if receiving_stats is not None and receiving_stats.shape[0] > 0:
        # Determine merge columns based on aggregation_type
        if aggregation_type == "season":
            rb_merge_cols = ['season', 'player_id', 'team']
        elif aggregation_type == "week":
            rb_merge_cols = ['season', 'week', 'player_id', 'team']
        else:  # career
            rb_merge_cols = ['player_id', 'team']
        
        # Outer join to include players who only rush or only receive
        rb_stats = rushing_stats.merge(
            receiving_stats,
            on=rb_merge_cols,
            how='outer'
        )
    else:
        rb_stats = rushing_stats
        # Add empty receiving columns
        receiving_cols = [
            'receptions', 'targets', 'receiving_yards', 'receiving_tds',
            'receiving_first_downs', 'receiving_fumbles', 'receiving_fumbles_lost',
            'receiving_air_yards', 'receiving_yards_after_catch',
            'receiving_epa', 'receiving_2pt_conversions', 'racr',
            'yards_per_reception', 'catch_rate'
        ]
        for col in receiving_cols:
            rb_stats[col] = np.nan
    
    # Fill NaN values with appropriate defaults
    numeric_cols = rb_stats.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        rb_stats[col] = rb_stats[col].fillna(0)
    
    # Calculate fantasy points for RBs
    rb_stats['fantasy_points'] = (
        (1/10) * rb_stats['rushing_yards'] +
        6 * rb_stats['rushing_tds'] +
        2 * rb_stats['rushing_2pt_conversions'] +
        (1/10) * rb_stats['receiving_yards'] +
        6 * rb_stats['receiving_tds'] +
        2 * rb_stats['receiving_2pt_conversions'] +
        -2 * (rb_stats['rushing_fumbles_lost'] + rb_stats['receiving_fumbles_lost'])
    )
    
    # PPR fantasy points (add 1 point per reception)
    rb_stats['fantasy_points_ppr'] = rb_stats['fantasy_points'] + rb_stats['receptions']
    
    # Add player name if available in the play-by-play data
    if 'rusher_player_name' in filtered_pbp.columns:
        player_names = (
            filtered_pbp[['rusher_player_id', 'rusher_player_name']]
            .drop_duplicates()
            .rename(columns={
                'rusher_player_id': 'player_id',
                'rusher_player_name': 'player_name'
            })
        )
        rb_stats = rb_stats.merge(player_names, on='player_id', how='left')
    
    # For consistency with the R version, include season_type for non-career aggregations
    if 'season_type' in filtered_pbp.columns and aggregation_type != "career":
        if aggregation_type == "season":
            # For season summary, we use a concatenated string of unique season types
            season_types = (
                filtered_pbp[['season', 'season_type']]
                .drop_duplicates()
                .groupby('season')
                .agg({'season_type': lambda x: '+'.join(sorted(set(x)))})
                .reset_index()
            )
            rb_stats = rb_stats.merge(season_types, on='season', how='left')
        else:  # week
            # For weekly summaries, we just take the season type for that week
            season_types = (
                filtered_pbp[['season', 'week', 'season_type']]
                .drop_duplicates()
                .groupby(['season', 'week'])
                .agg({'season_type': 'first'})
                .reset_index()
            )
            rb_stats = rb_stats.merge(season_types, on=['season', 'week'], how='left')
    elif aggregation_type == "career" and 'season_type' in filtered_pbp.columns:
        # For career stats, add the season_type as supplied in the function parameter
        rb_stats['season_type'] = season_type
    
    # Determine final columns based on aggregation_type
    base_cols = [
        'player_id', 'team', 'games_played',
        # Rushing stats
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_first_downs',
        'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_2pt_conversions',
        'rushing_epa', 'yards_per_carry', 'epa_per_carry',
        # Receiving stats
        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'receiving_first_downs', 'receiving_fumbles', 'receiving_fumbles_lost',
        'receiving_2pt_conversions', 'receiving_air_yards', 'receiving_yards_after_catch',
        'receiving_epa', 'racr', 'yards_per_reception', 'catch_rate',
        # Fantasy
        'fantasy_points', 'fantasy_points_ppr'
    ]
    
    # Add player_name to base_cols only if it exists in the DataFrame
    if 'player_name' in rb_stats.columns:
        base_cols.insert(1, 'player_name')
    
    if aggregation_type == "season":
        result_cols = ['season'] + base_cols
    elif aggregation_type == "week":
        result_cols = ['season', 'week'] + base_cols
    else:  # career
        result_cols = base_cols
        
    # Add season_type if available
    if 'season_type' in rb_stats.columns:
        result_cols.append('season_type')
    
    # Ensure all columns exist (some might be missing if no data was found)
    for col in result_cols:
        if col not in rb_stats.columns:
            rb_stats[col] = 0 if col != 'player_name' else None
            
    # Add player name if requested
    result = rb_stats[result_cols]
    
    # Add player name and position from players dataset if requested
    if add_player_name and 'player_id' in result.columns and not result.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            player_info = players_df[players_df['gsis_id'].isin(result['player_id'])].set_index('gsis_id')
            
            # Add player name if not already present
            if 'player_name' not in result.columns:
                result['player_name'] = result['player_id'].map(
                    lambda x: player_info.loc[x, 'display_name'] if x in player_info.index else f"Unknown ({x})"
                )
            
            # Always use the correct position based on the function type, don't use database position 
            # Database position may be incorrect (that's causing our issue)
            if 'calculate_qb_stats' in sys._getframe().f_code.co_name:
                result['position'] = "QB"
            elif 'calculate_rb_stats' in sys._getframe().f_code.co_name:
                result['position'] = "RB"
            elif 'calculate_wr_stats' in sys._getframe().f_code.co_name:
                result['position'] = "WR"
            # Add additional position functions as needed

        except Exception as e:
            print(f"Could not add player name/position: {e}")
    
    # Add player names for all players if requested
    elif add_player_name and 'player_name' not in result.columns and not result.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            # Create a mapping of player IDs to display names
            player_name_map = dict(zip(players_df['gsis_id'], players_df['display_name']))
            # Create a mapping for positions
            player_pos_map = dict(zip(players_df['gsis_id'], players_df['position']))
            
            # Add the display name as player_name column
            result['player_name'] = result['player_id'].map(
                lambda x: player_name_map.get(x, f"Unknown ({x})")
            )
            # For QB stats always set position to QB, don't use player_pos_map
            # This ensures correct position regardless of database values
            result['position'] = "QB"
        except Exception as e:
            print(f"Could not add player names/positions: {e}")
    
    # Ensure 'position' column is included in final selection if added
    if 'position' in result.columns and 'position' not in result_cols:
        # Try inserting after 'team' or 'player_name' if they exist
        try:
            insert_idx = result_cols.index('team') + 1
        except ValueError:
            try:
                insert_idx = result_cols.index('player_name') + 1
            except ValueError:
                insert_idx = 2 # Fallback index
        result_cols.insert(insert_idx, 'position')
    
    # Re-select columns to ensure order and inclusion of potentially added 'position'
    final_result = result[[col for col in result_cols if col in result.columns]]
    
    # Explicitly add/update position column using player_id just before returning
    if 'player_id' in final_result.columns and not final_result.empty:
        # Assuming position is consistent for a player within the group
        first_player_id = final_result['player_id'].iloc[0]
        if first_player_id:
            fetched_pos = get_player_position(first_player_id)
            final_result['position'] = fetched_pos
             # Ensure 'position' is in the final columns list again, in case it wasn't added before
            if 'position' not in final_result.columns.tolist():
                 cols = final_result.columns.tolist()
                 try:
                    insert_idx = cols.index('team') + 1
                 except ValueError:
                    try:
                        insert_idx = cols.index('player_name') + 1
                    except ValueError:
                        insert_idx = 2 # Fallback
                 cols.insert(insert_idx, 'position')
                 final_result = final_result[cols] # Recreate DataFrame with correct column order


    # Add the detected position back if it wasn't retrieved from players.parquet
    # This covers cases where add_player_name was False or player wasn't in players.parquet
    # if 'position' not in final_result.columns and position: # REMOVED - Handled above now
    #     final_result['position'] = position
    #     # Reorder to place position near name/team if possible
    #     cols = final_result.columns.tolist()
    #     if 'position' in cols:
    #          cols.remove('position')
    #          try:
    #              insert_idx = cols.index('team') + 1
    #          except ValueError:
    #              try:
    #                  insert_idx = cols.index('player_name') + 1
    #              except ValueError:
    #                  insert_idx = 2 # Fallback
    #          cols.insert(insert_idx, 'position')
    #          final_result = final_result[cols]

    # Ensure the position column exists and is populated
    if 'player_id' in final_result.columns and 'position' not in final_result.columns:
        try:
            # Use the first player_id in the group to determine position
            first_player_id = final_result['player_id'].iloc[0]
            if first_player_id:
                 final_result['position'] = get_player_position(first_player_id)
            else:
                 final_result['position'] = None # Or use the initially detected position if available?
        except Exception as e:
            logger.error(f"Error adding position column in calculate_rb_stats: {e}")
            final_result['position'] = None # Default to None on error
    elif 'position' not in final_result.columns:
         final_result['position'] = None # Ensure column exists even if player_id wasn't present


    return final_result

def calculate_wr_stats(
    pbp: pd.DataFrame,
    player_id: Optional[str] = None,
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
    Compute wide receiver stats based on NFL play-by-play data.
    
    Parameters:
    -----------
    pbp : pd.DataFrame
        Play-by-play data
    player_id : str, optional
        Specific player ID to filter results for
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
    opponent_on_streak : bool, default False
        If True, only include plays against opponents on win streaks
    opponent_streak_length : int, default 3
        Minimum consecutive wins for opponent to be considered on a streak
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing WR stats according to specified parameters
    """
    # Input validation
    if aggregation_type not in ["season", "week", "career"]:
        raise ValueError("aggregation_type must be one of 'season', 'week', or 'career'")
    
    if season_type not in ["REG", "POST", "REG+POST"]:
        raise ValueError("season_type must be one of 'REG', 'POST', or 'REG+POST'")
    
    # Make a copy to avoid modifying the original
    filtered_pbp = pbp.copy()
    
    # Handle seasons parameter
    if seasons is not None:
        if isinstance(seasons, int):
            seasons = [seasons]
        filtered_pbp = filtered_pbp[filtered_pbp['season'].isin(seasons)]
    
    # Filter by season_type if needed
    if season_type in ["REG", "POST"]:
        filtered_pbp = filtered_pbp[filtered_pbp['season_type'] == season_type]
        if len(filtered_pbp) == 0:
            print(f"Filtering to season_type = {season_type} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by player_id if specified
    if player_id is not None:
        filtered_pbp = filtered_pbp[filtered_pbp['receiver_player_id'] == player_id]
        if len(filtered_pbp) == 0:
            print(f"Filtering to player_id = {player_id} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by week if specified and relevant
    if week is not None and aggregation_type == "week":
        filtered_pbp = filtered_pbp[filtered_pbp['week'] == week]
        if len(filtered_pbp) == 0:
            print(f"Filtering to week = {week} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter for redzone plays if specified
    if redzone_only:
        filtered_pbp = filtered_pbp[filtered_pbp['yardline_100'] <= 20]
        if len(filtered_pbp) == 0:
            print("Filtering to redzone plays resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by downs if specified
    if downs is not None:
        filtered_pbp = filtered_pbp[filtered_pbp['down'].isin(downs)]
        if len(filtered_pbp) == 0:
            downs_str = ", ".join(str(d) for d in downs)
            print(f"Filtering to downs ({downs_str}) resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
            
    # Filter by opponent team if specified
    if opponent_team is not None:
        # The opponent of the posteam (possession team) is defteam
        filtered_pbp = filtered_pbp[filtered_pbp['defteam'] == opponent_team]
        if len(filtered_pbp) == 0:
            print(f"Filtering to opponent_team = {opponent_team} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Filter by score differential range if specified
    if score_differential_range is not None:
        min_diff, max_diff = score_differential_range
        filtered_pbp = filtered_pbp[(filtered_pbp['score_differential'] >= min_diff) & 
                                    (filtered_pbp['score_differential'] <= max_diff)]
        if len(filtered_pbp) == 0:
            print(f"Filtering to score_differential between {min_diff} and {max_diff} resulted in 0 rows. Returning empty DataFrame.")
            return pd.DataFrame()
    
    # Set grouping variables based on aggregation_type
    if aggregation_type == "season":
        group_cols = ['season', 'receiver_player_id', 'posteam']
    elif aggregation_type == "week":
        group_cols = ['season', 'week', 'receiver_player_id', 'posteam']
    else:  # career - aggregate across all seasons for the player
        group_cols = ['receiver_player_id', 'posteam']
    
    # WR-specific receiving stats
    receiver_filter = ~filtered_pbp['receiver_player_id'].isna()
    
    # Extract WR receiving stats
    receiving_stats = (
        filtered_pbp[receiver_filter]
        .groupby(group_cols)
        .agg({
            'complete_pass': 'sum',  # Receptions
            'yards_gained': 'sum',
            'touchdown': 'sum',
            'pass_touchdown': 'sum',
            'first_down_pass': 'sum',
            'fumble': 'sum',
            'fumble_lost': 'sum',
            'air_yards': 'sum',
            'yards_after_catch': 'sum',
            'epa': lambda x: x.sum(skipna=True),
            'game_id': 'nunique'  # For game count
        })
        .reset_index()
        .rename(columns={
            'receiver_player_id': 'player_id',
            'posteam': 'team',
            'complete_pass': 'receptions',
            'yards_gained': 'receiving_yards',
            'pass_touchdown': 'receiving_tds',
            'first_down_pass': 'receiving_first_downs',
            'fumble': 'receiving_fumbles',
            'fumble_lost': 'receiving_fumbles_lost',
            'air_yards': 'receiving_air_yards',
            'yards_after_catch': 'receiving_yards_after_catch',
            'epa': 'receiving_epa',
            'game_id': 'games_played'
        })
    )
    
    # Count targets from pbp data
    targets_count = (
        filtered_pbp[receiver_filter]
        .groupby(group_cols)
        .size()
        .reset_index()
        .rename(columns={0: 'targets'})
    )
    
    # Rename columns to match receiving_stats
    rename_dict = {}
    if 'receiver_player_id' in targets_count.columns:
        rename_dict['receiver_player_id'] = 'player_id'
    if 'posteam' in targets_count.columns:
        rename_dict['posteam'] = 'team'
    targets_count = targets_count.rename(columns=rename_dict)
    
    # Determine merge columns for targets
    target_merge_cols = []
    for col in group_cols:
        if col == 'receiver_player_id':
            target_merge_cols.append('player_id')
        elif col == 'posteam':
            target_merge_cols.append('team')
        else:
            target_merge_cols.append(col)
    
    # Merge the target counts with receiving stats
    receiving_stats = receiving_stats.merge(
        targets_count,
        on=target_merge_cols,
        how='left'
    )
    
    # Calculate receiving 2-point conversions
    rec_2pt_filter = (filtered_pbp['play_type'] == 'pass') & (filtered_pbp['two_point_conv_result'] == 'success')
    
    # Set grouping variables for 2-point conversions (receiving)
    if aggregation_type == "season":
        rec_2pt_groups = ['season', 'receiver_player_id']
    elif aggregation_type == "week":
        rec_2pt_groups = ['season', 'week', 'receiver_player_id']
    else:  # career - aggregate all seasons together
        rec_2pt_groups = ['receiver_player_id']
    
    # Check if we have any receiving 2-point conversions
    if rec_2pt_filter.sum() > 0:
        rec_2pt_stats = (
            filtered_pbp[rec_2pt_filter]
            .groupby(rec_2pt_groups)
            .size()
            .reset_index()
            .rename(columns={
                0: 'receiving_2pt_conversions',
                'receiver_player_id': 'player_id'
            })
        )
        
        # Join with receiving stats if we have any 2-point conversions
        if rec_2pt_stats.shape[0] > 0:
            # Determine merge columns based on aggregation_type
            if aggregation_type == "season":
                rec_2pt_merge_cols = ['season', 'player_id']
            elif aggregation_type == "week":
                rec_2pt_merge_cols = ['season', 'week', 'player_id']
            else:  # career
                rec_2pt_merge_cols = ['player_id']
            
            receiving_stats = receiving_stats.merge(
                rec_2pt_stats,
                on=rec_2pt_merge_cols,
                how='left'
            )
        else:
            receiving_stats['receiving_2pt_conversions'] = 0
    else:
        receiving_stats['receiving_2pt_conversions'] = 0
    
    # Calculate WR-specific metrics
    # RACR (Receiving Air Conversion Ratio)
    receiving_stats['racr'] = np.where(
        receiving_stats['receiving_air_yards'] != 0,
        receiving_stats['receiving_yards'] / receiving_stats['receiving_air_yards'],
        np.nan
    )
    
    # Calculate yards per reception
    receiving_stats['yards_per_reception'] = np.where(
        receiving_stats['receptions'] > 0,
        receiving_stats['receiving_yards'] / receiving_stats['receptions'],
        np.nan
    )
    
    # Calculate catch rate
    receiving_stats['catch_rate'] = np.where(
        receiving_stats['targets'] > 0,
        receiving_stats['receptions'] / receiving_stats['targets'],
        np.nan
    )
    
    # Calculate fantasy points for WRs
    receiving_stats['fantasy_points'] = (
        (1/10) * receiving_stats['receiving_yards'] +
        6 * receiving_stats['receiving_tds'] +
        2 * receiving_stats['receiving_2pt_conversions'] +
        -2 * receiving_stats['receiving_fumbles_lost']
    )
    
    # PPR fantasy points (add 1 point per reception)
    receiving_stats['fantasy_points_ppr'] = receiving_stats['fantasy_points'] + receiving_stats['receptions']
    
    # Add target share calculation - first we need team totals
    # Calculate team targets for each group level
    if aggregation_type == "season":
        team_groups = ['season', 'posteam']
    elif aggregation_type == "week":
        team_groups = ['season', 'week', 'posteam']
    else:  # career
        team_groups = ['posteam']
        
    team_targets = (
        filtered_pbp[~filtered_pbp['receiver_player_id'].isna()]
        .groupby(team_groups)
        .size()
        .reset_index()
        .rename(columns={0: 'team_targets', 'posteam': 'team'})
    )
    
    # Calculate team air yards
    team_air_yards = (
        filtered_pbp[~filtered_pbp['receiver_player_id'].isna()]
        .groupby(team_groups)
        .agg({'air_yards': 'sum'})
        .reset_index()
        .rename(columns={'air_yards': 'team_air_yards', 'posteam': 'team'})
    )
    
    # Merge team totals to calculate share metrics
    # Determine merge columns for team stats
    if aggregation_type == "season":
        team_merge_cols = ['season', 'team']
    elif aggregation_type == "week":
        team_merge_cols = ['season', 'week', 'team']
    else:  # career
        team_merge_cols = ['team']
        
    # Merge team targets
    receiving_stats = receiving_stats.merge(
        team_targets,
        on=team_merge_cols,
        how='left'
    )
    
    # Merge team air yards
    receiving_stats = receiving_stats.merge(
        team_air_yards,
        on=team_merge_cols,
        how='left'
    )
    
    # Calculate target share
    receiving_stats['target_share'] = np.where(
        receiving_stats['team_targets'] > 0,
        receiving_stats['targets'] / receiving_stats['team_targets'],
        0
    )
    
    # Calculate air yards share
    receiving_stats['air_yards_share'] = np.where(
        receiving_stats['team_air_yards'] > 0,
        receiving_stats['receiving_air_yards'] / receiving_stats['team_air_yards'],
        0
    )
    
    # Calculate WOPR (Weighted Opportunity Rating)
    # WOPR = 1.5 * target share + 0.7 * air yards share
    receiving_stats['wopr'] = 1.5 * receiving_stats['target_share'] + 0.7 * receiving_stats['air_yards_share']
    
    # Add player name if available in the play-by-play data
    if 'receiver_player_name' in filtered_pbp.columns:
        player_names = (
            filtered_pbp[['receiver_player_id', 'receiver_player_name']]
            .drop_duplicates()
            .rename(columns={
                'receiver_player_id': 'player_id',
                'receiver_player_name': 'player_name'
            })
        )
        receiving_stats = receiving_stats.merge(player_names, on='player_id', how='left')
    
    # For consistency with the R version, include season_type for non-career aggregations
    if 'season_type' in filtered_pbp.columns and aggregation_type != "career":
        if aggregation_type == "season":
            # For season summary, we use a concatenated string of unique season types
            season_types = (
                filtered_pbp[['season', 'season_type']]
                .drop_duplicates()
                .groupby('season')
                .agg({'season_type': lambda x: '+'.join(sorted(set(x)))})
                .reset_index()
            )
            receiving_stats = receiving_stats.merge(season_types, on='season', how='left')
        else:  # week
            # For weekly summaries, we just take the season type for that week
            season_types = (
                filtered_pbp[['season', 'week', 'season_type']]
                .drop_duplicates()
                .groupby(['season', 'week'])
                .agg({'season_type': 'first'})
                .reset_index()
            )
            receiving_stats = receiving_stats.merge(season_types, on=['season', 'week'], how='left')
    elif aggregation_type == "career" and 'season_type' in filtered_pbp.columns:
        # For career stats, add the season_type as supplied in the function parameter
        receiving_stats['season_type'] = season_type
    
    # Determine final columns based on aggregation_type
    base_cols = [
        'player_id', 'team', 'games_played',
        # Receiving stats
        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'receiving_first_downs', 'receiving_fumbles', 'receiving_fumbles_lost',
        'receiving_2pt_conversions', 'receiving_air_yards', 'receiving_yards_after_catch',
        'receiving_epa', 'racr', 'yards_per_reception', 'catch_rate',
        # Share metrics
        'target_share', 'air_yards_share', 'wopr',
        # Team totals
        'team_targets', 'team_air_yards',
        # Fantasy
        'fantasy_points', 'fantasy_points_ppr'
    ]
    
    # Add player_name to base_cols only if it exists in the DataFrame
    if 'player_name' in receiving_stats.columns:
        base_cols.insert(1, 'player_name')
    
    if aggregation_type == "season":
        result_cols = ['season'] + base_cols
    elif aggregation_type == "week":
        result_cols = ['season', 'week'] + base_cols
    else:  # career
        result_cols = base_cols
        
    # Add season_type if available
    if 'season_type' in receiving_stats.columns:
        result_cols.append('season_type')
    
    # Fill NaN values with appropriate defaults
    numeric_cols = receiving_stats.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        receiving_stats[col] = receiving_stats[col].fillna(0)
    
    # Ensure all columns exist (some might be missing if no data was found)
    for col in result_cols:
        if col not in receiving_stats.columns:
            receiving_stats[col] = 0 if col != 'player_name' else None
            
    # Add player name if requested
    result = receiving_stats[result_cols]
    
    # Add player name and position from players dataset if requested
    if add_player_name and 'player_id' in result.columns and not result.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            player_info = players_df[players_df['gsis_id'].isin(result['player_id'])].set_index('gsis_id')
            
            # Add player name if not already present
            if 'player_name' not in result.columns:
                result['player_name'] = result['player_id'].map(
                    lambda x: player_info.loc[x, 'display_name'] if x in player_info.index else f"Unknown ({x})"
                )
            
            # Always use the correct position based on the function type, don't use database position 
            # Database position may be incorrect (that's causing our issue)
            if 'calculate_qb_stats' in sys._getframe().f_code.co_name:
                result['position'] = "QB"
            elif 'calculate_rb_stats' in sys._getframe().f_code.co_name:
                result['position'] = "RB"
            elif 'calculate_wr_stats' in sys._getframe().f_code.co_name:
                result['position'] = "WR"
            # Add additional position functions as needed

        except Exception as e:
            print(f"Could not add player name/position: {e}")
    
    # Add player names for all players if requested
    elif add_player_name and 'player_name' not in result.columns and not result.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            # Create a mapping of player IDs to display names
            player_name_map = dict(zip(players_df['gsis_id'], players_df['display_name']))
            # Create a mapping for positions
            player_pos_map = dict(zip(players_df['gsis_id'], players_df['position']))
            
            # Add the display name as player_name column
            result['player_name'] = result['player_id'].map(
                lambda x: player_name_map.get(x, f"Unknown ({x})")
            )
            # For QB stats always set position to QB, don't use player_pos_map
            # This ensures correct position regardless of database values
            result['position'] = "QB"
        except Exception as e:
            print(f"Could not add player names/positions: {e}")
    
    # Ensure 'position' column is included in final selection if added
    if 'position' in result.columns and 'position' not in result_cols:
        # Try inserting after 'team' or 'player_name' if they exist
        try:
            insert_idx = result_cols.index('team') + 1
        except ValueError:
            try:
                insert_idx = result_cols.index('player_name') + 1
            except ValueError:
                insert_idx = 2 # Fallback index
        result_cols.insert(insert_idx, 'position')
    
    # Re-select columns to ensure order and inclusion of potentially added 'position'
    final_result = result[[col for col in result_cols if col in result.columns]]
    
    # Explicitly add/update position column using player_id just before returning
    if 'player_id' in final_result.columns and not final_result.empty:
        # Assuming position is consistent for a player within the group
        first_player_id = final_result['player_id'].iloc[0]
        if first_player_id:
            fetched_pos = get_player_position(first_player_id)
            final_result['position'] = fetched_pos
             # Ensure 'position' is in the final columns list again, in case it wasn't added before
            if 'position' not in final_result.columns.tolist():
                 cols = final_result.columns.tolist()
                 try:
                    insert_idx = cols.index('team') + 1
                 except ValueError:
                    try:
                        insert_idx = cols.index('player_name') + 1
                    except ValueError:
                        insert_idx = 2 # Fallback
                 cols.insert(insert_idx, 'position')
                 final_result = final_result[cols] # Recreate DataFrame with correct column order


    # Add the detected position back if it wasn't retrieved from players.parquet
    # This covers cases where add_player_name was False or player wasn't in players.parquet
    # if 'position' not in final_result.columns and position: # REMOVED - Handled above now
    #     final_result['position'] = position
    #     # Reorder to place position near name/team if possible
    #     cols = final_result.columns.tolist()
    #     if 'position' in cols:
    #          cols.remove('position')
    #          try:
    #              insert_idx = cols.index('team') + 1
    #          except ValueError:
    #              try:
    #                  insert_idx = cols.index('player_name') + 1
    #              except ValueError:
    #                  insert_idx = 2 # Fallback
    #          cols.insert(insert_idx, 'position')
    #          final_result = final_result[cols]

    # Ensure the position column exists and is populated
    if 'player_id' in final_result.columns and 'position' not in final_result.columns:
        try:
            # Use the first player_id in the group to determine position
            first_player_id = final_result['player_id'].iloc[0]
            if first_player_id:
                 final_result['position'] = get_player_position(first_player_id)
            else:
                 final_result['position'] = None
        except Exception as e:
            logger.error(f"Error adding position column in calculate_wr_stats: {e}")
            final_result['position'] = None
    elif 'position' not in final_result.columns:
         final_result['position'] = None


    return final_result

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
        position_stats['position'] = "QB"  # Explicitly set position for all players in this result
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
                
                # Merge with position_stats
                position_stats = position_stats.merge(
                    rushing_stats,
                    on=merge_cols,
                    how='left'
                )
                
                # Fill NAs with 0
                for col in rushing_stats.columns:
                    if col not in merge_cols and col in position_stats.columns:
                        position_stats[col] = position_stats[col].fillna(0)
                
                # Update fantasy points to include rushing
                position_stats['fantasy_points'] = (
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
                        position_stats[col] = 0
                
            # Ensure all rushing columns exist at this point regardless of flow path
            rushing_columns = [
                'rushing_yards', 'rushing_tds', 'rushing_first_downs', 
                'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_epa',
                'carries', 'yards_per_carry', 'rushing_2pt_conversions'
            ]
            
            # Initialize any missing rushing columns with zeros
            for col in rushing_columns:
                if col not in position_stats.columns:
                    position_stats[col] = 0
                    
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
        position_stats['position'] = "RB"  # Explicitly set position for all players in this result
        standardized_position = "RB"
    elif position_upper in WR_POSITIONS:
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
        position_stats['position'] = "WR"  # Explicitly set position for all players in this result
        standardized_position = "WR"
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
        position_stats['position'] = "TE"  # Explicitly set position for all players in this result
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
                
                # Merge with position_stats
                position_stats = position_stats.merge(
                    rushing_stats,
                    on=merge_cols,
                    how='left'
                )
                
                # Fill NAs with 0
                for col in rushing_stats.columns:
                    if col not in merge_cols and col in position_stats.columns:
                        position_stats[col] = position_stats[col].fillna(0)
            else:
                # If no rushing plays found, add all rushing columns with zeros to avoid KeyError
                position_stats['rushing_yards'] = 0
                position_stats['rushing_tds'] = 0
                position_stats['rushing_first_downs'] = 0
                position_stats['rushing_fumbles'] = 0
                position_stats['rushing_fumbles_lost'] = 0
                position_stats['rushing_epa'] = 0
                position_stats['carries'] = 0
                position_stats['yards_per_carry'] = 0
                
                # Also add the specific sort_by column if needed
                if sort_by not in position_stats.columns:
                    position_stats[sort_by] = 0
                
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
        
        top_players = filtered_stats.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position='last'
        ).head(n)
        print(f"DEBUG: After sorting, got {len(top_players)} top players for {aggregation_type} aggregation")
    
    # Always add player names from players.parquet (regardless of include_player_details setting)
    if not top_players.empty:
        try:
            players_df = pd.read_parquet(Path("cache/players.parquet"))
            
            # Create a mapping of player IDs to display names 
            player_name_map = dict(zip(players_df['gsis_id'], players_df['display_name']))
            
            # Add the display name as player_name column
            top_players['player_name'] = top_players['player_id'].map(
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
                    top_players[f'player_{col}'] = top_players['player_id'].map(
                        lambda x: player_info.loc[x, col] if x in player_info.index else None
                    )
                
                # Add database position as player_position but preserve the original position
                if 'position' in player_info.columns:
                    top_players['player_position'] = top_players['player_id'].map(
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