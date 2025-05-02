import pandas as pd
from typing import Optional, List, Union
import numpy as np

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
        'rushing_yards', 'rushing_tds',
        'total_tds',  # Combined passing + rushing TDs
        'qb_epa', 'passing_cpoe', 'epa_per_dropback',
        'passing_air_yards', 'passing_yards_after_catch', 'pacr',
        'passing_first_downs', 'passing_2pt_conversions',
        'sacks_suffered', 'passing_fumbles', 'passing_fumbles_lost', 
        'qb_dropback', 'fantasy_points'
    ]
    
    # Add rushing carries column if it exists in the DataFrame
    if 'carries' in passing_stats.columns:
        base_cols.insert(11, 'carries')
    
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


    # Final cleanup - remove any columns with _x and _y suffixes and handle other redundant columns
    if not final_result.empty:
        # Find all columns with _x or _y suffixes
        x_columns = [col for col in final_result.columns if col.endswith('_x')]
        y_columns = [col for col in final_result.columns if col.endswith('_y')]
        
        print(f"DEBUG: Found {len(x_columns)} columns with _x suffix and {len(y_columns)} columns with _y suffix")
        
        # Process each x column
        for col_x in x_columns:
            base_col = col_x[:-2]  # Remove _x suffix
            col_y = f"{base_col}_y"  # Corresponding _y column
            
            # If there's also a _y version, prefer that data (from rushing stats)
            if col_y in y_columns:
                # Always use the _y column's data for the base column
                final_result[base_col] = final_result[col_y]
                # Drop both suffixed columns
                final_result = final_result.drop(columns=[col_x, col_y])
                print(f"DEBUG: Final cleanup - Fixed duplicate {base_col} by using {col_y} and dropping both")
            else:
                # If only _x exists, use its data
                final_result[base_col] = final_result[col_x]
                # Drop the suffixed column
                final_result = final_result.drop(columns=[col_x])
                print(f"DEBUG: Final cleanup - Fixed {col_x} by removing suffix")
        
        # Handle any remaining _y columns
        remaining_y_cols = [col for col in final_result.columns if col.endswith('_y')]
        for col_y in remaining_y_cols:
            base_col = col_y[:-2]  # Remove _y suffix
            final_result[base_col] = final_result[col_y]
            final_result = final_result.drop(columns=[col_y])
            print(f"DEBUG: Final cleanup - Fixed {col_y} by removing suffix")
        
        # Remove redundant touchdown column if we have passing_tds/rushing_tds
        if 'touchdown' in final_result.columns and 'passing_tds' in final_result.columns:
            final_result = final_result.drop(columns=['touchdown'])
            print(f"DEBUG: Final cleanup - Removed redundant 'touchdown' column")
        
        # Ensure all required columns for fantasy points exist
        fantasy_required_cols = [
            'passing_yards', 'passing_tds', 'passing_interceptions',
            'passing_fumbles_lost', 'passing_2pt_conversions',
            'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions'
        ]
        
        # Fill missing fantasy point columns with zeros
        for col in fantasy_required_cols:
            if col not in final_result.columns:
                final_result[col] = 0
                print(f"DEBUG: Final cleanup - Added missing fantasy column {col}")
            else:
                final_result[col] = final_result[col].fillna(0)
        
        # Recalculate fantasy points with cleaned data
        final_result['fantasy_points'] = (
            (1/25) * final_result['passing_yards'] +
            4 * final_result['passing_tds'] +
            -2 * final_result['passing_interceptions'] +
            -2 * final_result['passing_fumbles_lost'] +
            2 * final_result['passing_2pt_conversions'] +
            # Add rushing points
            (1/10) * final_result['rushing_yards'] +
            6 * final_result['rushing_tds'] +
            -2 * final_result['rushing_fumbles_lost'] +
            2 * final_result['rushing_2pt_conversions']
        )
        print(f"DEBUG: Final cleanup - Recalculated fantasy points with cleaned data")
    
    return final_result
