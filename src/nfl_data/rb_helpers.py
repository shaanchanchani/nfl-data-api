import pandas as pd
from typing import Optional, List, Union
import numpy as np
import sys
from pathlib import Path
# Import the player_position function from stats_helpers
from src.nfl_data.stats_helpers import get_player_position

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
        
        # Remove redundant touchdown column if we have rushing_tds/receiving_tds
        if 'touchdown' in final_result.columns and ('rushing_tds' in final_result.columns or 'receiving_tds' in final_result.columns):
            final_result = final_result.drop(columns=['touchdown'])
            print(f"DEBUG: Final cleanup - Removed redundant 'touchdown' column")
        
        # Ensure all required columns for fantasy points exist
        fantasy_required_cols = [
            'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions',
            'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost', 'receiving_2pt_conversions'
        ]
        
        # Fill missing fantasy point columns with zeros
        for col in fantasy_required_cols:
            if col not in final_result.columns:
                final_result[col] = 0
                print(f"DEBUG: Final cleanup - Added missing fantasy column {col}")
            else:
                final_result[col] = final_result[col].fillna(0)
        
        # Recalculate fantasy points with cleaned data
        if 'fantasy_points' in final_result.columns:
            final_result['fantasy_points'] = (
                # Rushing points
                (1/10) * final_result['rushing_yards'] +
                6 * final_result['rushing_tds'] +
                -2 * final_result.get('rushing_fumbles_lost', 0) +
                2 * final_result.get('rushing_2pt_conversions', 0) +
                # Add receiving points
                (1/10) * final_result['receiving_yards'] +
                6 * final_result['receiving_tds'] +
                -2 * final_result.get('receiving_fumbles_lost', 0) +
                2 * final_result.get('receiving_2pt_conversions', 0)
            )
            print(f"DEBUG: Final cleanup - Recalculated fantasy points with cleaned data")
        
        # Also recalculate PPR fantasy points if they exist
        if 'fantasy_points_ppr' in final_result.columns:
            final_result['fantasy_points_ppr'] = final_result['fantasy_points'] + final_result['receptions']
            print(f"DEBUG: Final cleanup - Recalculated PPR fantasy points")
    
    return final_result