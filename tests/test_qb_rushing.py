"""Test QB rushing statistics functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from nfl_data.endpoints.player_stats import calculate_qb_stats

@pytest.fixture
def sample_pbp_data():
    """Create a sample play-by-play DataFrame for testing QB rushing stats."""
    # Create a small sample dataset with both passing and rushing plays
    data = {
        'season': [2023, 2023, 2023, 2023, 2023, 2023, 2023],
        'week': [1, 1, 1, 2, 2, 2, 2],
        'season_type': ['REG', 'REG', 'REG', 'REG', 'REG', 'REG', 'REG'],
        'play_type': ['pass', 'pass', 'run', 'pass', 'run', 'run', 'pass'],
        'passer_player_id': ['00-0033873', '00-0033873', None, '00-0033873', None, None, '00-0033873'],
        'passer_player_name': ['Patrick Mahomes', 'Patrick Mahomes', None, 'Patrick Mahomes', None, None, 'Patrick Mahomes'],
        'rusher_player_id': [None, None, '00-0033873', None, '00-0033873', '00-0029520', None],
        'rusher_player_name': [None, None, 'Patrick Mahomes', None, 'Patrick Mahomes', 'Isiah Pacheco', None],
        'posteam': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
        'defteam': ['LV', 'LV', 'LV', 'DEN', 'DEN', 'DEN', 'DEN'],
        'yardline_100': [75, 65, 55, 45, 35, 25, 15],
        'down': [1, 2, 3, 1, 2, 3, 4],
        'yards_gained': [10, 15, 12, 20, 8, 7, 15],
        'touchdown': [0, 1, 1, 0, 0, 1, 0],
        'pass_touchdown': [0, 1, 0, 0, 0, 0, 0],
        'rush_touchdown': [0, 0, 1, 0, 0, 1, 0],
        'complete_pass': [1, 1, 0, 1, 0, 0, 1],
        'incomplete_pass': [0, 0, 0, 0, 0, 0, 0],
        'interception': [0, 0, 0, 0, 0, 0, 0],
        'sack': [0, 0, 0, 0, 0, 0, 0],
        'fumble': [0, 0, 0, 0, 0, 0, 0],
        'fumble_lost': [0, 0, 0, 0, 0, 0, 0],
        'qb_epa': [0.5, 2.5, 0, 1.5, 0, 0, 0.75],
        'cpoe': [5.0, 10.0, None, 7.5, None, None, 15.0],
        'air_yards': [7, 12, 0, 15, 0, 0, 10],
        'yards_after_catch': [3, 3, 0, 5, 0, 0, 5],
        'epa': [0.5, 2.5, 1.5, 1.5, 0.8, 0.6, 0.75],
        'first_down_pass': [0, 1, 0, 1, 0, 0, 1],
        'first_down_rush': [0, 0, 1, 0, 0, 1, 0],
        'game_id': ['2023_01_KC_LV', '2023_01_KC_LV', '2023_01_KC_LV',
                   '2023_02_KC_DEN', '2023_02_KC_DEN', '2023_02_KC_DEN', '2023_02_KC_DEN'],
        'two_point_conv_result': [None, None, None, None, None, None, None]
    }
    
    # Convert to DataFrame
    return pd.DataFrame(data)

def test_qb_stats_includes_rushing(sample_pbp_data):
    """Test that calculate_qb_stats includes QB rushing stats."""
    # Calculate stats for Mahomes
    qb_stats = calculate_qb_stats(
        pbp=sample_pbp_data,
        player_id='00-0033873',  # Patrick Mahomes
        aggregation_type='season',
        seasons=2023,
        season_type='REG'
    )
    
    # Verify passing stats are present
    assert 'passing_yards' in qb_stats.columns
    assert 'passing_tds' in qb_stats.columns
    assert 'passing_interceptions' in qb_stats.columns
    assert 'completions' in qb_stats.columns
    assert 'attempts' in qb_stats.columns
    
    # Verify rushing stats are present
    assert 'rushing_yards' in qb_stats.columns
    assert 'rushing_tds' in qb_stats.columns
    assert 'carries' in qb_stats.columns
    
    # Verify the values match our sample data
    assert qb_stats['passing_yards'].iloc[0] == 60  # 10 + 15 + 20 + 15
    assert qb_stats['passing_tds'].iloc[0] == 1
    assert qb_stats['rushing_yards'].iloc[0] == 20  # 12 + 8
    assert qb_stats['rushing_tds'].iloc[0] == 1
    assert qb_stats['carries'].iloc[0] == 2

def test_qb_stats_all_qbs_includes_rushing(sample_pbp_data):
    """Test calculate_qb_stats includes rushing stats for all QBs."""
    # Calculate stats for all QBs
    qb_stats = calculate_qb_stats(
        pbp=sample_pbp_data,
        aggregation_type='season',
        seasons=2023,
        season_type='REG'
    )
    
    # Verify passing and rushing stats are present
    assert 'passing_yards' in qb_stats.columns
    assert 'rushing_yards' in qb_stats.columns
    assert 'carries' in qb_stats.columns
    
    # Verify only Mahomes appears in results
    assert len(qb_stats) == 1
    assert qb_stats['player_id'].iloc[0] == '00-0033873'
    
    # Verify the rushing stats values
    assert qb_stats['rushing_yards'].iloc[0] == 20
    assert qb_stats['rushing_tds'].iloc[0] == 1
    assert qb_stats['carries'].iloc[0] == 2

def test_qb_stats_fantasy_points_include_rushing(sample_pbp_data):
    """Test fantasy points calculation includes rushing stats for QBs."""
    # Calculate stats for Mahomes
    qb_stats = calculate_qb_stats(
        pbp=sample_pbp_data,
        player_id='00-0033873',  # Patrick Mahomes
        aggregation_type='season',
        seasons=2023,
        season_type='REG'
    )
    
    # Verify fantasy points are calculated correctly with rushing stats
    passing_fantasy_points = (
        (1/25) * 60 +  # Passing yards: 60 / 25 = 2.4
        4 * 1 +  # Passing TDs: 4 * 1 = 4
        -2 * 0 +  # Interceptions: -2 * 0 = 0
        -2 * 0 +  # Fumbles lost: -2 * 0 = 0
        2 * 0  # 2pt conversions: 2 * 0 = 0
    )
    
    rushing_fantasy_points = (
        (1/10) * 20 +  # Rushing yards: 20 / 10 = 2
        6 * 1 +  # Rushing TDs: 6 * 1 = 6
        -2 * 0 +  # Fumbles lost: -2 * 0 = 0
        2 * 0  # 2pt conversions: 2 * 0 = 0
    )
    
    expected_fantasy_points = passing_fantasy_points + rushing_fantasy_points
    
    # Using almost equal to account for floating point precision
    assert abs(qb_stats['fantasy_points'].iloc[0] - expected_fantasy_points) < 0.0001
    
    # Verify total touchdowns calculation
    assert 'total_tds' in qb_stats.columns
    assert qb_stats['total_tds'].iloc[0] == 2  # 1 passing TD + 1 rushing TD

def test_qb_stats_by_week_includes_rushing(sample_pbp_data):
    """Test weekly QB stats include rushing stats."""
    # Calculate weekly stats
    qb_stats = calculate_qb_stats(
        pbp=sample_pbp_data,
        player_id='00-0033873',  # Patrick Mahomes
        aggregation_type='week',
        seasons=2023,
        season_type='REG'
    )
    
    # Verify there are two weeks of data
    assert len(qb_stats) == 2
    
    # Sort by week for easier assertions
    qb_stats = qb_stats.sort_values('week')
    
    # Week 1 stats
    week1 = qb_stats.iloc[0]
    assert week1['week'] == 1
    assert week1['passing_yards'] == 25  # 10 + 15
    assert week1['rushing_yards'] == 12  # One rush for 12 yards
    
    # Week 2 stats
    week2 = qb_stats.iloc[1]
    assert week2['week'] == 2
    assert week2['passing_yards'] == 35  # 20 + 15
    assert week2['rushing_yards'] == 8  # One rush for 8 yards

def test_init_of_required_columns(sample_pbp_data):
    """Test that required columns are initialized even in edge cases."""
    # Create a copy of the sample data with no rushing plays for the QB
    no_rushing_pbp = sample_pbp_data.copy()
    
    # Remove QB rushing plays to test initialization of columns
    no_rushing_pbp.loc[no_rushing_pbp['rusher_player_id'] == '00-0033873', 'rusher_player_id'] = None
    no_rushing_pbp.loc[no_rushing_pbp['rusher_player_name'] == 'Patrick Mahomes', 'rusher_player_name'] = None
    
    # Calculate stats with no rushing data
    qb_stats = calculate_qb_stats(
        pbp=no_rushing_pbp,
        player_id='00-0033873',  # Patrick Mahomes
        aggregation_type='season',
        seasons=2023,
        season_type='REG'
    )
    
    # Verify required columns exist
    assert 'rushing_yards' in qb_stats.columns
    assert 'qb_dropback' in qb_stats.columns
    assert 'qb_epa' in qb_stats.columns
    assert 'epa_per_dropback' in qb_stats.columns
    assert 'fantasy_points' in qb_stats.columns
    
    # Verify rushing values are zero
    assert qb_stats['rushing_yards'].iloc[0] == 0
    assert qb_stats['rushing_tds'].iloc[0] == 0
    assert qb_stats['carries'].iloc[0] == 0