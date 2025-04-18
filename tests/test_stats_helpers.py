import pytest
import pandas as pd
import numpy as np
from src.nfl_data.stats_helpers import get_position_specific_stats_from_pbp

# Test data fixtures
@pytest.fixture
def qb_plays():
    """Sample play-by-play data for QB tests"""
    return pd.DataFrame({
        'pass_attempt': [1, 1, 1, 1, 0],
        'complete_pass': [1, 0, 1, 1, 0],
        'passing_yards': [10, 0, 15, 20, 0],
        'pass_touchdown': [0, 0, 1, 0, 0],
        'interception': [0, 1, 0, 0, 0],
        'rush_attempt': [0, 0, 0, 0, 1],
        'rushing_yards': [0, 0, 0, 0, 5],
        'sack': [0, 0, 0, 0, 0]
    })

@pytest.fixture
def rb_plays():
    """Sample play-by-play data for RB tests"""
    return pd.DataFrame({
        'rush_attempt': [1, 1, 1, 0, 1],
        'rushing_yards': [5, 10, -2, 0, 15],
        'rush_touchdown': [0, 1, 0, 0, 0],
        'broken_tackles': [1, 2, 0, 0, 1],
        'receiver_player_id': [None, None, None, '123', None],
        'first_down_rush': [0, 1, 0, 0, 0],
        'fumble': [0, 0, 0, 0, 0]
    })

@pytest.fixture
def wr_plays():
    """Sample play-by-play data for WR tests"""
    return pd.DataFrame({
        'receiver_player_id': ['123', '123', '123', '123', '123'],
        'complete_pass': [1, 0, 1, 1, 0],
        'receiving_yards': [10, 0, 15, 20, 0],
        'pass_touchdown': [0, 0, 1, 0, 0],
        'yards_after_catch': [5, 0, 8, 10, 0],
        'incomplete_pass': [0, 1, 0, 0, 1],
        'first_down_pass': [1, 0, 1, 0, 0]
    })

def test_empty_plays():
    """Test handling of empty play data"""
    empty_df = pd.DataFrame()
    assert get_position_specific_stats_from_pbp(empty_df, 'QB') == {}
    assert get_position_specific_stats_from_pbp(empty_df, 'RB') == {}
    assert get_position_specific_stats_from_pbp(empty_df, 'WR') == {}
    assert get_position_specific_stats_from_pbp(empty_df, 'TE') == {}

def test_qb_stats(qb_plays):
    """Test QB statistics calculation"""
    stats = get_position_specific_stats_from_pbp(qb_plays, 'QB')
    
    assert isinstance(stats, dict)
    
    # Test completion percentage: 3 completions out of 4 attempts = 75%
    assert stats['completion_percentage'] == 75.0
    
    # Test yards per attempt: 45 total yards / 4 attempts = 11.25
    assert stats['yards_per_attempt'] == 11.25
    
    # Test touchdown percentage: 1 TD / 4 attempts = 25%
    assert stats['touchdown_percentage'] == 25.0
    
    # Test interception percentage: 1 INT / 4 attempts = 25%
    assert stats['interception_percentage'] == 25.0

def test_rb_stats(rb_plays):
    """Test RB statistics calculation"""
    stats = get_position_specific_stats_from_pbp(rb_plays, 'RB')
    
    assert isinstance(stats, dict)
    
    # Test yards per carry: 28 yards / 4 attempts = 7.0
    assert stats['yards_per_carry'] == 7.0
    
    # Test rush touchdown rate: 1 TD / 4 attempts = 25%
    assert stats['rush_touchdown_rate'] == 25.0
    
    # Test first down rate: 1 first down / 4 attempts = 25%
    assert stats['first_down_rate'] == 25.0

def test_wr_stats(wr_plays):
    """Test WR statistics calculation"""
    stats = get_position_specific_stats_from_pbp(wr_plays, 'WR')
    
    assert isinstance(stats, dict)
    
    # Test catch rate: 3 catches / 5 targets = 60%
    assert stats['catch_rate'] == 60.0
    
    # Test yards per reception: 45 yards / 3 receptions = 15.0
    assert stats['yards_per_reception'] == 15.0
    
    # Test touchdown rate: 1 TD / 5 targets = 20%
    assert stats['touchdown_rate'] == 20.0
    
    # Test yards per target: 45 yards / 5 targets = 9.0
    assert stats['yards_per_target'] == 9.0
    
    # Test yards after catch per reception: 23 YAC / 3 receptions = 7.67
    assert abs(stats['yac_per_reception'] - 7.67) < 0.01

def test_te_stats(wr_plays):
    """Test TE statistics calculation (should be same as WR)"""
    wr_stats = get_position_specific_stats_from_pbp(wr_plays, 'WR')
    te_stats = get_position_specific_stats_from_pbp(wr_plays, 'TE')
    assert wr_stats == te_stats

def test_unknown_position():
    """Test handling of unknown position"""
    df = pd.DataFrame({'some_column': [1, 2, 3]})
    assert get_position_specific_stats_from_pbp(df, 'K') == {} 