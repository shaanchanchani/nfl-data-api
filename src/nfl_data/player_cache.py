"""Functions for caching player responses."""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
import logging
from functools import lru_cache

from .config import CACHE_DIR
from .data_loader import (
    load_weekly_stats,
    load_players
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PLAYER_CACHE_DIR = CACHE_DIR / "player_responses"
PLAYER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_top_players_by_position(seasons: List[int], n_players: int = 24) -> Dict[str, List[Dict]]:
    """Get top players for each position based on fantasy points.
    
    Args:
        seasons: List of seasons to analyze
        n_players: Number of top players to return per position
        
    Returns:
        Dictionary mapping position to list of top player info
    """
    # Load data
    weekly_stats = pd.concat([load_weekly_stats([season]) for season in seasons])
    players_df = load_players()
    
    # Calculate fantasy points per game
    player_stats = weekly_stats.groupby(['player_id', 'player_name', 'position']).agg({
        'fantasy_points_ppr': ['mean', 'count'],  # mean for per game, count for games played
        'passing_yards': 'sum',
        'passing_tds': 'sum',
        'rushing_yards': 'sum',
        'rushing_tds': 'sum',
        'receptions': 'sum',
        'receiving_yards': 'sum',
        'receiving_tds': 'sum'
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['player_id', 'player_name', 'position', 
                          'fantasy_ppg', 'games_played',
                          'passing_yards', 'passing_tds',
                          'rushing_yards', 'rushing_tds',
                          'receptions', 'receiving_yards', 'receiving_tds']
    
    # Filter for players with minimum games played (at least 25% of possible games)
    min_games = len(seasons) * 17 * 0.25  # 17 games per season
    player_stats = player_stats[player_stats['games_played'] >= min_games]
    
    top_players = {}
    
    # Get top QBs
    qbs = player_stats[player_stats['position'] == 'QB'].nlargest(n_players, 'fantasy_ppg')
    top_players['QB'] = qbs.to_dict('records')
    
    # Get top RBs
    rbs = player_stats[player_stats['position'] == 'RB'].nlargest(n_players, 'fantasy_ppg')
    top_players['RB'] = rbs.to_dict('records')
    
    # Get top WRs
    wrs = player_stats[player_stats['position'] == 'WR'].nlargest(n_players, 'fantasy_ppg')
    top_players['WR'] = wrs.to_dict('records')
    
    return top_players

def cache_player_response(player_name: str, season: int, response: Dict) -> None:
    """Cache a player's API response.
    
    Args:
        player_name: Player's name
        season: Season year
        response: API response to cache
    """
    cache_file = PLAYER_CACHE_DIR / f"{player_name.lower().replace(' ', '_')}_{season}.json"
    with open(cache_file, 'w') as f:
        json.dump(response, f)
    logger.info(f"Cached response for {player_name} ({season})")

def get_cached_response(player_name: str, season: int) -> Optional[Dict]:
    """Get a cached player response if available.
    
    Args:
        player_name: Player's name
        season: Season year
        
    Returns:
        Cached response if available, None otherwise
    """
    cache_file = PLAYER_CACHE_DIR / f"{player_name.lower().replace(' ', '_')}_{season}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None

def precompute_top_player_responses(seasons: List[int] = None) -> None:
    """Precompute responses for top players.
    
    Args:
        seasons: List of seasons to analyze (defaults to [2024, 2023, 2022])
    """
    if seasons is None:
        seasons = [2024, 2023, 2022]
    
    # Get top players
    top_players = get_top_players_by_position(seasons)
    
    # Import necessary functions
    from .main import get_player_information
    
    # Cache responses for each player and season
    for position, players in top_players.items():
        for player in players:
            player_name = player['player_name']
            logger.info(f"Precomputing responses for {player_name}")
            
            for season in seasons:
                try:
                    # Skip if already cached
                    if get_cached_response(player_name, season):
                        logger.info(f"Response already cached for {player_name} ({season})")
                        continue
                    
                    # Get response
                    response = get_player_information(player_name, season)
                    
                    # Cache response
                    cache_player_response(player_name, season, response)
                    
                except Exception as e:
                    logger.error(f"Error precomputing response for {player_name} ({season}): {str(e)}") 