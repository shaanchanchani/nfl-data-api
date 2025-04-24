"""Functions for loading and caching nflverse data."""

import os
import pandas as pd
import requests
import requests_cache
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
from functools import lru_cache
import time
from requests.exceptions import RequestException
import logging
import redis.asyncio as redis
import json

from .config import (
    CACHE_DIR,
    NFLVERSE_BASE_URL,
    MAX_RETRIES,
    RETRY_DELAY,
    DATASET_VERSIONS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Redis for GitHub API caching
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
GITHUB_CACHE_PREFIX = "github-api:"
GITHUB_CACHE_EXPIRE = 3600  # 1 hour in seconds

async def get_from_cache(key: str) -> Optional[Dict]:
    """Get cached GitHub API response."""
    cached = await redis_client.get(f"{GITHUB_CACHE_PREFIX}{key}")
    return json.loads(cached) if cached else None

async def set_in_cache(key: str, value: Dict) -> None:
    """Cache GitHub API response."""
    await redis_client.set(
        f"{GITHUB_CACHE_PREFIX}{key}",
        json.dumps(value),
        ex=GITHUB_CACHE_EXPIRE
    )

async def get_latest_release_info() -> Dict:
    """Get the latest release information from nflverse."""
    cache_key = "latest_release"
    
    # Check cache first
    cached = await get_from_cache(cache_key)
    if cached:
        return cached
    
    # If not in cache, fetch from GitHub
    for attempt in range(MAX_RETRIES):
        try:
            url = "https://api.github.com/repos/nflverse/nflverse-data/releases/latest"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            await set_in_cache(cache_key, data)
            return data
        except RequestException as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed to get latest release info: {str(e)}")
                raise RuntimeError(f"Failed to get latest release info: {str(e)}")
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Failed to get latest release info after {MAX_RETRIES} attempts")

async def get_dataset_version(dataset: str) -> str:
    """Get the appropriate version tag for a dataset."""
    if dataset in DATASET_VERSIONS:
        return DATASET_VERSIONS[dataset]
    
    # Fallback to latest release
    try:
        release_info = await get_latest_release_info()
        return release_info["tag_name"]
    except Exception as e:
        logger.error(f"Failed to get version for {dataset}: {str(e)}")
        return "latest"

def download_parquet(url: str, cache_path: Path, dataset_name: str = "") -> None:
    """Download a parquet file from nflverse with improved error handling."""
    temp_path = cache_path.with_suffix('.tmp')
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading {dataset_name} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Check content type (but be permissive as some GitHub responses may vary)
            content_type = response.headers.get('content-type', '')
            if ('application/octet-stream' not in content_type and 
                'application/x-parquet' not in content_type and 
                'application/vnd.github.v3.raw' not in content_type and
                'binary/octet-stream' not in content_type):
                logger.warning(f"Unexpected content type for {dataset_name}: {content_type}, but continuing anyway")
            
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file size
            if temp_path.stat().st_size < 100:  # Arbitrary minimum size
                raise ValueError(f"Downloaded file for {dataset_name} is too small")
            
            # Atomic rename
            temp_path.rename(cache_path)
            logger.info(f"Successfully downloaded {dataset_name} to {cache_path}")
            return
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {dataset_name}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Failed to download {dataset_name} from {url}: {str(e)}")
            time.sleep(RETRY_DELAY)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    raise RuntimeError(f"Failed to download {dataset_name} after {MAX_RETRIES} attempts")

def safe_read_parquet(path: Path, dataset_name: str = "") -> pd.DataFrame:
    """Safely read a parquet file with improved error handling."""
    try:
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"Empty DataFrame loaded from {dataset_name}")
        return df
    except Exception as e:
        logger.error(f"Failed to read {dataset_name} from {path}: {str(e)}")
        # Delete corrupted file
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to read {dataset_name} from {path}: {str(e)}")

def load_pbp_data() -> pd.DataFrame:
    """Load play-by-play data from the condensed parquet file."""
    # First try the configured cache directory
    cache_path = CACHE_DIR / "play_by_play_condensed.parquet"
    
    # If file doesn't exist in the default cache directory, try the development path
    if not cache_path.exists():
        dev_cache_path = Path(os.path.expanduser("~/dev/nfl-data-api/cache/play_by_play_condensed.parquet"))
        if dev_cache_path.exists():
            logger.info(f"Using development path for play-by-play data: {dev_cache_path}")
            return pd.read_parquet(dev_cache_path)
        else:
            # Also try a relative path from current directory
            relative_path = Path("./cache/play_by_play_condensed.parquet")
            if relative_path.exists():
                logger.info(f"Using relative path for play-by-play data: {relative_path}")
                return pd.read_parquet(relative_path)
            
            raise FileNotFoundError(f"Condensed play-by-play file not found: {cache_path}, {dev_cache_path}, or {relative_path}")
    
    return pd.read_parquet(cache_path)

def extract_situational_stats_from_pbp(pbp_data: pd.DataFrame, player_id_variations: list, situation_type: str, 
                                       season: Optional[int] = None, week: Optional[int] = None) -> Dict:
    """
    Extract situation-specific stats from play-by-play data for a player.
    
    Args:
        pbp_data: DataFrame containing play-by-play data
        player_id_variations: List of player IDs to search for
        situation_type: Type of situation to filter by (red_zone, third_down, etc.)
        season: Optional season to filter by
        week: Optional week to filter by
        
    Returns:
        Dictionary of stats for the specified situation
    """
    # Make sure player_id_variations is a list and not empty
    if not player_id_variations:
        return {"error": "No player IDs provided"}
    
    # Skip known empty or invalid IDs
    valid_player_ids = [pid for pid in player_id_variations if pid and not pd.isna(pid)]
    if not valid_player_ids:
        return {"error": "No valid player IDs provided"}
    
    # Log player IDs we're looking for
    logger.info(f"Searching for player IDs in PBP data: {valid_player_ids}")
    
    # Initialize a mask for all plays matching this player
    player_plays_mask = False
    
    # Only check columns that exist in the dataframe and are more likely to have this player
    player_cols = [
        'passer_player_id', 'receiver_player_id', 'rusher_player_id',
        'lateral_receiver_player_id', 'lateral_rusher_player_id',
        'fumbled_1_player_id', 'fumbled_2_player_id'
    ]
    
    # Filter the columns to only those that exist in the dataframe
    valid_cols = [col for col in player_cols if col in pbp_data.columns]
    logger.info(f"Valid player ID columns in PBP data: {valid_cols}")
    
    # Check each player ID in each valid column
    for pid in valid_player_ids:
        for col in valid_cols:
            player_plays_mask = player_plays_mask | (pbp_data[col] == pid)
            
    # Log number of plays found for debugging
    play_count = player_plays_mask.sum()
    logger.info(f"Found {play_count} plays involving the player")
    
    # Get plays involving this player
    player_plays = pbp_data[player_plays_mask].copy()
    
    # Apply season filter to PBP data if specified
    if season:
        player_plays = player_plays[player_plays['season'] == season]
        logger.info(f"After season filter: {len(player_plays)} plays")
    
    # Apply week filter to PBP data if specified
    if week:
        player_plays = player_plays[player_plays['week'] == week]
        logger.info(f"After week filter: {len(player_plays)} plays")
        
    # Apply situation-specific filters
    if situation_type == "red_zone":
        # Red zone plays (inside the 20 yard line)
        situation_plays = player_plays[player_plays['yardline_100'] <= 20]
        logger.info(f"Found {len(situation_plays)} red zone plays for player")
        
    elif situation_type == "third_down":
        # Third down plays
        situation_plays = player_plays[player_plays['down'] == 3]
        logger.info(f"Found {len(situation_plays)} third down plays for player")
        
    elif situation_type == "fourth_down":
        # Fourth down plays
        situation_plays = player_plays[player_plays['down'] == 4]
        logger.info(f"Found {len(situation_plays)} fourth down plays for player")
        
    elif situation_type == "goal_line":
        # Goal line plays (inside the 5 yard line)
        situation_plays = player_plays[player_plays['yardline_100'] <= 5]
        logger.info(f"Found {len(situation_plays)} goal line plays for player")
        
    elif situation_type == "two_minute_drill":
        # Two minute drill (last 2 minutes of half or game)
        two_min_mask = (
            ((player_plays['qtr'] == 2) & (player_plays['half_seconds_remaining'] <= 120)) |
            ((player_plays['qtr'] == 4) & (player_plays['half_seconds_remaining'] <= 120))
        )
        situation_plays = player_plays[two_min_mask]
        logger.info(f"Found {len(situation_plays)} two-minute drill plays for player")
    else:
        # Default to all plays if situation type not recognized
        situation_plays = player_plays
        logger.warning(f"Unrecognized situation type: {situation_type}")
    
    # If we don't have any plays, return early
    if situation_plays.empty:
        return {"play_count": 0, "note": f"No {situation_type} plays found for this player"}
    
    # Calculate basic stats from situation plays
    stats = {}
    
    # Extract games with situation plays
    stats['games'] = int(situation_plays['game_id'].nunique())
    stats['play_count'] = int(len(situation_plays))
    
    # Passing stats
    passer_plays = situation_plays[situation_plays['passer_player_id'].isin(valid_player_ids)]
    if not passer_plays.empty:
        stats['passing_attempts'] = int(len(passer_plays))
        stats['passing_completions'] = int(len(passer_plays[passer_plays['complete_pass'] == 1])) if 'complete_pass' in passer_plays.columns else 0
        stats['passing_yards'] = float(passer_plays['yards_gained'].sum()) if 'yards_gained' in passer_plays.columns else 0
        stats['passing_tds'] = int(passer_plays['pass_touchdown'].sum()) if 'pass_touchdown' in passer_plays.columns else 0
        stats['interceptions'] = int(passer_plays['interception'].sum()) if 'interception' in passer_plays.columns else 0
        stats['sacks'] = int(passer_plays['sack'].sum()) if 'sack' in passer_plays.columns else 0
        stats['passing_epa'] = float(passer_plays['epa'].sum()) if 'epa' in passer_plays.columns else 0
    
    # Rushing stats  
    rusher_plays = situation_plays[situation_plays['rusher_player_id'].isin(valid_player_ids)]
    if not rusher_plays.empty:
        stats['rushing_attempts'] = int(len(rusher_plays))
        stats['rushing_yards'] = float(rusher_plays['yards_gained'].sum()) if 'yards_gained' in rusher_plays.columns else 0
        stats['rushing_tds'] = int(rusher_plays['rush_touchdown'].sum()) if 'rush_touchdown' in rusher_plays.columns else 0
        stats['rushing_epa'] = float(rusher_plays['epa'].sum()) if 'epa' in rusher_plays.columns else 0
        stats['rushing_fumbles'] = int(rusher_plays['fumble_lost'].sum()) if 'fumble_lost' in rusher_plays.columns else 0
    
    # Receiving stats
    receiver_plays = situation_plays[situation_plays['receiver_player_id'].isin(valid_player_ids)]
    if not receiver_plays.empty:
        stats['targets'] = int(len(receiver_plays))
        stats['receptions'] = int(len(receiver_plays[receiver_plays['complete_pass'] == 1])) if 'complete_pass' in receiver_plays.columns else 0
        stats['receiving_yards'] = float(receiver_plays['yards_gained'].sum()) if 'yards_gained' in receiver_plays.columns else 0
        stats['receiving_tds'] = int(receiver_plays['pass_touchdown'].sum()) if 'pass_touchdown' in receiver_plays.columns else 0
        if 'air_yards' in receiver_plays.columns:
            stats['air_yards'] = float(receiver_plays['air_yards'].sum())
        if 'yards_after_catch' in receiver_plays.columns:
            stats['yards_after_catch'] = float(receiver_plays['yards_after_catch'].sum())
    
    # Add a summary field for total TDs
    total_tds = 0
    if 'passing_tds' in stats:
        total_tds += stats['passing_tds']
    if 'rushing_tds' in stats:
        total_tds += stats['rushing_tds']
    if 'receiving_tds' in stats:
        total_tds += stats['receiving_tds']
    stats['total_tds'] = total_tds
    
    # Success rate - success for offense vs expected
    if 'success' in situation_plays.columns:
        success_plays = situation_plays[situation_plays['success'] == 1].shape[0]
        stats['success_rate'] = round(success_plays / len(situation_plays) * 100, 1)
    
    return stats

async def load_weekly_stats(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load weekly player stats from the condensed parquet file."""
    try:
        # Use condensed file if available
        condensed_path = CACHE_DIR / "weekly_stats_condensed.parquet"
        if condensed_path.exists():
            logger.info(f"Loading weekly stats from condensed file: {condensed_path}")
            stats_df = pd.read_parquet(condensed_path)
            
            # Filter by seasons if provided
            if seasons:
                stats_df = stats_df[stats_df['season'].isin(seasons)]
                
            return stats_df
        
        # Fall back to individual season files if condensed file not available
        logger.warning("Condensed weekly stats file not found, falling back to individual season files")
        if seasons is None:
            seasons = [2024]
        
        dfs = []
        errors = []
        
        for season in seasons:
            try:
                cache_path = CACHE_DIR / f"player_stats_{season}.parquet"
                
                # Download if not in cache
                if not cache_path.exists():
                    version = await get_dataset_version("player_stats")
                    url = f"{NFLVERSE_BASE_URL}/{version}/player_stats_{season}.parquet"
                    download_parquet(url, cache_path, f"player_stats_{season}")
                
                # Load from cache
                df = safe_read_parquet(cache_path, f"player_stats_{season}")
                dfs.append(df)
            except Exception as e:
                errors.append(f"Season {season}: {str(e)}")
        
        if not dfs and errors:
            raise RuntimeError(f"Failed to load any seasons. Errors: {'; '.join(errors)}")
        
        return pd.concat(dfs) if dfs else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading weekly stats: {str(e)}")
        return pd.DataFrame()

async def load_players() -> pd.DataFrame:
    """Load player information from the condensed parquet file."""
    try:
        # Try loading from condensed file first
        condensed_path = CACHE_DIR / "players_condensed.parquet"
        if condensed_path.exists():
            logger.info(f"Loading players from condensed file: {condensed_path}")
            df = pd.read_parquet(condensed_path)
            if not df.empty:
                return df
                
        # Fall back to standard players file if condensed not available
        logger.warning("Condensed players file not found, falling back to standard file")
        cache_path = CACHE_DIR / "players.parquet"
        
        # Download if not in cache or older than 1 day
        if not cache_path.exists() or (datetime.now().timestamp() - cache_path.stat().st_mtime > 86400):
            version = await get_dataset_version("players")
            url = f"{NFLVERSE_BASE_URL}/{version}/players.parquet"
            download_parquet(url, cache_path, "players")
            logger.info(f"Downloaded players data to {cache_path}")
        
        df = safe_read_parquet(cache_path, "players")
        if df.empty:
            raise ValueError("Empty DataFrame loaded from players.parquet")
        return df
    except Exception as e:
        logger.error(f"Failed to load players data: {str(e)}")
        raise RuntimeError(f"Failed to load players data: {str(e)}")

def load_schedules(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load game schedules for specified seasons."""
    if seasons is None:
        seasons = [2024]
    
    cache_path = CACHE_DIR / "schedules.parquet"
    
    # Download if not in cache or older than 1 day
    if not cache_path.exists() or (datetime.now().timestamp() - cache_path.stat().st_mtime > 86400):
        version = get_dataset_version("schedules")
        url = f"{NFLVERSE_BASE_URL}/{version}/schedules.parquet"
        download_parquet(url, cache_path)
    
    df = pd.read_parquet(cache_path)
    return df[df["season"].isin(seasons)] if seasons else df

async def load_injuries(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load injury reports from the condensed parquet file."""
    try:
        # Try loading from condensed file first
        condensed_path = CACHE_DIR / "injuries_condensed.parquet"
        if condensed_path.exists():
            logger.info(f"Loading injuries from condensed file: {condensed_path}")
            injuries_df = pd.read_parquet(condensed_path)
            
            # Filter by seasons if provided
            if seasons:
                injuries_df = injuries_df[injuries_df['season'].isin(seasons)]
                
            return injuries_df
        
        # Fall back to individual season files if condensed file not available
        logger.warning("Condensed injuries file not found, falling back to individual season files")
        if seasons is None:
            seasons = [2024]  # Use previous year since current might not be available
        
        dfs = []
        errors = []
        
        for season in seasons:
            try:
                cache_path = CACHE_DIR / f"injuries_{season}.parquet"
                
                # Download if not in cache or older than 1 hour (injuries update frequently)
                if not cache_path.exists() or (datetime.now().timestamp() - cache_path.stat().st_mtime > 3600):
                    version = await get_dataset_version("injuries")
                    url = f"{NFLVERSE_BASE_URL}/{version}/injuries_{season}.parquet"
                    download_parquet(url, cache_path, f"injuries_{season}")
                
                # Load from cache
                df = safe_read_parquet(cache_path, f"injuries_{season}")
                dfs.append(df)
            except Exception as e:
                errors.append(f"Season {season}: {str(e)}")
        
        if not dfs and errors:
            raise RuntimeError(f"Failed to load any seasons. Errors: {'; '.join(errors)}")
        
        return pd.concat(dfs) if dfs else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading injuries data: {str(e)}")
        return pd.DataFrame()

async def load_depth_charts(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load team depth charts from the condensed parquet file."""
    try:
        # Try loading from condensed file first
        condensed_path = CACHE_DIR / "depth_charts_condensed.parquet"
        if condensed_path.exists():
            logger.info(f"Loading depth charts from condensed file: {condensed_path}")
            depth_df = pd.read_parquet(condensed_path)
            
            # Filter by seasons if provided
            if seasons:
                depth_df = depth_df[depth_df['season'].isin(seasons)]
                
            return depth_df
            
        # Fall back to individual season files if condensed file not available
        logger.warning("Condensed depth charts file not found, falling back to individual season files")
        if seasons is None:
            seasons = [2024]  # Use previous year since current might not be available
        
        dfs = []
        errors = []
        
        for season in seasons:
            try:
                cache_path = CACHE_DIR / f"depth_charts_{season}.parquet"
                
                # Download if not in cache or older than 1 day
                if not cache_path.exists() or (datetime.now().timestamp() - cache_path.stat().st_mtime > 86400):
                    version = await get_dataset_version("depth_charts")
                    url = f"{NFLVERSE_BASE_URL}/{version}/depth_charts_{season}.parquet"
                    download_parquet(url, cache_path, f"depth_charts_{season}")
                
                # Load from cache
                df = safe_read_parquet(cache_path, f"depth_charts_{season}")
                dfs.append(df)
            except Exception as e:
                errors.append(f"Season {season}: {str(e)}")
        
        if not dfs and errors:
            raise RuntimeError(f"Failed to load any seasons. Errors: {'; '.join(errors)}")
        
        return pd.concat(dfs) if dfs else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading depth charts data: {str(e)}")
        return pd.DataFrame()

async def load_rosters(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load roster data from the condensed parquet file."""
    try:
        # Try loading from condensed file first
        condensed_path = CACHE_DIR / "rosters_condensed.parquet"
        if condensed_path.exists():
            logger.info(f"Loading rosters from condensed file: {condensed_path}")
            rosters_df = pd.read_parquet(condensed_path)
            
            # Filter by seasons if provided
            if seasons:
                rosters_df = rosters_df[rosters_df['season'].isin(seasons)]
                
            return rosters_df
            
        # Fall back to individual season files if condensed file not available
        logger.warning("Condensed rosters file not found, falling back to individual season files")
        if seasons is None:
            seasons = [2024]  # Use current season by default
        
        dfs = []
        errors = []
        
        for season in seasons:
            try:
                cache_path = CACHE_DIR / f"roster_{season}.parquet"
                
                # Download if not in cache
                if not cache_path.exists():
                    version = await get_dataset_version("rosters")
                    url = f"{NFLVERSE_BASE_URL}/{version}/roster_{season}.parquet"
                    download_parquet(url, cache_path)
                
                # Load from cache
                df = safe_read_parquet(cache_path)
                dfs.append(df)
            except Exception as e:
                errors.append(f"Season {season}: {str(e)}")
        
        if not dfs and errors:
            raise RuntimeError(f"Failed to load any seasons. Errors: {'; '.join(errors)}")
        
        return pd.concat(dfs) if dfs else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading rosters data: {str(e)}")
        return pd.DataFrame()

def get_available_seasons() -> List[int]:
    """Get list of available seasons in the dataset from condensed files."""
    try:
        # First check if we have condensed files and extract seasons from them
        condensed_files = [
            "weekly_stats_condensed.parquet",
            "play_by_play_condensed.parquet",
            "rosters_condensed.parquet",
            "depth_charts_condensed.parquet",
            "injuries_condensed.parquet"
        ]
        
        for file_name in condensed_files:
            file_path = CACHE_DIR / file_name
            if file_path.exists():
                try:
                    # Read the file and get unique seasons
                    df = pd.read_parquet(file_path)
                    if 'season' in df.columns:
                        seasons = sorted(df['season'].unique().tolist())
                        if seasons:
                            return seasons
                except Exception as e:
                    logger.warning(f"Failed to read seasons from {file_name}: {e}")
        
        # If no condensed files or couldn't extract seasons, check individual files
        seasons = []
        for file in CACHE_DIR.glob("play_by_play_*.parquet"):
            try:
                season = int(file.stem.split("_")[-1])
                # Verify file is readable
                if safe_read_parquet(file) is not None:
                    seasons.append(season)
            except (ValueError, RuntimeError):
                continue
        
        # If no cached files or all files are corrupt, return last 5 years
        if not seasons:
            current_year = datetime.now().year
            seasons = list(range(current_year - 4, current_year + 1))
        
        return sorted(seasons)
    except Exception as e:
        logger.error(f"Error getting available seasons: {e}")
        # Fallback to current year only
        return [datetime.now().year] 