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
    cache_path = CACHE_DIR / "play_by_play_condensed.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(f"Condensed play-by-play file not found: {cache_path}")
    return pd.read_parquet(cache_path)

async def load_weekly_stats(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load weekly player stats for specified seasons."""
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

async def load_players() -> pd.DataFrame:
    """Load player information."""
    cache_path = CACHE_DIR / "players.parquet"
    
    try:
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
    """Load injury reports for specified seasons."""
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

async def load_depth_charts(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load team depth charts for specified seasons."""
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

async def load_rosters(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load roster data for specified seasons."""
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

def get_available_seasons() -> List[int]:
    """Get list of available seasons in the dataset."""
    try:
        # Check play-by-play files in cache
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
        # Fallback to current year only
        return [datetime.now().year] 