"""Configuration settings for the NFL data API."""

import os
from pathlib import Path

# Determine if we're running on Railway
IS_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT"))

# Set cache directory based on environment
if IS_RAILWAY:
    # On Railway, use the persistent volume mount point
    CACHE_DIR = Path("/data/cache")
else:
    # Locally, use the user's cache directory
    import appdirs
    CACHE_DIR = Path(appdirs.user_cache_dir("nfl-data-api"))

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Constants
NFLVERSE_BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Dataset version mappings
DATASET_VERSIONS = {
    "play_by_play": "pbp",
    "player_stats": "player_stats",
    "players": "players",
    "rosters": "rosters",
    "injuries": "injuries",
    "depth_charts": "depth_charts",
    "schedules": "schedules",
    "snap_counts": "snap_counts",
    "nextgen_stats": "nextgen_stats",
    "pfr_advstats": "pfr_advstats"
} 