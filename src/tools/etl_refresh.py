import asyncio
import pandas as pd
import os
import logging
import sys
from pathlib import Path
from nfl_data.data_loader import (
    load_players, load_weekly_stats, load_rosters,
    load_depth_charts, load_injuries, load_pbp_data
)
import requests
import time
from datetime import datetime
import traceback

# Import configuration from nfl_data module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nfl_data.config import CACHE_DIR, NFLVERSE_BASE_URL, MAX_RETRIES, RETRY_DELAY, IS_RAILWAY

# Ensure the cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Get current NFL season - use last 5 seasons including current
current_year = datetime.now().year
SEASONS = list(range(current_year - 4, current_year + 1))

# Configure logging with more detail for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("etl_refresh")

# Log environment details
logger.info(f"ETL refresh starting with environment:")
logger.info(f"  Railway: {IS_RAILWAY}")
logger.info(f"  Cache directory: {CACHE_DIR}")
logger.info(f"  Seasons to process: {SEASONS}")
logger.info(f"  Current directory: {os.getcwd()}")

def atomic_save(df, path):
    temp_path = path.with_suffix(".tmp")
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)
    logger.info(f"Saved {path.name} ({len(df)} rows)")

def download_parquet(url: str, cache_path: Path, dataset_name: str = "") -> None:
    temp_path = cache_path.with_suffix('.tmp')
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Downloading {dataset_name} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            temp_path.rename(cache_path)
            print(f"Successfully downloaded {dataset_name} to {cache_path}")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {dataset_name}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Failed to download {dataset_name} from {url}: {str(e)}")
            time.sleep(RETRY_DELAY)
        finally:
            if temp_path.exists():
                temp_path.unlink()

async def main():
    errors = []
    success = []

    try:
        # Players
        logger.info("Loading players data...")
        players_df = await load_players(force_rebuild=True)
        if not players_df.empty:
            atomic_save(players_df, CACHE_DIR / "players_condensed.parquet")
            success.append("players")
            logger.info(f"Successfully processed players with {len(players_df)} rows")
        else:
            error_msg = "Failed to load players data (empty DataFrame)"
            logger.error(error_msg)
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing players: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        logger.error(traceback.format_exc())

    try:
        # Weekly stats
        logger.info("Loading weekly stats...")
        weekly_stats_df = await load_weekly_stats(seasons=SEASONS, force_rebuild=True)
        if not weekly_stats_df.empty:
            atomic_save(weekly_stats_df, CACHE_DIR / "weekly_stats_condensed.parquet")
            success.append("weekly_stats")
            logger.info(f"Successfully processed weekly stats with {len(weekly_stats_df)} rows")
        else:
            error_msg = "Failed to load weekly stats (empty DataFrame)"
            logger.error(error_msg)
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing weekly stats: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        logger.error(traceback.format_exc())

    try:
        # Rosters
        logger.info("Loading rosters...")
        rosters_df = await load_rosters(seasons=SEASONS, force_rebuild=True)
        if not rosters_df.empty:
            atomic_save(rosters_df, CACHE_DIR / "rosters_condensed.parquet")
            success.append("rosters")
            logger.info(f"Successfully processed rosters with {len(rosters_df)} rows")
        else:
            error_msg = "Failed to load rosters (empty DataFrame)"
            logger.error(error_msg)
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing rosters: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        logger.error(traceback.format_exc())

    try:
        # Depth charts
        logger.info("Loading depth charts...")
        depth_charts_df = await load_depth_charts(seasons=SEASONS, force_rebuild=True)
        if not depth_charts_df.empty:
            atomic_save(depth_charts_df, CACHE_DIR / "depth_charts_condensed.parquet")
            success.append("depth_charts")
            logger.info(f"Successfully processed depth charts with {len(depth_charts_df)} rows")
        else:
            error_msg = "Failed to load depth charts (empty DataFrame)"
            logger.error(error_msg)
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing depth charts: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        logger.error(traceback.format_exc())

    try:
        # Injuries
        logger.info("Loading injuries...")
        injuries_df = await load_injuries(seasons=SEASONS, force_rebuild=True)
        if not injuries_df.empty:
            atomic_save(injuries_df, CACHE_DIR / "injuries_condensed.parquet")
            success.append("injuries")
            logger.info(f"Successfully processed injuries with {len(injuries_df)} rows")
        else:
            error_msg = "Failed to load injuries (empty DataFrame)"
            logger.error(error_msg)
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing injuries: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        logger.error(traceback.format_exc())

    try:
        # Play by play (condensed)
        logger.info("Loading play-by-play data across all seasons...")
        
        # Try loading the existing condensed file first (for debugging purposes)
        try:
            existing_pbp = load_pbp_data()
            if not existing_pbp.empty:
                logger.info(f"Found existing play_by_play_condensed.parquet with {len(existing_pbp)} rows")
                logger.info(f"Will rebuild anyway to ensure data integrity")
        except Exception as e:
            logger.info(f"No existing play-by-play condensed file found or it's corrupted: {str(e)}")
        
        # Always rebuild play-by-play from raw files
        pbp_dfs = []
        download_issues = []
        
        for season in SEASONS:
            file_path = CACHE_DIR / f"play_by_play_{season}.parquet"
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Always try to download the latest data (Railway has plenty of space)
            version = "pbp"  # Use the default version tag for pbp
            url = f"{NFLVERSE_BASE_URL}/{version}/play_by_play_{season}.parquet"
            
            try:
                logger.info(f"Downloading play-by-play data for season {season} from {url}")
                download_parquet(url, file_path, f"play_by_play_{season}")
                logger.info(f"Successfully downloaded play-by-play data for season {season}")
            except Exception as e:
                error_msg = f"Failed to download {file_path}: {str(e)}"
                logger.warning(error_msg)
                download_issues.append(error_msg)
                
                # If download failed but file exists, try to use the existing file
                if file_path.exists():
                    logger.info(f"Using existing file: {file_path}")
                else:
                    logger.warning(f"Cannot find data for season {season}, skipping")
                    continue
            
            # Try to read the file
            try:
                season_df = pd.read_parquet(file_path)
                if not season_df.empty:
                    logger.info(f"Loaded play_by_play_{season}.parquet with {len(season_df)} rows")
                    pbp_dfs.append(season_df)
                else:
                    error_msg = f"Empty DataFrame in {file_path}, skipping."
                    logger.warning(error_msg)
                    download_issues.append(error_msg)
            except Exception as e:
                error_msg = f"Failed to read {file_path}: {str(e)}"
                logger.warning(error_msg)
                download_issues.append(error_msg)
                
                # If file is corrupted, remove it
                try:
                    file_path.unlink()
                    logger.info(f"Removed corrupted file: {file_path}")
                except Exception:
                    pass
        
        if pbp_dfs:
            logger.info(f"Concatenating {len(pbp_dfs)} play-by-play DataFrames")
            pbp_df = pd.concat(pbp_dfs, ignore_index=True)
            logger.info(f"Combined play-by-play data has {len(pbp_df)} rows")
            
            # Columns to keep (from PLANS.md, adjust as needed)
            columns_to_keep = [
                "play_id", "game_id", "old_game_id", "season", "week", "posteam", "defteam", "side_of_field",
                "passer_player_id", "rusher_player_id", "receiver_player_id", "interception_player_id",
                "sack_player_id", "pass_defense_1_player_id", "pass_defense_2_player_id",
                "fumble_forced_player_id", "fumble_recovery_player_id", "solo_tackle_player_id",
                "assist_tackle_player_id", "tackle_for_loss_player_id", "qb_hit_player_id", "kicker_player_id",
                "punter_player_id", "kickoff_returner_player_id", "punt_returner_player_id", "td_player_id",
                "down", "ydstogo", "yardline_100", "qtr", "game_half", "game_seconds_remaining",
                "quarter_seconds_remaining", "half_seconds_remaining", "time", "score_differential",
                "posteam_score", "defteam_score", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
                "goal_to_go", "shotgun", "no_huddle", "posteam_type", "play_type", "pass_length",
                "pass_location", "run_gap", "run_location", "yards_gained", "complete_pass", "incomplete_pass",
                "interception", "sack", "fumble_lost", "touchdown", "pass_touchdown", "rush_touchdown",
                "return_touchdown", "safety", "field_goal_result", "extra_point_result", "two_point_conv_result",
                "first_down", "first_down_pass", "first_down_rush", "first_down_penalty", "penalty",
                "penalty_yards", "penalty_type", "penalty_team", "sp", "special_teams_play", "pass_attempt",
                "rush_attempt", "rushing_yards",
                "qb_dropback", "qb_scramble", "qb_spike", "qb_kneel", "punt_attempt",
                "kickoff_attempt", "field_goal_attempt", "extra_point_attempt", "two_point_attempt", "epa", "wpa",
                "air_epa", "yac_epa", "comp_air_epa", "comp_yac_epa", "air_yards", "yards_after_catch", "cp",
                "cpoe", "pass_oe", "qb_epa", "xyac_epa", "xyac_mean_yardage", "xyac_success", "series_success",
                "success", "drive", "fixed_drive_result",
                "receiving_yards",
            ]
            # Only keep columns that exist in the DataFrame
            existing_columns = set(pbp_df.columns)
            columns_to_keep = [col for col in columns_to_keep if col in existing_columns]
            logger.info(f"Keeping {len(columns_to_keep)} columns out of {len(existing_columns)} total columns")
            
            # Special handling for Railway: if we're missing critical columns, log them
            critical_columns = ["passer_player_id", "rusher_player_id", "receiver_player_id", 
                               "season", "week", "down", "yardline_100"]
            missing_critical = [col for col in critical_columns if col not in existing_columns]
            if missing_critical:
                logger.warning(f"Missing critical columns in PBP data: {missing_critical}")
            
            pbp_condensed = pbp_df[columns_to_keep]
            
            # Log file sizes to diagnose Railway disk space issues
            try:
                logger.info(f"Original PBP data memory usage: {pbp_df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
                logger.info(f"Condensed PBP data memory usage: {pbp_condensed.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            except Exception as e:
                logger.warning(f"Could not calculate memory usage: {str(e)}")
            
            logger.info(f"Saving condensed play-by-play data with {len(pbp_condensed)} rows and {len(columns_to_keep)} columns")
            
            # Save to multiple locations to ensure it's available
            condensed_path = CACHE_DIR / "play_by_play_condensed.parquet"
            try:
                atomic_save(pbp_condensed, condensed_path)
                logger.info(f"Successfully saved condensed PBP data to {condensed_path}")
                success.append("play_by_play")
            except Exception as e:
                error_msg = f"Failed to save to {condensed_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
            
            # For Railway, also try an alternative path if the main one failed
            if IS_RAILWAY:
                railway_paths = [
                    Path("/data/cache/play_by_play_condensed.parquet"),
                    Path("/app/cache/play_by_play_condensed.parquet")
                ]
                
                for alt_path in railway_paths:
                    if alt_path != condensed_path:
                        try:
                            alt_path.parent.mkdir(parents=True, exist_ok=True)
                            atomic_save(pbp_condensed, alt_path)
                            logger.info(f"Also saved condensed PBP data to alternative path: {alt_path}")
                        except Exception as e:
                            logger.warning(f"Could not save to alternative path {alt_path}: {str(e)}")
            
            # Verify the save worked by trying to read it back
            try:
                verification = pd.read_parquet(condensed_path)
                logger.info(f"Verification: Successfully read back {len(verification)} rows from condensed file")
                if len(verification) != len(pbp_condensed):
                    logger.warning(f"Row count mismatch after save: expected {len(pbp_condensed)}, got {len(verification)}")
            except Exception as e:
                error_msg = f"Failed to verify condensed file: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
            
            # Clean up per-season files to save space (optional on Railway)
            if IS_RAILWAY:
                logger.info("Running on Railway - keeping individual season files for redundancy")
            else:
                for season in SEASONS:
                    file_path = CACHE_DIR / f"play_by_play_{season}.parquet"
                    if file_path.exists():
                        try:
                            file_path.unlink()
                            logger.info(f"Deleted {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not delete {file_path}: {e}")
        else:
            error_msg = "No play-by-play files found for any season. Condensed file not created."
            logger.error(error_msg)
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"Error processing play-by-play data: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        logger.error(traceback.format_exc())

    # TODO: Add schedule loading if available
    # logger.info("Loading schedule data...")
    # schedule_df = await load_schedule(seasons=SEASONS)
    # atomic_save(schedule_df, CACHE_DIR / "schedule_condensed.parquet")

    # Print overall summary
    logger.info("=" * 50)
    logger.info("ETL REFRESH SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Successfully processed: {', '.join(success) if success else 'None'}")
    
    if errors:
        logger.error(f"There were {len(errors)} errors during ETL refresh:")
        for i, error in enumerate(errors, 1):
            logger.error(f"{i}. {error}")
        
        if IS_RAILWAY:
            # On Railway, even if there are errors, we should try to make the service work with what we have
            logger.info("Continuing with available data despite errors (Railway deployment)")
    else:
        logger.info("ETL refresh completed successfully with no errors")
    
    # Check for available data files now
    try:
        available_files = list(CACHE_DIR.glob("*_condensed.parquet"))
        logger.info(f"Available condensed data files after ETL: {[f.name for f in available_files]}")
        
        # Check if play-by-play data is available (most important)
        pbp_available = any(f.name == "play_by_play_condensed.parquet" for f in available_files)
        if pbp_available:
            logger.info("Play-by-play data is available - API should function properly")
        else:
            logger.error("Play-by-play data is NOT available - API functionality will be limited")
    except Exception as e:
        logger.error(f"Error checking available files: {str(e)}")
    
    logger.info("=" * 50)
    logger.info("ETL refresh complete.")

if __name__ == "__main__":
    asyncio.run(main()) 