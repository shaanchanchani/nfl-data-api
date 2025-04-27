import asyncio
import pandas as pd
import os
import logging
from pathlib import Path
from nfl_data.data_loader import (
    load_players, load_weekly_stats, load_rosters,
    load_depth_charts, load_injuries, load_pbp_data
)
import requests
import time

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEASONS = list(range(2020, 2025))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("etl_refresh")

NFLVERSE_BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

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
    # Players
    logger.info("Loading players data...")
    players_df = await load_players()
    atomic_save(players_df, CACHE_DIR / "players_condensed.parquet")

    # Weekly stats
    logger.info("Loading weekly stats...")
    weekly_stats_df = await load_weekly_stats(seasons=SEASONS)
    atomic_save(weekly_stats_df, CACHE_DIR / "weekly_stats_condensed.parquet")

    # Rosters
    logger.info("Loading rosters...")
    rosters_df = await load_rosters(seasons=SEASONS)
    atomic_save(rosters_df, CACHE_DIR / "rosters_condensed.parquet")

    # Depth charts
    logger.info("Loading depth charts...")
    depth_charts_df = await load_depth_charts(seasons=SEASONS)
    atomic_save(depth_charts_df, CACHE_DIR / "depth_charts_condensed.parquet")

    # Injuries
    logger.info("Loading injuries...")
    injuries_df = await load_injuries(seasons=SEASONS)
    atomic_save(injuries_df, CACHE_DIR / "injuries_condensed.parquet")

    # Play by play (condensed)
    logger.info("Loading play-by-play data across all seasons...")
    pbp_dfs = []
    for season in SEASONS:
        file_path = CACHE_DIR / f"play_by_play_{season}.parquet"
        if not file_path.exists():
            version = "pbp"  # Use the default version tag for pbp
            url = f"{NFLVERSE_BASE_URL}/{version}/play_by_play_{season}.parquet"
            try:
                download_parquet(url, file_path, f"play_by_play_{season}")
            except Exception as e:
                logger.warning(f"Failed to download {file_path}: {e}")
        if file_path.exists():
            pbp_dfs.append(pd.read_parquet(file_path))
        else:
            logger.warning(f"{file_path} not found, skipping.")
    if pbp_dfs:
        pbp_df = pd.concat(pbp_dfs, ignore_index=True)
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
        columns_to_keep = [col for col in columns_to_keep if col in pbp_df.columns]
        pbp_condensed = pbp_df[columns_to_keep]
        atomic_save(pbp_condensed, CACHE_DIR / "play_by_play_condensed.parquet")
        # Clean up per-season files to save space
        for season in SEASONS:
            file_path = CACHE_DIR / f"play_by_play_{season}.parquet"
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
    else:
        logger.warning("No play-by-play files found for any season. Condensed file not created.")

    # TODO: Add schedule loading if available
    # logger.info("Loading schedule data...")
    # schedule_df = await load_schedule(seasons=SEASONS)
    # atomic_save(schedule_df, CACHE_DIR / "schedule_condensed.parquet")

    logger.info("ETL refresh complete.")

if __name__ == "__main__":
    asyncio.run(main()) 