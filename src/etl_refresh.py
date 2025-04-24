import asyncio
import pandas as pd
import os
import logging
from pathlib import Path
from nfl_data.data_loader import (
    load_players, load_weekly_stats, load_rosters,
    load_depth_charts, load_injuries, load_pbp_data
)

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEASONS = list(range(2020, 2025))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("etl_refresh")

def atomic_save(df, path):
    temp_path = path.with_suffix(".tmp")
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)
    logger.info(f"Saved {path.name} ({len(df)} rows)")

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
    logger.info("Loading play-by-play data...")
    pbp_df = await load_pbp_data(seasons=SEASONS)
    # TODO: Condense pbp_df to only the columns you want (see PLANS.md)
    # For now, save all columns
    atomic_save(pbp_df, CACHE_DIR / "play_by_play_condensed.parquet")

    # TODO: Add schedule loading if available
    # logger.info("Loading schedule data...")
    # schedule_df = await load_schedule(seasons=SEASONS)
    # atomic_save(schedule_df, CACHE_DIR / "schedule_condensed.parquet")

    logger.info("ETL refresh complete.")

if __name__ == "__main__":
    asyncio.run(main()) 