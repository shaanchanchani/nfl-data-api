"""Functions for importing NFL data."""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from .data_loader import (
    load_pbp_data,
    load_weekly_stats,
    load_players,
    load_schedules,
    load_injuries,
    load_depth_charts,
    get_available_seasons
)

def import_pbp_data() -> pd.DataFrame:
    """Import play-by-play data from the condensed file."""
    return load_pbp_data()

async def import_weekly_data(seasons: List[int]) -> pd.DataFrame:
    """Import weekly player stats for specified seasons."""
    return await load_weekly_stats(seasons)

async def import_players() -> pd.DataFrame:
    """Import player information."""
    return await load_players()

def import_schedules(seasons: List[int]) -> pd.DataFrame:
    """Import game schedules for specified seasons."""
    return load_schedules(seasons)

async def import_injuries(seasons: List[int]) -> pd.DataFrame:
    """Import injury reports for specified seasons."""
    return await load_injuries(seasons)

async def import_depth_charts(seasons: List[int]) -> pd.DataFrame:
    """Import team depth charts for specified seasons."""
    return await load_depth_charts(seasons) 