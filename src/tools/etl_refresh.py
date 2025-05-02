import asyncio
import pandas as pd
import os
from pathlib import Path
import requests
from datetime import datetime

# Config
CACHE_DIR = Path("cache")
NFLVERSE_BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
SEASONS = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_parquet(url: str, cache_path: Path) -> None:
    print(f"Downloading from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

async def process_dataset(name: str, seasons: list) -> pd.DataFrame:
    dfs = []
    for season in seasons:
        file_path = CACHE_DIR / f"{name}_{season}.parquet"
        url_path = 'pbp' if name == 'play_by_play' else name
        url = f"{NFLVERSE_BASE_URL}/{url_path}/{name}_{season}.parquet"
        try:
            download_parquet(url, file_path)
            df = pd.read_parquet(file_path)
            dfs.append(df)
            os.remove(file_path)  # Clean up individual season file
        except Exception as e:
            print(f"Error processing {name} for {season}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

async def main():
    datasets = {
        "play_by_play": [
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
            "interception", "sack","fumble", "fumble_lost", "touchdown", "pass_touchdown", "rush_touchdown",
            "return_touchdown", "safety", "field_goal_result", "extra_point_result", "two_point_conv_result",
            "first_down", "first_down_pass", "first_down_rush", "first_down_penalty", "penalty",
            "penalty_yards", "penalty_type", "penalty_team", "sp", "special_teams_play", "pass_attempt",
            "rush_attempt", "rushing_yards", "receiving_yards",
            "qb_dropback", "qb_scramble", "qb_spike", "qb_kneel", "punt_attempt",
            "kickoff_attempt", "field_goal_attempt", "extra_point_attempt", "two_point_attempt", "epa", "wpa",
            "air_epa", "yac_epa", "comp_air_epa", "comp_yac_epa", "air_yards", "yards_after_catch", "cp",
            "cpoe", "pass_oe", "qb_epa", "xyac_epa", "xyac_mean_yardage", "xyac_success", "series_success",
            "success", "drive", "fixed_drive_result", "aborted_play", "season_type"
        ]
    }

    for name, columns in datasets.items():
        print(f"\nProcessing {name}...")
        df = await process_dataset(name, SEASONS)
        if not df.empty:
            # Keep only essential columns that exist
            cols_to_keep = [col for col in columns if col in df.columns]
            df = df[cols_to_keep]
            df.to_parquet(CACHE_DIR / f"{name}_condensed.parquet", index=False)
            print(f"Saved {name} with {len(df)} rows")
    
    # Download and save players data
    players_url = f"{NFLVERSE_BASE_URL}/players/players.parquet"
    players_file = CACHE_DIR / "players.parquet"
    try:
        download_parquet(players_url, players_file)
        print("Saved players data")
    except Exception as e:
        print(f"Error downloading players data: {e}")

    

if __name__ == "__main__":
    asyncio.run(main()) 