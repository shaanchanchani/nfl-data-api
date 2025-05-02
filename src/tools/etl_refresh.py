import asyncio
import pandas as pd
import os
from pathlib import Path
import requests
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

# Config
CACHE_DIR = Path("cache")
NFLVERSE_BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
# Limit seasons for testing/memory, adjust as needed
# SEASONS = [str(y) for y in range(2020, 2025)] # Example: Last 5 seasons
SEASONS = [str(y) for y in range(2000, 2025)] # Original full range

# Essential PBP columns (add any others absolutely needed by your stats functions)
pbp_essential_columns = [
    "play_id", "game_id", "season", "week", "posteam", "defteam", 
    "passer_player_id", "rusher_player_id", "receiver_player_id",
    "passer_player_name", "rusher_player_name", "receiver_player_name", # Added names
    "down", "ydstogo", "yardline_100", "qtr", "game_half", "game_seconds_remaining",
    "score_differential", "posteam_score", "defteam_score", 
    "play_type", "yards_gained", "complete_pass", "incomplete_pass", 
    "interception", "sack", "fumble", "fumble_lost", "touchdown", "pass_touchdown", 
    "rush_touchdown", "first_down", "first_down_pass", "first_down_rush", 
    "pass_attempt", "rush_attempt", "rushing_yards", "receiving_yards", 
    "qb_dropback", "qb_epa", "cpoe", "air_yards", "yards_after_catch", "epa",
    "season_type", "two_point_conv_result", "game_date", "home_team", "away_team"
]

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_parquet(url: str, cache_path: Path) -> None:
    print(f"Downloading from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Finished downloading {cache_path.name}")

async def process_and_append_pbp(season: str, writer: pq.ParquetWriter):
    name = "play_by_play"
    file_path = CACHE_DIR / f"{name}_{season}.parquet"
    url_path = 'pbp'
    url = f"{NFLVERSE_BASE_URL}/{url_path}/{name}_{season}.parquet"
    try:
        download_parquet(url, file_path)
        # Read only essential columns if possible (might need full read first)
        # For simplicity here, read full then filter
        df = pd.read_parquet(file_path)
        
        # Keep only essential columns that exist in this season's file
        cols_to_keep = [col for col in pbp_essential_columns if col in df.columns]
        df_filtered = df[cols_to_keep]
        
        # Convert DataFrame to Arrow Table
        table = pa.Table.from_pandas(df_filtered, preserve_index=False)
        
        # Write table to the Parquet file via the writer
        writer.write_table(table)
        print(f"Appended {name} for {season}, {len(df_filtered)} rows")
        os.remove(file_path)  # Clean up individual season file
    except Exception as e:
        print(f"Error processing {name} for {season}: {e}")
        if file_path.exists():
            os.remove(file_path) # Clean up even on error

async def main():
    # --- Process Play-by-Play Data --- 
    pbp_output_file = CACHE_DIR / "play_by_play_condensed.parquet"
    first_season = True
    schema = None
    writer = None

    print("\nProcessing play_by_play...")
    # Process seasons sequentially to manage memory
    for season in SEASONS:
        temp_file_path = CACHE_DIR / f"play_by_play_{season}.parquet"
        url_path = 'pbp'
        url = f"{NFLVERSE_BASE_URL}/{url_path}/play_by_play_{season}.parquet"
        try:
            download_parquet(url, temp_file_path)
            # Read schema from the first downloaded file
            if first_season:
                table = pq.read_table(temp_file_path, columns=pbp_essential_columns)
                schema = table.schema
                writer = pq.ParquetWriter(pbp_output_file, schema)
                writer.write_table(table)
                print(f"Wrote schema and first season ({season}), {len(table)} rows")
                first_season = False
            else:
                 # Read subsequent seasons and append
                 table = pq.read_table(temp_file_path, columns=pbp_essential_columns)
                 # Ensure schema matches if necessary (optional, might slow down)
                 # table = table.cast(schema)
                 writer.write_table(table)
                 print(f"Appended play_by_play for {season}, {len(table)} rows")
            
            os.remove(temp_file_path) # Clean up
        except Exception as e:
            print(f"Error processing play_by_play for {season}: {e}")
            if temp_file_path.exists():
                os.remove(temp_file_path) # Clean up even on error

    # Close the writer after processing all seasons
    if writer:
        writer.close()
        print(f"Finished writing {pbp_output_file}")
    else:
        print("No play-by-play data was processed.")

    # --- Download and save players data (remains the same) ---
    players_url = f"{NFLVERSE_BASE_URL}/players/players.parquet"
    players_file = CACHE_DIR / "players.parquet"
    try:
        print("\nDownloading players data...")
        download_parquet(players_url, players_file)
        print(f"Saved players data to {players_file}")
    except Exception as e:
        print(f"Error downloading players data: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 