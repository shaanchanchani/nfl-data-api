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

    # --- Download and save players data with position correction ---
    players_url = f"{NFLVERSE_BASE_URL}/players/players.parquet"
    players_file = CACHE_DIR / "players.parquet"
    try:
        print("\nDownloading players data...")
        download_parquet(players_url, players_file)
        
        # Load, fix positions, and resave
        print("Processing player data to fix any position issues...")
        players_df = pd.read_parquet(players_file)
        
        # Print some diagnostics
        print(f"Total players: {len(players_df)}")
        print(f"NULL positions: {players_df['position'].isna().sum()}")
        
        # Fill null positions from position_group if available
        null_positions = players_df['position'].isna()
        if null_positions.any():
            print(f"Found {null_positions.sum()} players with NULL positions")
            # For players with null position but valid position_group, use position_group
            has_position_group = ~players_df['position_group'].isna()
            to_fix = null_positions & has_position_group
            if to_fix.any():
                players_df.loc[to_fix, 'position'] = players_df.loc[to_fix, 'position_group']
                print(f"Fixed {to_fix.sum()} NULL positions using position_group values")
        
        # Check for position mismatches - specifically QBs marked as 'P'
        position_corrections = {}
        
        # Check for known QBs incorrectly labeled
        qb_keywords = ["quarterback", "passer", "qb"]
        possible_qb_mask = players_df["display_name"].str.lower().apply(
            lambda name: any(keyword in name.lower() for keyword in qb_keywords)
        )
        qb_check_df = players_df[possible_qb_mask & (players_df["position"] == "P")]
        
        if len(qb_check_df) > 0:
            print(f"WARNING: Found {len(qb_check_df)} players with QB-like names but position = 'P'")
            print(f"Sample: {qb_check_df[['gsis_id', 'display_name', 'position', 'position_group']].head()}")
        
        # Apply corrections if needed
        known_qbs = [
            "00-0010346",  # Peyton Manning
            # Add more known QB IDs that need correction
        ]
        
        # Identify anomalies by checking if position doesn't match position_group
        anomalies = players_df[
            (players_df["position"] != players_df["position_group"]) & 
            (players_df["position_group"] == "QB") & 
            (players_df["position"] != "QB")
        ]
        if len(anomalies) > 0:
            print(f"Found {len(anomalies)} players with position_group='QB' but position != 'QB'")
            print(f"Sample: {anomalies[['gsis_id', 'display_name', 'position', 'position_group']].head()}")
            
            # Fix these anomalies by setting position=position_group
            players_df.loc[
                (players_df["position_group"] == "QB") & (players_df["position"] != "QB"),
                "position"
            ] = "QB"
        
        # Apply known corrections for specific players
        for qb_id in known_qbs:
            idx = players_df["gsis_id"] == qb_id
            if idx.any():
                old_pos = players_df.loc[idx, "position"].iloc[0]
                players_df.loc[idx, "position"] = "QB"
                print(f"Corrected position for {players_df.loc[idx, 'display_name'].iloc[0]} from {old_pos} to QB")
                
        # Double-check there are no remaining obvious problems with QBs
        remaining_issues = players_df[
            possible_qb_mask & 
            (players_df["position"] != "QB") & 
            ~players_df["position"].isna()
        ]
        if len(remaining_issues) > 0:
            print(f"WARNING: Still found {len(remaining_issues)} potential QB players with non-QB positions")
            print(f"Sample: {remaining_issues[['gsis_id', 'display_name', 'position', 'position_group']].head()}")
        
        # Save the corrected data
        players_df.to_parquet(players_file)
        print(f"Saved corrected players data to {players_file}")
    except Exception as e:
        print(f"Error processing players data: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 