
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the play-by-play data
pbp_path = "./cache/play_by_play_condensed.parquet"
logger.info(f"Loading PBP data from {pbp_path}")
pbp = pd.read_parquet(pbp_path)

# Print columns and sample data
logger.info(f"PBP data shape: {pbp.shape}")
logger.info(f"PBP columns: {pbp.columns.tolist()}")

# Check for Patrick Mahomes
mahomes_id = "00-0033873"
player_cols = [
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "lateral_receiver_player_id", "lateral_rusher_player_id",
    "fumbled_1_player_id", "fumbled_2_player_id"
]

# Check each column
for col in player_cols:
    if col in pbp.columns:
        count = (pbp[col] == mahomes_id).sum()
        logger.info(f"Mahomes found in {col}: {count} times")

# Check red zone plays
if "yardline_100" in pbp.columns:
    rz_plays = pbp[pbp["yardline_100"] <= 20]
    logger.info(f"Total red zone plays: {len(rz_plays)}")
    
    # Check Mahomes red zone plays
    mahomes_mask = False
    for col in player_cols:
        if col in pbp.columns:
            mahomes_mask = mahomes_mask | (rz_plays[col] == mahomes_id)
    
    mahomes_rz = rz_plays[mahomes_mask]
    logger.info(f"Mahomes red zone plays: {len(mahomes_rz)}")
    
    # Check 2023 season
    mahomes_rz_2023 = mahomes_rz[mahomes_rz["season"] == 2023] if "season" in mahomes_rz.columns else pd.DataFrame()
    logger.info(f"Mahomes 2023 red zone plays: {len(mahomes_rz_2023)}")
    
    # Sample of stats
    if not mahomes_rz_2023.empty and "passer_player_id" in mahomes_rz_2023.columns:
        passer_plays = mahomes_rz_2023[mahomes_rz_2023["passer_player_id"] == mahomes_id]
        logger.info(f"Mahomes as passer in red zone 2023: {len(passer_plays)} plays")
        
        if "complete_pass" in passer_plays.columns:
            completions = (passer_plays["complete_pass"] == 1).sum()
            logger.info(f"Completions in red zone: {completions}")
        
        if "pass_touchdown" in passer_plays.columns:
            tds = passer_plays["pass_touchdown"].sum()
            logger.info(f"Passing TDs in red zone: {tds}")

