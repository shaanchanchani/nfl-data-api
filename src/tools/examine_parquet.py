import pandas as pd
import os

cache_dir = "/Users/shaanchanchani/dev/nfl-data-api/cache"
parquet_files = [
    "players_condensed.parquet",
    "weekly_stats_condensed.parquet",
    "rosters_condensed.parquet",
    "depth_charts_condensed.parquet",
    "injuries_condensed.parquet"
]

for file in parquet_files:
    file_path = os.path.join(cache_dir, file)
    print(f"\n{'='*80}")
    print(f"EXAMINING {file}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\nSample data (first 2 rows):")
        print(df.head(2).to_string())
        
    except Exception as e:
        print(f"Error examining {file}: {e}")
    
    print(f"\n{'='*80}")