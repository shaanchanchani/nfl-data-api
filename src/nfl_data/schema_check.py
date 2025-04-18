"""Utility to check schemas of NFL data parquet files."""

import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .data_loader import (
    CACHE_DIR,
    NFLVERSE_BASE_URL,
    get_dataset_version,
    load_pbp_data,
    load_weekly_stats,
    load_players,
    load_schedules,
    load_injuries,
    load_depth_charts,
    download_parquet,
    safe_read_parquet
)

def get_schema_info(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Get schema information for a DataFrame."""
    return {
        'dataset': dataset_name,
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict()
    }

def load_dataset(dataset_name: str, season: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """Load a dataset directly without using cached functions."""
    if season is None:
        season = datetime.now().year - 1  # Use last season's data
        
    if dataset_name == "play_by_play":
        cache_path = CACHE_DIR / f"play_by_play_{season}.parquet"
        if not cache_path.exists():
            version = "pbp"  # This is the actual release tag
            url = f"https://github.com/nflverse/nflverse-data/releases/download/{version}/play_by_play_{season}.parquet"
            download_parquet(url, cache_path)
        return safe_read_parquet(cache_path), "Play by Play"
        
    elif dataset_name == "weekly_stats":
        cache_path = CACHE_DIR / f"player_stats_{season}.parquet"
        if not cache_path.exists():
            version = "player_stats"  # This is the actual release tag
            url = f"https://github.com/nflverse/nflverse-data/releases/download/{version}/player_stats_{season}.parquet"
            download_parquet(url, cache_path)
        return safe_read_parquet(cache_path), "Weekly Player Stats"
        
    elif dataset_name == "players":
        cache_path = CACHE_DIR / "players.parquet"
        if not cache_path.exists():
            version = "players"  # This is the actual release tag
            url = f"https://github.com/nflverse/nflverse-data/releases/download/{version}/players.parquet"
            download_parquet(url, cache_path)
        return safe_read_parquet(cache_path), "Players"
        
    elif dataset_name == "rosters":
        cache_path = CACHE_DIR / f"roster_{season}.parquet"
        if not cache_path.exists():
            version = "rosters"  # This is the actual release tag
            url = f"https://github.com/nflverse/nflverse-data/releases/download/{version}/roster_{season}.parquet"
            download_parquet(url, cache_path)
        df = safe_read_parquet(cache_path)
        return df, "Rosters"
        
    elif dataset_name == "injuries":
        cache_path = CACHE_DIR / f"injuries_{season}.parquet"
        if not cache_path.exists():
            version = "injuries"  # This is the actual release tag
            url = f"https://github.com/nflverse/nflverse-data/releases/download/{version}/injuries_{season}.parquet"
            download_parquet(url, cache_path)
        df = safe_read_parquet(cache_path)
        return df, "Injuries"
        
    elif dataset_name == "depth_charts":
        cache_path = CACHE_DIR / f"depth_charts_{season}.parquet"
        if not cache_path.exists():
            version = "depth_charts"  # This is the actual release tag
            url = f"https://github.com/nflverse/nflverse-data/releases/download/{version}/depth_charts_{season}.parquet"
            download_parquet(url, cache_path)
        df = safe_read_parquet(cache_path)
        return df, "Depth Charts"
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def check_all_schemas(season: Optional[int] = None) -> Dict[str, Dict]:
    """Check schemas for all NFL data parquet files."""
    if season is None:
        season = datetime.now().year - 1  # Use last season's data
    
    schemas = {}
    datasets = [
        "play_by_play",
        "weekly_stats",
        "players",
        "rosters",
        "injuries", 
        "depth_charts"
    ]
    
    for dataset in datasets:
        try:
            df, dataset_name = load_dataset(dataset, season)
            schemas[dataset] = get_schema_info(df, dataset_name)
        except Exception as e:
            schemas[dataset] = {'error': str(e)}
    
    return schemas

def print_schema_report(schemas: Dict[str, Dict]) -> None:
    """Print a formatted report of schemas."""
    for dataset_name, info in schemas.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print('='*80)
        
        if 'error' in info:
            print(f"Error loading dataset: {info['error']}")
            continue
        
        print(f"\nNumber of columns: {info['num_columns']}")
        print("\nColumns:")
        for col in sorted(info['columns']):
            dtype = info['dtypes'][col]
            print(f"  - {col} ({dtype})")

def save_schema_report(schemas: Dict[str, Dict], output_file: str) -> None:
    """Save a formatted report of schemas to a file."""
    with open(output_file, 'w') as f:
        for dataset_name, info in schemas.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write('='*80 + "\n")
            
            if 'error' in info:
                f.write(f"Error loading dataset: {info['error']}\n")
                continue
            
            f.write(f"\nNumber of columns: {info['num_columns']}\n")
            f.write("\nColumns:\n")
            for col in sorted(info['columns']):
                dtype = info['dtypes'][col]
                f.write(f"  - {col} ({dtype})\n")

def main():
    """Main function to run schema check."""
    current_year = 2023  # Use 2023 season data for testing
    schemas = check_all_schemas(current_year)
    print_schema_report(schemas)
    
    # Save to file
    output_file = "nfl_data_schemas.txt"
    save_schema_report(schemas, output_file)
    print(f"\nSchema information saved to {output_file}")

if __name__ == '__main__':
    main() 