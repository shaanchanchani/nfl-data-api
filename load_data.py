from src.nfl_data.data_loader import (
    load_players, load_weekly_stats, load_rosters,
    load_depth_charts, load_injuries
)

# Define seasons we want to load
SEASONS = list(range(2020, 2025))  # Last 5 seasons

print("Loading players data...")
players_df = load_players()
print(f"Loaded {len(players_df)} players")

print("\nLoading weekly stats...")
weekly_stats_df = load_weekly_stats(seasons=SEASONS)
print(f"Loaded weekly stats for {len(weekly_stats_df)} player-weeks")

print("\nLoading rosters...")
rosters_df = load_rosters(seasons=SEASONS)
print(f"Loaded {len(rosters_df)} roster entries")

print("\nLoading depth charts...")
depth_charts_df = load_depth_charts(seasons=SEASONS)
print(f"Loaded {len(depth_charts_df)} depth chart entries")

print("\nLoading injuries...")
injuries_df = load_injuries(seasons=SEASONS)
print(f"Loaded {len(injuries_df)} injury reports") 