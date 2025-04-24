# NFL Data API Caching Implementation Plan

## Phase 1: Cache Infrastructure Setup

### 1.1 Railway Volume Configuration
- [ ] Configure 8GB container storage on Railway Hobby plan
- [ ] Verify `/data/cache` mount point is working
  ```bash
  # Verification steps:
  - Check mount exists: df -h /data/cache
  - Test write access: touch /data/cache/test.txt
  - Verify persistence across restarts
  ```
- [ ] Verify proper permissions on cache directory
  ```bash
  # Verification steps:
  - Check permissions: ls -la /data/cache
  - Test application user access
  - Verify log writing permissions
  ```

### 1.2 Cache Directory Structure
- [ ] Create and verify cache subdirectories:
```bash
# Directory setup
mkdir -p /data/cache/{parquet,sqlite,memory_dump,logs}

# Verification
tree /data/cache
```

## Phase 1.5: Analytics-Ready Database Requirements

### Database Structure
1. play_by_play_condensed (Source: play_by_play)
Purpose: Core table for detailed play analysis, situational filtering, player involvement, and advanced metrics.
Grain: One row per play.
Primary Keys (Conceptual): game_id, play_id
Key Columns to Keep:
Identifiers: play_id, game_id, old_game_id (if needed for joins), season, week, posteam, defteam, side_of_field.
Player Involvement (Standardized to gsis_id):
passer_player_id, rusher_player_id, receiver_player_id
interception_player_id
sack_player_id (primary - combine sack_player_id and half_sack_1/2_player_id logic during ETL)
pass_defense_1_player_id, pass_defense_2_player_id (for matchup analysis)
fumble_forced_player_id (combine forced_fumble_player_1_player_id etc.)
fumble_recovery_player_id (combine fumble_recovery_1_player_id etc.)
solo_tackle_player_id (combine solo_tackle_1_player_id etc.)
assist_tackle_player_id (combine assist_tackle_1_player_id etc.)
tackle_for_loss_player_id (combine tackle_for_loss_1_player_id etc.)
qb_hit_player_id (combine qb_hit_1_player_id etc.)
kicker_player_id, punter_player_id
kickoff_returner_player_id, punt_returner_player_id
td_player_id
Situational Context: down, ydstogo, yardline_100, qtr, game_half, game_seconds_remaining, quarter_seconds_remaining, half_seconds_remaining, time (game clock), score_differential, posteam_score, defteam_score, posteam_timeouts_remaining, defteam_timeouts_remaining, goal_to_go, shotgun, no_huddle, posteam_type (Home/Away).
Play Description: play_type (Pass/Rush/Punt/etc.), pass_length, pass_location, run_gap, run_location.
Play Outcome: yards_gained, complete_pass, incomplete_pass, interception, sack, fumble_lost, touchdown, pass_touchdown, rush_touchdown, return_touchdown, safety, field_goal_result, extra_point_result, two_point_conv_result, first_down, first_down_pass, first_down_rush, first_down_penalty, penalty, penalty_yards, penalty_type, penalty_team, sp (scoring play flag), special_teams_play.
Play Classification Flags: pass_attempt, rush_attempt, qb_dropback, qb_scramble, qb_spike, qb_kneel, punt_attempt, kickoff_attempt, field_goal_attempt, extra_point_attempt, two_point_attempt.
Advanced Metrics: epa, wpa, air_epa, yac_epa, comp_air_epa, comp_yac_epa, air_yards, yards_after_catch, cp, cpoe, pass_oe, qb_epa, xyac_epa, xyac_mean_yardage, xyac_success, series_success, success.
Drive Context: drive, fixed_drive_result.

## Phase 2: Data Caching Implementation

### 2.1 Parquet File Caching
- [ ] Implement versioned Parquet file storage
- [ ] Add data validation on cache read/write
- [ ] Set up automatic cache cleanup for old versions
```python
def cache_parquet_data(df, dataset_name, season):
    """Cache DataFrame as Parquet with version control."""
    pass
```

### 2.2 In-Memory LRU Cache
- [ ] Configure size-appropriate LRU caches:
  - Player info: 2000 entries
  - Season stats: 500 entries
  - Historical analysis: 100 entries
- [ ] Add cache persistence for restarts
- [ ] Implement cache warming on startup

### 2.3 Request Caching
- [ ] Set up SQLite request cache
- [ ] Configure reasonable expiration times
- [ ] Implement selective caching based on endpoints

## Phase 3: Cache Monitoring & Management

### 3.1 Monitoring Implementation
- [ ] Create cache statistics endpoint
- [ ] Add cache size monitoring
- [ ] Implement cache hit/miss tracking
- [ ] Set up alert thresholds

### 3.2 Cache Management Tools
- [ ] Add cache cleanup endpoints
- [ ] Implement selective cache invalidation
- [ ] Create cache prewarming utilities

## Phase 4: Optimization & Testing

### 4.1 Performance Optimization
- [ ] Profile cache access patterns
- [ ] Optimize cache key generation
- [ ] Implement batch cache operations
- [ ] Add cache compression where beneficial

### 4.2 Testing
- [ ] Create cache integration tests
- [ ] Test cache persistence across restarts
- [ ] Verify cache size limits
- [ ] Load test with realistic data

## Phase 5: Documentation & Monitoring

### 5.1 Documentation
- [ ] Document cache architecture
- [ ] Create cache operation playbook
- [ ] Add cache-related API documentation
- [ ] Document recovery procedures

### 5.2 Monitoring Setup
- [ ] Set up cache metrics dashboard
- [ ] Configure alerting
- [ ] Create cache health checks
- [ ] Document monitoring procedures

## Implementation Details

### Cache Size Limits
```python
# Memory Cache Limits (keep memory usage under 512MB)
PLAYER_INFO_CACHE_SIZE = 1000    # Small objects
SEASON_STATS_CACHE_SIZE = 250    # Medium objects
HISTORICAL_STATS_CACHE_SIZE = 50  # Large objects

# Storage Limits for 8GB Hobby Plan
MAX_PARQUET_CACHE_SIZE = 6 * 1024 * 1024 * 1024  # 6GB for NFL data
MAX_SQLITE_CACHE_SIZE = 1 * 1024 * 1024 * 1024   # 1GB for request cache
MAX_MEMORY_DUMP_SIZE = 512 * 1024 * 1024         # 512MB for LRU persistence
RESERVED_SPACE = 512 * 1024 * 1024               # 512MB reserved for logs and system

# Cleanup thresholds (trigger cleanup when reaching these limits)
PARQUET_CLEANUP_THRESHOLD = 5.5 * 1024 * 1024 * 1024  # 5.5GB
SQLITE_CLEANUP_THRESHOLD = 900 * 1024 * 1024          # 900MB
```

### Cache Directory Structure
```
/data/cache/
  ├── parquet/
  │   ├── play_by_play/
  │   ├── weekly_stats/
  │   ├── rosters/
  │   └── metadata.json
  ├── sqlite/
  │   └── http_cache.db
  ├── memory_dump/
  │   └── lru_cache.pkl
  └── logs/
      └── cache_operations.log
```

### Monitoring Endpoints
```python
@app.get("/api/cache/stats")
async def cache_stats():
    return {
        "usage": get_cache_stats(),
        "hit_rate": get_cache_hit_rate(),
        "health": check_cache_health()
    }
```

### Cache Cleanup
```python
@app.post("/api/cache/cleanup")
async def cleanup_cache(
    cache_type: str = Query(..., enum=["parquet", "sqlite", "memory", "all"]),
    older_than_days: Optional[int] = None
):
    """Clean up old cache entries."""
    pass
```

## Success Metrics

1. **Performance**
   - Cache hit rate > 90%
   - Average response time < 100ms
   - Cache miss penalty < 500ms

2. **Reliability**
   - Cache persistence across restarts
   - No memory leaks
   - Graceful degradation under load

3. **Efficiency**
   - Cache size within limits
   - Optimal compression ratios
   - Minimal duplicate data

## Next Steps

1. Begin with Phase 1 implementation
2. Set up basic monitoring
3. Implement core caching functionality
4. Add management tools
5. Deploy and test
6. Optimize based on real usage patterns 