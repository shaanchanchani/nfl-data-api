# NFL Data API

## Overview
A FastAPI-based API for accessing and analyzing NFL data, with multi-layer caching and robust endpoint coverage.

## Repository Map

### Project Structure
```
nfl-data-api/
├── src/
│   ├── nfl_data/                          # Main package
│   │   ├── endpoints/                     # Modular endpoint files
│   │   │   ├── __init__.py                # Package exports
│   │   │   ├── player.py                  # All /api/player/* endpoints
│   │   │   ├── team.py                    # All /api/team/* endpoints
│   │   │   ├── game.py                    # All /api/game/* endpoints
│   │   │   ├── compare.py                 # Player comparison endpoints
│   │   │   ├── qb.py                      # QB-specific advanced endpoints
│   │   │   ├── utility.py                 # Utility and system endpoints
│   │   │   └── README.md                  # How to add new endpoints
│   │   ├── __init__.py                    # Package exports
│   │   ├── config.py                      # Configuration settings
│   │   ├── data_import.py                 # Data import wrapper functions
│   │   ├── data_loader.py                 # Core data loading utilities
│   │   ├── main.py                        # FastAPI app entrypoint
│   │   ├── pbp_schema.txt                 # Play-by-play data schema
│   │   └── stats_helpers.py               # Statistical calculation functions
│   └── etl_refresh.py                     # ETL script for refreshing data files
├── tests/                                 # Test suite
│   ├── test_main.py                       # API endpoints tests
│   ├── test_stats_helpers.py              # Unit tests for stats calculations
│   ├── test_integration.py                # Integration tests
│   └── test_api.py                        # API-specific tests
├── cache/                                 # Data cache directory
│   ├── play_by_play_condensed.parquet     # Condensed play-by-play data
│   ├── weekly_stats_condensed.parquet     # Condensed weekly stats data
│   ├── players_condensed.parquet          # Condensed player information
│   ├── rosters_condensed.parquet          # Condensed roster data
│   ├── depth_charts_condensed.parquet     # Condensed depth chart data
│   └── injuries_condensed.parquet         # Condensed injury data
├── examine_parquet.py                     # Tool to examine parquet file structure
├── situational-debug.py                   # Debug tool for situation stats
├── load_data.py                           # Simple data loading script
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Package configuration
├── uv.lock                                # Dependency lock file
├── Procfile                               # Deployment configuration
```

### Core Components

#### Data Pipeline
1. **ETL Refresh (`etl_refresh.py`)**: Consolidates data from nflverse into optimized parquet files
   - Runs as a scheduled cron job every 4 hours to keep data fresh
   - Can also be run manually for immediate updates
2. **Data Loader (`data_loader.py`)**: Loads data from cached parquet files or fetches from source
3. **Data Import (`data_import.py`)**: Thin wrapper around data_loader for backward compatibility

#### API Structure
1. **Main App (`main.py`)**: FastAPI application entrypoint with router registration
2. **Endpoint Modules (`endpoints/`)**:
   - Organized by resource/feature (player, team, game, etc.)
   - Each file defines a FastAPI `APIRouter` and related endpoints
3. **Stats Helpers (`stats_helpers.py`)**: Shared statistical calculation functions

#### Caching System
1. **API Response Cache**: Redis-based caching for endpoint responses
2. **GitHub API Cache**: Redis-based caching for nflverse data versioning
3. **Parquet File Cache**: Filesystem-based caching for data files

## Endpoints

### Player Endpoints
- **`/api/player/{name}`**: Basic player info (ID, position, team, age, experience) and summarized career stats.
- **`/api/player/{name}/info`**: Detailed player information. Returns a dictionary with keys: `player_id`, `name`, `position`, `team`, `age`, `experience`, `college`, `height`, `weight`, `headshot_url`.
- **`/api/player/{name}/stats`**: Comprehensive player statistics. Returns player/team info, aggregation level, filters applied, and a `stats` list. The content of the `stats` list dictionaries depends on aggregation/filters:
    - If `aggregate=week` (and no situations): Each dictionary contains all weekly stats columns for a game (e.g., `week`, `opponent_team`, `passing_yards`, `rushing_attempts`, `receiving_tds`, `fantasy_points_ppr`).
    - If situations requested: Stats calculated from PBP data, keys vary by position (e.g., `total_attempts`, `completions`, `passing_yards`, `total_rush_attempts`, `total_targets`, `plays`).
    - *Note: Season/Career aggregation from weekly data is not fully implemented.*
- **`/api/player/{name}/history`**: Player history data. Specify `?type=roster`, `?type=depth`, or `?type=injury`. Returns a list of records for the specified type (e.g., seasonal roster info, weekly depth chart position, weekly injury status with description).
- **`/api/player/{name}/headshot`**: Player information (ID, name, team, position) and the URL for their headshot image.
- **`/api/player/{name}/career`**: Aggregated career statistics for the player across all available seasons.
- **`/api/player/{name}/gamelog`**: Game-by-game statistics for a specific season. Returns player info (`player_id`, `player_name`, `team`, `position`, `season`) and a `games` list. Each dictionary in `games` contains all columns from the weekly stats data for that game (e.g., `week`, `opponent_team`, `completions`, `attempts`, `passing_yards`, `rushing_yards`, `receptions`, `targets`, `fantasy_points_ppr`).
- **`/api/player/{name}/situation?situations=...`**: Player statistics filtered for specific game situations (e.g., `red_zone`, `third_down`). Returns aggregated stats calculated from PBP data for those situations.
- **`/api/player/qb-stats`**: Advanced QB statistics (e.g., passer rating under pressure, completion percentage over expected). *Note: Implementation details may vary.*
- **`/api/player/schedule-analysis`**: Analysis of a player's upcoming schedule, potentially including opponent defensive rankings. *Note: Implementation details may vary.*
- **`/api/player/on-off-impact`**: Player's statistical performance when another specified player is on or off the field. *Note: Implementation details may vary.*

### Team Endpoints
- **`/api/team/{team}`**: Comprehensive team statistics including offensive, defensive, injury, and depth chart information for a specified season (defaults to most recent).

### Game Endpoints
- **`/api/game`**: Details for a specific game (e.g., score, key plays, stats). Requires game identification parameters. *Note: Implementation details may vary.*
- **`/api/game/outlook`**: Analysis and outlook for an upcoming game, potentially including key matchups and predictions. *Note: Implementation details may vary.*

### Comparison Endpoints
- **`/api/compare`**: Side-by-side statistical comparison of multiple players. *Note: Implementation details may vary.*

### Utility Endpoints
- **`/`**: Simple welcome message indicating the API is running.
- **`/health`**: Health check status (e.g., `{"status": "ok"}`).
- **`/api`**: General API information (e.g., version, documentation links).
- **`/api/seasons`**: List of available seasons for which data is present.
- **`/api/cache/clear`**: Confirmation message upon clearing cache entries. Requires authentication/permissions.
- **`/api/cache/status`**: Status of the caching system (e.g., connection status, memory usage).

## Caching Architecture

**1. API Response Cache (Redis via fastapi-cache2)**
- Caches endpoint responses
- 12-hour expiration for most endpoints
- Prefix: `nfl-api-cache:`

**2. GitHub API Cache (Redis)**
- Caches GitHub API responses for nflverse data versioning
- 1-hour expiration
- Prefix: `github-api:`

**3. Parquet File Cache (Filesystem)**
- Caches downloaded data files (play-by-play, rosters, etc.)
- Located in `CACHE_DIR` (default: `./cache/`)
- Expiration varies by data type

## Test Coverage

### Implemented Tests
- **Unit tests** for statistical calculations
- **API endpoint tests** for all major endpoints
- **Integration tests** for data pipeline and calculations

### Future Test Areas
- Resource-specific test modules
- Cache integration tests
- Performance and load tests

## Development

### Getting Started
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```
   export REDIS_URL=redis://localhost:6379
   export CACHE_DIR=./cache
   ```

3. Run the ETL refresh script to download data:
   ```
   python src/etl_refresh.py
   ```

4. Run the API server:
   ```
   uvicorn src.nfl_data.main:app --reload
   ```

5. Visit the API documentation:
   ```
   http://localhost:8000/docs
   ```

### Creating New Endpoints
See the guide in `src/nfl_data/endpoints/README.md` for detailed instructions on adding new endpoints.

### Design Principles
- **Modularity**: Related endpoints are grouped in the same file
- **Caching**: Consistent caching pattern across endpoints
- **Error Handling**: Consistent error handling approach
- **Testing**: Comprehensive test coverage

## Future Development
- Add more granular tests for each endpoint group
- Enhance OpenAPI documentation
- Split large endpoint files into more focused modules
- Implement missing endpoint functionality
- Further optimize data loading and caching

## Data Refresh Schedule
The data files are automatically updated via the `etl_refresh.py` script, which runs as a scheduled cron job:
- **Daily**: Player information, rosters, injuries
- **Weekly**: Play-by-play data, weekly stats 
- **Season start**: Team data, schedules

Manual refresh can be triggered by running:
```
python src/etl_refresh.py
```

## More Information
- For questions or contributions, open an issue or PR


