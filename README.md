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
- **`/api/player/{name}`**: Basic player info and career stats
- **`/api/player/{name}/info`**: Detailed player information
- **`/api/player/{name}/stats`**: Comprehensive player statistics with filtering
- **`/api/player/{name}/history`**: Player roster, depth chart, and injury history
- **`/api/player/{name}/headshot`**: Player headshot URL
- **`/api/player/{name}/career`**: Career statistics
- **`/api/player/{name}/gamelog`**: Game-by-game statistics
- **`/api/player/{name}/situation/{situation_type}`**: Situation-specific stats
- **`/api/player/{name}/on-field`**: Stats when another player is on/off field
- **`/api/player/qb-stats`**: Advanced QB statistics
- **`/api/player/schedule-analysis`**: Upcoming schedule analysis
- **`/api/player/on-off-impact`**: Player performance with/without another player

### Team Endpoints
- **`/api/team/{team}`**: Comprehensive team statistics

### Game Endpoints
- **`/api/game`**: Game details
- **`/api/game/outlook`**: Game analysis and outlook

### Comparison Endpoints
- **`/api/compare`**: Multi-player comparison

### Utility Endpoints
- **`/`**: Welcome message
- **`/health`**: Health check
- **`/api`**: API information
- **`/api/seasons`**: Available seasons
- **`/api/cache/clear`**: Clear cache entries
- **`/api/cache/status`**: Cache status

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