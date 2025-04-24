# NFL Data API

## Overview
A FastAPI-based API for accessing and analyzing NFL data, with multi-layer caching and robust endpoint coverage.

---

## Caching Architecture

**1. API Response Cache (Redis via fastapi-cache2)**
- Caches endpoint responses (e.g., /api/player/{name}, /api/players/top/{position})
- 12-hour expiration for most endpoints
- Prefix: `nfl-api-cache:`

**2. GitHub API Cache (Redis)**
- Caches GitHub API responses for nflverse data versioning
- 1-hour expiration
- Prefix: `github-api:`

**3. Parquet File Cache (Filesystem)**
- Caches downloaded data files (play-by-play, rosters, etc.)
- Located in `CACHE_DIR` (e.g., `/data/cache` on Railway)
- Expiration varies by data type

---

## Key Endpoints

### Implemented
- `/` : Welcome message, docs URL
- `/health` : Health check
- `/api` : API info and available endpoints
- `/api/seasons` : List available seasons
- `/api/players/top/{position}` : Top players by position (cached)
- `/api/player/{name}` : Player info, stats, roster, injuries (cached, LLM-friendly ambiguity handling)
- `/api/player/{name}/headshot` : Player headshot URL (cached)
- `/api/player/{name}/career` : Player career stats (cached)
- `/api/player/{name}/gamelog` : Player game-by-game stats
- `/api/player/{name}/situation/{situation_type}` : Player stats for specific situations
- `/api/compare` : Compare multiple players
- `/api/player/on-off-impact` : Analyze player performance with/without another player
- `/api/player/qb-stats` : Advanced QB stats
- `/api/player/schedule-analysis` : Analyze upcoming schedule for a player
- `/api/player/{name}/on-field` : Player performance with another player on/off field
- `/api/team/{team}` : Team stats
- `/api/game` : Game details
- `/api/game/outlook` : Game outlook/analysis
- `/api/cache/clear` : Clear cache entries (admin)
- `/api/cache/status` : Cache status/monitoring

### Stubs / Planned (see PLANS.md)
- `/api/cache/stats` : Cache statistics (hit/miss, size, health)
- `/api/cache/cleanup` : Selective cache cleanup by type/age
- Batch cache operations, cache prewarming, and more (see PLANS.md)

---

## Test Coverage

### Implemented
- **Unit tests** for:
  - Player stats (QB, RB, WR, TE)
  - Team stats
  - Defensive stats
  - Player/game resolution
  - Situation filters
  - Error handling (invalid names, params, etc.)
- **API endpoint tests** for:
  - All major endpoints (see above)
  - Parameterized tests for filters, situations, and edge cases
- **Integration tests** for:
  - Stats calculation (cross-checking play-by-play and weekly data)
  - Multi-season and partial-season logic
  - Defensive and situational stats

### Stubs / Yet to Implement
- Cache integration tests (planned)
- Cache persistence/restart tests (planned)
- Load/performance tests (planned)
- Monitoring/alerting tests (planned)
- Any new endpoints from PLANS.md

---

## Quickstart
1. Install dependencies: `pip install -r requirements.txt`
2. Set up Redis and configure `REDIS_URL`
3. Run the API: `uvicorn src.nfl_data.main:app --reload`
4. Visit `/docs` for interactive API documentation

---

## More
- See `PLANS.md` for roadmap, cache details, and future work.
- For questions or contributions, open an issue or PR.
