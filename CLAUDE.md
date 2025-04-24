Okay, let's draft a phased plan to refactor the /api/player/{name} endpoint, incorporating your goals for flexibility, efficiency, and future advanced stats capabilities.

Overarching Goal: Transition from a single, verbose endpoint to a modular set of endpoints that provide focused data efficiently, allowing for complex aggregations and splits in the future.

Core Principles:

Modularity: Break down functionality into logical, independent parts.

Efficiency: Minimize payload size and server processing for common requests.

Flexibility: Allow users to request exactly the data they need (sections, granularity, filters).

Scalability: Design endpoints and data processing to accommodate future advanced stats and situational splits.

Clarity: Make the API structure intuitive and well-documented.

Phase 1: Foundational Split & Normalization

Objective: Break the monolithic endpoint into core dedicated components and reduce immediate redundancy.

Steps:

Create /api/player/{name}/info Endpoint:

Action: Create a new route GET /api/player/{name}/info.

Data: Return only the content currently in the player_info block (ID, name, position, team, age, experience, college, physicals, headshot).

Logic: Load players_condensed.parquet, find the player, calculate age, return the relevant fields.

Create /api/player/{name}/history Endpoint:

Action: Create a new route GET /api/player/{name}/history.

Parameter: Add a required query parameter type: str = Query(..., enum=["roster", "depth", "injury"]).

Data Normalization:

Load the relevant dataset (rosters_condensed, depth_charts_condensed, injuries_condensed) based on type.

Filter by gsis_id.

Crucially: Modify the data returned for each record in the list. Only include fields specific to that history type and timestamp (e.g., for roster: season, team, status, jersey; for depth: season, week, depth_team, depth_position; for injury: season, week, injury details, status). Do NOT repeat basic player info like name, birthdate, height, weight here. Include gsis_id if needed for client-side joining, but ideally just the specific history event data.

Filtering (Optional but Recommended): Add optional query parameters like season: Optional[int] = None to filter history records.

Create Basic /api/player/{name}/stats Endpoint:

Action: Create a new route GET /api/player/{name}/stats.

Default Behavior: Return aggregated seasonal stats by default.

Logic:

Load weekly_stats_condensed.parquet.

Filter by player_id.

Group data by season and season_type (REG, POST).

Aggregate relevant stats (sum yards/TDs, completions/attempts, etc.). Use the existing career stats logic but apply it per season.

Handle NaN values during aggregation (e.g., use .sum(skipna=True), .mean(skipna=True)).

Sanitize the final aggregated records (convert NaN to None).

Data: Return a list of objects, each representing a season's aggregated stats. Include career_stats as a separate top-level key in the response for this endpoint.

Deprecate/Modify Original Endpoint (Optional):

Decide whether to keep the original /api/player/{name}.

Option A (Remove): Remove it entirely once Phases 1-2 are complete.

Option B (Modify): Change it to return only player_info and career_stats as a basic summary, directing users to other endpoints for details.

Option C (Keep with Include/Exclude): Implement basic include/exclude here only if absolutely necessary during transition, but aim to phase it out.

Phase 2: Enhancing Flexibility & Granularity

Objective: Add more control over the data returned by the new endpoints.

Steps:

Enhance /api/player/{name}/stats Granularity:

Action: Add query parameter aggregate: str = Query("season", enum=["career", "season", "week"]).

Logic:

If aggregate="career": Return only the career_stats block (calculated as before).

If aggregate="season": Return the default seasonal aggregation (from Phase 1).

If aggregate="week": Return the raw (but sanitized) weekly stats, similar to the original endpoint's season_stats block.

Filtering: Add more filtering options: season: Optional[int] = None, week: Optional[int] = None, season_type: Optional[str] = Query(None, enum=["REG", "POST"]). Ensure these filters work correctly with the chosen aggregation level.

Refine Data Loading/Caching:

Action: Centralize data loading functions (e.g., load_weekly_stats, load_rosters) used by multiple endpoints.

Action: Implement caching (e.g., using fastapi-cache) on the data loading functions to avoid repeated file reads.

Introduce Parameterization (If Needed):

Re-evaluate if include/exclude is still desired after the dedicated endpoints exist. It might be less necessary.

If yes, consider adding it to a summary endpoint (e.g., a modified original endpoint or a new /api/player/{name}/summary) that combines info and career_stats by default, allowing inclusion of other sections. However, lean towards using the dedicated endpoints.

Phase 3: Advanced Stats, Splits & Optimization

Objective: Build the infrastructure for complex queries and optimize performance.

Steps:

Design Advanced Stat Calculation:

Action: Define how advanced stats (EPA, WOPR, Air Yards Share, custom metrics) will be calculated – during data loading/preprocessing, or on-the-fly during aggregation? (On-the-fly is more flexible but computationally heavier).

Action: Integrate these calculations into the /stats endpoint logic, ensuring they work with different aggregation levels.

Implement Stat Splits:

Action: Define query parameters for desired splits (e.g., split_by: Optional[str] = Query(None, enum=["opponent", "home_away", "game_situation"]), opponent_abbr: Optional[str] = None, location: Optional[str] = Query(None, enum=["home", "away"])).

Action: Modify the aggregation logic in /stats to handle grouping by these split categories before calculating stats. This requires ensuring the necessary columns (opponent team, home/away status, etc.) are available in the weekly_stats data.

Complexity Note: This is the most complex part. Start with one split type (e.g., vs. opponent) and build iteratively.

Performance Optimization:

Action: Profile endpoint performance, especially /stats with aggregation and splits.

Action: Optimize Pandas operations (vectorization, avoiding loops where possible).

Action: Enhance caching strategies – cache results of specific aggregations/splits if they are frequently requested.

Comprehensive Documentation & Testing:

Action: Thoroughly update the OpenAPI (Swagger/ReDoc) documentation for all endpoints, parameters, and response schemas. Provide clear examples.

Action: Write unit and integration tests covering different parameter combinations, aggregation levels, splits, and edge cases (missing data, players with no stats).

Key Considerations Throughout:

Data Model (Pydantic): Define clear Pydantic models for request parameters and response bodies for each endpoint.

Error Handling: Ensure robust error handling (e.g., player not found, invalid parameter combinations).

Data Loading: Keep data loading efficient. Load only necessary columns from Parquet files when possible.

Asynchronous Operations: Use async def for routes and ensure any I/O-bound operations (like file reading, if not cached) are handled asynchronously if possible (though Pandas operations are often CPU-bound).

This phased plan breaks down the complex task into manageable steps, starting with the most critical structural changes and gradually adding flexibility and advanced features. Remember to commit changes frequently and test thoroughly at each stage.