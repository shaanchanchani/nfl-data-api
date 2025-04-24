# NFL Data API Endpoints

This directory contains modular FastAPI endpoint files organized by resource/feature.

## Adding a New Endpoint

To add a new endpoint:

1. Decide which existing file to add your endpoint to, based on what resource it operates on:
   - `player.py` - Player-related endpoints
   - `team.py` - Team-related endpoints
   - `game.py` - Game-related endpoints
   - etc.

2. If your endpoint represents an entirely new resource, create a new file:
   ```python
   """New resource endpoints for the NFL Data API."""

   import logging
   from typing import Dict, List, Optional, Any
   from fastapi import APIRouter, Query, Path, HTTPException
   from fastapi.responses import JSONResponse

   # Import necessary helpers
   from ..stats_helpers import (
       # Import relevant helper functions
   )

   from ..data_loader import (
       # Import relevant data loading functions
   )

   from fastapi_cache.decorator import cache

   # Set up logging
   logger = logging.getLogger(__name__)

   # Create router
   router = APIRouter()

   @router.get("/api/my-resource")
   @cache(expire=43200)  # Optional: 12 hours caching
   async def get_my_resource(
       param1: str = Query(..., description="Description of parameter"),
       param2: Optional[int] = Query(None, description="Optional parameter")
   ):
       """Endpoint description."""
       try:
           # Implementation
           return {"result": "data"}
       except Exception as e:
           logger.error(f"Error in get_my_resource: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
   ```

3. Add your new module to `__init__.py` if you created a new file:
   ```python
   from . import my_resource
   ```

4. Add your router to `main.py`:
   ```python
   from .endpoints import my_resource
   # ...
   app.include_router(my_resource.router)
   ```

5. Add tests for your new endpoint.

## Best Practices

1. **Group Related Endpoints:** Keep endpoints related to the same resource in the same file.

2. **Use Async:** Make your endpoint functions `async` to take advantage of FastAPI's async capabilities.

3. **Error Handling:** Wrap your implementation in a try/except block and use HTTPException for errors.

4. **Logging:** Log errors and important events using the `logger`.

5. **Caching:** Use the `@cache` decorator for endpoints that don't change frequently.

6. **Parameter Validation:** Use FastAPI's parameter validation and provide good descriptions.

7. **Documentation:** Add clear docstrings to all functions and endpoints.

8. **Type Hints:** Use proper type hints for all parameters and return values.

9. **Consistent Naming:** Follow the existing naming conventions for consistency.

10. **Testing:** Write tests for all new endpoints.