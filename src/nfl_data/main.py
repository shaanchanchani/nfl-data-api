"""FastAPI application for NFL data analysis."""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
import redis.asyncio as redis

# Import routers from endpoint modules
from .endpoints import player, team, game, compare, utility, qb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NaN and Infinity values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        return super().default(obj)

# Load environment variables
load_dotenv()

class CustomJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        try:
            return json.dumps(
                content,
                ensure_ascii=False,
                allow_nan=True,  # Allow NaN values to pass through
                indent=None,
                separators=(",", ":"),
                cls=CustomJSONEncoder,
            ).encode("utf-8")
        except Exception as e:
            logger.error(f"JSON serialization error: {e}")
            # Create a simplified response that will definitely serialize
            error_response = {"error": "Could not serialize response", "message": str(e)}
            return json.dumps(error_response).encode("utf-8")

app = FastAPI(
    title="NFL Data API",
    description="API for accessing and analyzing NFL data",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=os.getenv("ROOT_PATH", ""),
    default_response_class=CustomJSONResponse
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Initialize Redis cache on startup."""
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    FastAPICache.init(RedisBackend(redis_client), prefix="nfl-api-cache")

# Include routers from each endpoint module
app.include_router(player.router)
app.include_router(team.router)
app.include_router(game.router)
app.include_router(compare.router)
app.include_router(qb.router)
app.include_router(utility.router)