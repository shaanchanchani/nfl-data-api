"""Utility endpoints for the NFL Data API."""

import logging
import os
from typing import Dict, List, Optional
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime
import redis.asyncio as redis

from ..stats_helpers import (
    get_available_seasons
)

from fastapi_cache import FastAPICache

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/api/seasons")
async def get_seasons_endpoint():
    """Get list of available seasons in the dataset."""
    seasons = get_available_seasons()
    return {"seasons": seasons}

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/")
async def root():
    """Root endpoint redirects to API documentation."""
    return {
        "message": "Welcome to the NFL Data API", 
        "docs_url": "/docs",
        "warning": "Note: The first request may take 30-60 seconds as data is downloaded and cached. Subsequent requests will be much faster."
    }

@router.get("/api")
async def read_root():
    """Root endpoint providing basic API information."""
    return {
        "message": "Welcome to the NFL Data API!",
        "description": "An API for accessing NFL player and team statistics using nflfastR data.",
        "documentation": "/docs",
        "available_endpoints": [
            "/api/seasons",
            "/api/player/{name}",
            "/api/player/{name}/info",
            "/api/player/{name}/history",
            "/api/player/{name}/stats",
            "/api/player/{name}/headshot",
            "/api/player/{name}/career",
            "/api/player/{name}/gamelog",
            "/api/player/{name}/situation/{situation_type}",
            "/api/compare",
            "/api/team/{team}",
            "/api/game",
        ]
    }

@router.post("/api/cache/clear")
async def clear_cache(
    pattern: str = Query("*", description="Cache key pattern to clear")
):
    """Clear cache entries matching the pattern."""
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    keys = await redis_client.keys(f"nfl-api-cache:{pattern}")
    if keys:
        await redis_client.delete(*keys)
    return {"message": f"Cleared {len(keys)} cache entries"}

@router.get("/api/cache/status")
async def get_cache_status():
    """Get current cache status."""
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    
    # Get all cache keys
    keys = await redis_client.keys("nfl-api-cache:*")
    
    # Get cache stats
    stats = {
        "total_keys": len(keys),
        "keys": [key.decode('utf-8') for key in keys[:10]],  # Show first 10 keys
        "cache_enabled": FastAPICache.get_enable_status(),
        "backend_type": "Redis",
        "prefix": "nfl-api-cache"
    }
    
    return stats

@router.get("/api/debug/pbp-test")
async def test_pbp_data():
    """Diagnostic endpoint to test loading play-by-play data."""
    # This is a placeholder implementation
    return {"status": "Not implemented"}

@router.get("/api/debug/situation-test")
async def test_situation_extraction():
    """Test the situation extraction function."""
    try:
        # This is a placeholder implementation
        return {"status": "Not implemented"}
    except Exception as e:
        logger.error(f"Error testing situation extraction: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }