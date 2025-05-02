"""Utility endpoints for the NFL Data API."""

import logging
import os
from typing import Dict, List, Optional
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime
import redis.asyncio as redis



from fastapi_cache import FastAPICache

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()



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
