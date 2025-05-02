"""NFL Data API package.

Only expose the FastAPI application instance so ASGI servers (e.g. Uvicorn,
Gunicorn) can import ``nfl_data.app`` or ``nfl_data.main:app`` directly.  All
other helpers are now accessed through their respective sub-modules and no
longer re-exported here to keep the public surface minimal and avoid stale
symbols.
"""

# Public ASGI application -----------------------------------------------------------------------------------------
from .main import app  # noqa: F401

# Define the explicit re-export list so ``from nfl_data import *`` only exposes
# the FastAPI application.
__all__ = ["app"]
