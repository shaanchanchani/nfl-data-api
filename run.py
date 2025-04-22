"""Entry point for the NFL Data API."""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("nfl_data.main:app", host="0.0.0.0", port=port, reload=True) 