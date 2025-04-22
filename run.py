"""Entry point for the NFL Data API."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("nfl_data.main:app", host="0.0.0.0", port=8000, reload=True) 