"""Entry point for the NFL Data API."""

import os
import uvicorn
# import asyncio # No longer needed if ETL isn't run here
# import sys # No longer needed if ETL isn't run here

# Remove ETL import
# try:
#    from src.tools.etl_refresh import main as etl_main
# except ImportError as e:
#    print(f"Error importing ETL script: {e}")
#    print("Please ensure src/tools/etl_refresh.py exists and is importable.")
#    etl_main = None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    # REMOVE or COMMENT OUT the ETL execution block
    # # --- Run ETL Refresh Before Starting Server ---
    # if etl_main:
    #    print("--- Running ETL Refresh ---")
    #    try:
    #        # Run the async main function from the ETL script
    #        asyncio.run(etl_main())
    #        print("--- ETL Refresh Completed Successfully ---")
    #    except Exception as e:
    #        print(f"--- ETL Refresh FAILED: {e} ---", file=sys.stderr)
    #        # Optionally exit if ETL failure is critical
    #        # sys.exit(1)
    # else:
    #    print("--- Skipping ETL Refresh (Import Failed) ---")

    # --- Start FastAPI Server ---
    print(f"--- Starting FastAPI server on port {port} ---")
    # Make sure reload=False for deployment to avoid unnecessary restarts
    uvicorn.run("src.nfl_data.main:app", host="0.0.0.0", port=port, reload=False) # Corrected path 