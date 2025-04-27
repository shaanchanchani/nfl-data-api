#!/bin/bash

# Base URL for the API
BASE_URL="https://nfl-data-api-production.up.railway.app"
LOG_FILE="test_run.log"

# Initialize log file (overwrite if exists)
echo "Starting NFL Data API endpoint tests..." > "$LOG_FILE"
echo "Base URL: ${BASE_URL}" >> "$LOG_FILE"
echo "Log File: ${LOG_FILE}" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Example player and team
PLAYER_NAME="Patrick Mahomes"
TEAM_ABBR="KC"
SITUATION="3rd_down"

# Function to make request and append to log file
# Takes endpoint path as argument
make_request() {
  ENDPOINT=$1
  URL="${BASE_URL}${ENDPOINT}"

  echo "Testing endpoint: ${ENDPOINT}"
  echo "URL: ${URL}" >> "$LOG_FILE"
  echo "Response:" >> "$LOG_FILE"

  # Make the request, pretty-print if jq exists, and append to log file
  if command -v jq &> /dev/null; then
    curl -sS "${URL}" | jq '.' >> "$LOG_FILE"
  else
    curl -sS "${URL}" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE" # Add a newline for readability if jq is not present
  fi

  # Check if curl command was successful (optional, useful for debugging)
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
      echo "Error fetching ${ENDPOINT}." >> "$LOG_FILE"
  fi

  echo "----------------------------------------" >> "$LOG_FILE"
  sleep 1 # Small delay between requests
}

# --- Player Endpoints ---
# Note: URL encoding might be needed for names with spaces/special chars in real use,
# but curl often handles simple spaces. Using "Patrick Mahomes" as is for this script.
make_request "/api/player/${PLAYER_NAME// /%20}"
make_request "/api/player/${PLAYER_NAME// /%20}/info"
make_request "/api/player/${PLAYER_NAME// /%20}/stats" # Add query params like ?season=2023 manually if needed
make_request "/api/player/${PLAYER_NAME// /%20}/history"
make_request "/api/player/${PLAYER_NAME// /%20}/headshot"
make_request "/api/player/${PLAYER_NAME// /%20}/career"
make_request "/api/player/${PLAYER_NAME// /%20}/gamelog" # Add query params like ?season=2023 manually if needed
make_request "/api/player/${PLAYER_NAME// /%20}/situation?situations=red_zone,third_down" # Example with multiple situations
# Updated situation endpoint to use query parameters
# make_request "/api/player/${PLAYER_NAME// /%20}/situation?situations=${SITUATION}" # Example with single situation from variable
# The following endpoints might require specific query parameters not included here:
# make_request "/api/player/${PLAYER_NAME// /%20}/on-field" # Needs ?other_player_id=...
# make_request "/api/player/qb-stats" # Needs query params like ?season=2023
# make_request "/api/player/schedule-analysis" # Needs ?player_name=... or ?team=...
# make_request "/api/player/on-off-impact" # Needs query params like ?player1=...&player2=...

# --- Team Endpoints ---
make_request "/api/team/${TEAM_ABBR}"

# --- Game Endpoints ---
# These likely require query parameters (e.g., ?game_id=...)
# make_request "/api/game"
# make_request "/api/game/outlook"

# --- Comparison Endpoints ---
# Requires query parameters (e.g., ?player_ids=...)
# make_request "/api/compare"

# --- Utility Endpoints ---
make_request "/"
make_request "/health"
make_request "/api"
make_request "/api/seasons"
# make_request "/api/cache/clear" # Commented out - potentially disruptive
make_request "/api/cache/status"


echo "========================================" >> "$LOG_FILE"
echo "Finished API endpoint tests." >> "$LOG_FILE"
echo "Results saved in: ${LOG_FILE}" >> "$LOG_FILE"
echo "Note: Some endpoints requiring specific query parameters were skipped or are commented out." >> "$LOG_FILE"
echo "Note: Endpoints like /api/cache/clear are commented out by default." >> "$LOG_FILE"

# Also print final messages to stdout
echo "========================================"
echo "Finished API endpoint tests."
echo "Results saved in: ${LOG_FILE}"
echo "Note: Some endpoints requiring specific query parameters were skipped or are commented out."
echo "Note: Endpoints like /api/cache/clear are commented out by default." 