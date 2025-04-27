#!/bin/bash

# --- Debug Script for Situational Stats Endpoint ---

# Base URL for the API
BASE_URL="https://nfl-data-api-dev.up.railway.app"
LOG_FILE="debug_output.log"

# Example player
PLAYER_NAME="Patrick Mahomes"
PLAYER_NAME_ENCODED="${PLAYER_NAME// /%20}"

# Initialize log file (overwrite if exists)
echo "Starting Situational Stats Debug Run..." > "$LOG_FILE"
echo "Base URL: ${BASE_URL}" >> "$LOG_FILE"
echo "Log File: ${LOG_FILE}" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Function to make request and append to log file
# Takes endpoint path (including query string) as argument
make_debug_request() {
  ENDPOINT_PATH=$1
  URL="${BASE_URL}${ENDPOINT_PATH}"

  echo "Testing URL: ${URL}"
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
      echo "Error fetching ${ENDPOINT_PATH}." >> "$LOG_FILE"
  fi

  echo "----------------------------------------" >> "$LOG_FILE"
  sleep 1 # Small delay between requests
}

# --- Test Cases ---

# 1. Single valid situation
echo "Test Case 1: Single valid situation (red_zone)" >> "$LOG_FILE"
make_debug_request "/api/player/${PLAYER_NAME_ENCODED}/situation?situations=red_zone"

# 2. Multiple valid situations
echo "Test Case 2: Multiple valid situations (third_down,fourth_down)" >> "$LOG_FILE"
make_debug_request "/api/player/${PLAYER_NAME_ENCODED}/situation?situations=third_down,fourth_down"

# 3. Mix of valid and invalid situations
echo "Test Case 3: Mix of valid and invalid (goal_line,fifth_down)" >> "$LOG_FILE"
make_debug_request "/api/player/${PLAYER_NAME_ENCODED}/situation?situations=goal_line,fifth_down"

# 4. Single invalid situation
echo "Test Case 4: Single invalid situation (first_and_ten)" >> "$LOG_FILE"
make_debug_request "/api/player/${PLAYER_NAME_ENCODED}/situation?situations=first_and_ten"

# 5. Empty situation list (should likely error or return default)
echo "Test Case 5: Empty situation list" >> "$LOG_FILE"
make_debug_request "/api/player/${PLAYER_NAME_ENCODED}/situation?situations="

# 6. No situations parameter (should error - parameter is required)
echo "Test Case 6: Missing situations parameter" >> "$LOG_FILE"
make_debug_request "/api/player/${PLAYER_NAME_ENCODED}/situation"

# --- End Test Cases ---

echo "========================================" >> "$LOG_FILE"
echo "Finished Situational Stats Debug Run." >> "$LOG_FILE"
echo "Results saved in: ${LOG_FILE}" >> "$LOG_FILE"

# Also print final messages to stdout
echo "========================================"
echo "Finished Situational Stats Debug Run."
echo "Results saved in: ${LOG_FILE}" 