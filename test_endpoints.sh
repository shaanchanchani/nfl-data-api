#!/bin/bash

###############################################################################
# test_endpoints.sh                                                           #
#                                                                             #
# Utility script to smoke-test various NFL Data API endpoints locally.        #
#                                                                             #
# USAGE:                                                                      #
#   ./test_endpoints.sh [BASE_URL] [TEAM] [PLAYER_NAME] [YEAR]                #
#                                                                             #
#   BASE_URL     – Root URL of the running FastAPI server (default:           #
#                  http://127.0.0.1:8000)                                     #
#   TEAM         – Team abbreviation for team/player tests (default: KC)      #
#   PLAYER_NAME  – Full player name for player tests (default: Patrick Mahomes)#
#   YEAR         – Season year for relevant endpoints (default: 2023)         #
#                                                                             #
# NOTE:                                                                       #
#   ‑ Requires the `jq` utility for pretty printing JSON. Install via:        #
#     brew install jq | sudo apt-get install jq | etc.                        #
###############################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# Parameter handling & helpers
# ---------------------------------------------------------------------------
BASE_URL="${1:-http://localhost:8000}"
TEAM_ABBR="${2:-KC}"
PLAYER_NAME="${3:-Patrick Mahomes}"
SEASON="${4:-2023}"

LOG_FILE="test_run_$(date +%Y%m%d_%H%M%S).log"

# URL-encode helper
urlencode() {
  local str="$1"
  local out=""
  local i c hex
  for (( i=0; i<${#str}; i++ )); do
    c="${str:i:1}"
    case "$c" in
      [a-zA-Z0-9.~_-]) out+="$c" ;;
      *) printf -v hex '%%%02X' "'${c}'"; out+="$hex" ;;
    esac
  done
  echo "$out"
}

PLAYER_ENC="$(urlencode "$PLAYER_NAME")"

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "Warning: 'jq' command not found. JSON output will not be pretty-printed." >&2
    JQ_CMD="cat" # Use cat as a fallback if jq is missing
else
    JQ_CMD="jq ."
fi

# Initialize log file (unique name)
echo "Starting NFL Data API endpoint tests..." > "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"
echo "Base URL : $BASE_URL" >> "$LOG_FILE"
echo "Team     : $TEAM_ABBR" >> "$LOG_FILE"
echo "Player   : $PLAYER_NAME" >> "$LOG_FILE"
echo "Season   : $SEASON" >> "$LOG_FILE"
echo "Log File : $LOG_FILE" >> "$LOG_FILE"
echo "=======================================" >> "$LOG_FILE"

echo "Starting NFL Data API endpoint tests (logging to $LOG_FILE)..."
echo "Using Base URL: ${BASE_URL}"

# Function to make request and append to log file
make_request() {
  local description="$1"
  local endpoint_path="$2"
  local url="${BASE_URL}${endpoint_path}"

  echo -e "\n🧪 Testing: ${description}"
  echo "   GET ${url}"

  echo "--- Testing: ${description} --- [${url}]" >> "$LOG_FILE"
  echo "Response:" >> "$LOG_FILE"

  # Make the request, use JQ_CMD for formatting, handle errors, append to log
  if curl -sS --fail "${url}" | $JQ_CMD >> "$LOG_FILE"; then
      echo "   ✅ Success"
  else
      local curl_exit_code=$?
      echo "   ❌ Error (curl exit code: ${curl_exit_code}) fetching ${endpoint_path}. See $LOG_FILE for details." >&2
      # Log the error marker to the file too
      echo "*** ERROR FETCHING ABOVE URL (curl exit code: ${curl_exit_code}) ***" >> "$LOG_FILE"
  fi

  echo "----------------------------------------" >> "$LOG_FILE"
  sleep 0.5 # Small delay between requests
}

# --- Player Endpoints --- (Based on src/nfl_data/endpoints/player.py)
make_request "Player Stats (Default Season)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}"
make_request "Player Stats (Weekly Aggregation)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}&aggregation=week&week=5"
make_request "Player Stats (Career Aggregation)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?aggregation=career"
make_request "Player Stats (POST Season)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}&season_type=POST"
make_request "Player Stats (Redzone Only)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}&redzone_only=true"
make_request "Player Stats (Downs 3,4)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}&downs=3,4"
make_request "Player Stats (vs DEN)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}&opponent_team=DEN"
make_request "Player Stats (Score Diff -7 to 7)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}&score_differential_range=-7,7"
make_request "Player Stats (Combined: RZ & Down 3)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/stats?seasons=${SEASON}&redzone_only=true&downs=3"

make_request "Player Info (With Season)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/info?season=${SEASON}"
make_request "Player Info (Default Season)" "/api/$(echo "$TEAM_ABBR" | tr '[:upper:]' '[:lower:]')/${PLAYER_ENC}/info"

make_request "Top Players (QB Example - Default)" "/api/players/top?position=QB&n=5&seasons=${SEASON}"
make_request "Top Players (RB Example - Rushing Yards)" "/api/players/top?position=RB&n=5&seasons=${SEASON}"
make_request "Top Players (RB Example - Rushing TDs Asc)" "/api/players/top?position=RB&n=5&seasons=${SEASON}&sort_by=rushing_tds&ascending=true"
make_request "Top Players (RB Example - Min Carries)" "/api/players/top?position=RB&n=5&seasons=${SEASON}&min_threshold=carries:150"
make_request "Top Players (QB Career EPA)" "/api/players/top?position=QB&n=5&aggregation_type=career&sort_by=qb_epa"
make_request "Top Players (WR Redzone TDs)" "/api/players/top?position=WR&n=5&seasons=${SEASON}&sort_by=receiving_tds&redzone_only=true"

# --- Team Endpoints --- (Example - Assuming it exists elsewhere)
make_request "Team Info" "/api/team/${TEAM_ABBR}" # Check if this endpoint actually exists

# --- Utility Endpoints --- (Standard API health/info)
make_request "Root/Docs Redirect" "/"
make_request "Health Check" "/health"
# make_request "Clear Cache" "/api/cache/clear" # Keep commented out unless needed


echo "=======================================" >> "$LOG_FILE"
echo "Finished API endpoint tests at $(date)." >> "$LOG_FILE"
echo "Results saved in: ${LOG_FILE}" >> "$LOG_FILE"

# Also print final messages to stdout
echo "======================================="
echo "Finished API endpoint tests."
echo "Results saved in: ${LOG_FILE}" 