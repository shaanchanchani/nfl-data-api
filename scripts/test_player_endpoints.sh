#!/usr/bin/env bash

###############################################################################
# test_player_endpoints.sh                                                    #
#                                                                             #
# Utility script to smoke-test the NFL-Data-API player endpoints locally.      #
#                                                                             #
# USAGE:                                                                      #
#   ./scripts/test_player_endpoints.sh [BASE_URL] [TEAM] [PLAYER_NAME] [YEAR] #
#                                                                             #
#   BASE_URL     – Root URL of the running FastAPI server (default:           #
#                  http://127.0.0.1:8000)                                     #
#   TEAM         – Current team abbreviation of the player (default: KC)      #
#   PLAYER_NAME  – Full player name with spaces (default: Patrick Mahomes)     #
#   YEAR         – Season year to query (default: 2023)                       #
#                                                                             #
# NOTE:                                                                       #
#   ‑ The script prints each request URL and its prettified JSON response.    #
#   ‑ Requires the `jq` utility for pretty printing JSON.                     #
###############################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# Parameter handling & helpers
# ---------------------------------------------------------------------------

BASE_URL="${1:-http://127.0.0.1:8000}"
TEAM="${2:-KC}"
PLAYER_NAME="${3:-Patrick Mahomes}"
SEASON="${4:-2023}"

# URL-encode helper for player names containing spaces/special chars.
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

print_request() {
  local description="$1"; shift
  local url="$1"; shift
  echo -e "\n🟢 $description"
  echo "   GET $url"
  echo "---------------------------------------------------------------------"
  curl -s "$url" | jq .
}

# ---------------------------------------------------------------------------
# Hit each endpoint
# ---------------------------------------------------------------------------

echo "==============================================================="
echo " NFL-Data-API – Local Player Endpoint Smoke Test"
echo " Base URL : $BASE_URL"
echo " Player   : $PLAYER_NAME ($TEAM)"
echo " Season   : $SEASON"
echo "==============================================================="

# 1) Basic player info
print_request "Player Info" \
  "$BASE_URL/api/player/$PLAYER_ENC/info?season=$SEASON"

# 2) Player statistics (team-prefixed route)
print_request "Player Stats" \
  "$BASE_URL/api/${TEAM,,}/$PLAYER_ENC/stats?aggregation=season&seasons=$SEASON&season_type=REG"

# 3) Headshot URL
print_request "Player Headshot" \
  "$BASE_URL/api/player/$PLAYER_ENC/headshot"

# 4) Game Log
print_request "Player Game Log" \
  "$BASE_URL/api/player/$PLAYER_ENC/gamelog?season=$SEASON"

# 5) Top players leaderboard (QB example)
print_request "Top Players Leaderboard (QB)" \
  "$BASE_URL/api/players/top?position=QB&n=5&seasons=$SEASON"

echo -e "\n✅  All requests completed." 