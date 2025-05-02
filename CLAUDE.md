NFL Data API
 0.1.0 
OAS 3.1
/openapi.json
API for accessing and analyzing NFL data

default


GET
/api/{team}/{name}/info
Get Player Info


GET
/api/player/{name}/stats
Get Player Stats Endpoint


GET
/api/player/{name}/headshot
Get Headshot


GET
/api/player/{name}/gamelog
Get Gamelog Endpoint


GET
/api/players/top
Get Top Players Endpoint

Return a leaderboard of the top N players for a given position/metric.

Parameters
Cancel
Name	Description
position
string
(query)
Player position to analyse

QB
n
integer
(query)
maximum: 100
minimum: 1
Number of players to return

10
sort_by
(query)
Column to sort by. If omitted a sensible default for the position is used.

sort_by
min_threshold
(query)
Comma-sep key:value pairs e.g. 'carries:100' to filter players

'carries:100'
ascending
boolean
(query)
Sort ascending instead of descending


false
seasons
(query)
Comma-separated list of seasons

2015,2016,2017,2020,2021,2022,2023,2024
week
(query)
Filter by week (only when aggregation_type='week')

week
season_type
string
(query)
Season type filter (REG, POST, or REG+POST)

REG
redzone_only
boolean
(query)
If true, only include red-zone plays


false
aggregation_type
string
(query)
Aggregation type for underlying calculation


season
include_player_details
boolean
(query)
Whether to include detailed player info columns


true
downs
(query)
Comma-separated list of downs to filter by

downs
opponent_team
(query)
Filter by opponent team

opponent_team
score_differential_range
(query)
Score diff range e.g. '-10,10'

score_differential_range
Execute
Clear
Responses
Curl

curl -X 'GET' \
  'http://0.0.0.0:8080/api/players/top?position=QB&n=10&min_threshold=%27carries%3A100%27&ascending=false&seasons=2015%2C2016%2C2017%2C2020%2C2021%2C2022%2C2023%2C2024&season_type=REG&redzone_only=false&aggregation_type=season&include_player_details=true' \
  -H 'accept: application/json'
Request URL
http://0.0.0.0:8080/api/players/top?position=QB&n=10&min_threshold=%27carries%3A100%27&ascending=false&seasons=2015%2C2016%2C2017%2C2020%2C2021%2C2022%2C2023%2C2024&season_type=REG&redzone_only=false&aggregation_type=season&include_player_details=true
Server response
Code	Details
200	
Response body
Download
{
  "position": "QB",
  "leaderboard": [
    {
      "season": 2017,
      "player_id": "00-0031079",
      "team": "CHI",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 38,
      "passing_tds": 1,
      "passing_interceptions": 0,
      "qb_epa": 6.29359512636438,
      "passing_cpoe": 32.90131092071533,
      "epa_per_dropback": 6.29359512636438,
      "passing_air_yards": 7,
      "passing_yards_after_catch": 31,
      "pacr": 5.428571428571429,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 5.52,
      "season_type": "REG",
      "player_name": "Pat O'Donnell",
      "player_first_name": "Patrick",
      "player_last_name": "O'Donnell",
      "player_position": "P",
      "player_position_group": "SPEC",
      "player_college_name": "Miami",
      "player_height": 76,
      "player_weight": 212,
      "player_birth_date": "1991-02-22",
      "player_draft_club": "CHI",
      "player_draft_number": 191,
      "player_team_abbr": "SF",
      "player_headshot": "https://static.www.nfl.com/image/upload/f_auto,q_auto/league/kputu02pppmvb69bkdfg"
    },
    {
      "season": 2016,
      "player_id": "00-0030140",
      "team": "WAS",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 31,
      "passing_tds": 0,
      "passing_interceptions": 0,
      "qb_epa": 5.162058001733385,
      "passing_cpoe": 76.59696042537689,
      "epa_per_dropback": 5.162058001733385,
      "passing_air_yards": 29,
      "passing_yards_after_catch": 2,
      "pacr": 1.0689655172413792,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 1.24,
      "season_type": "REG",
      "player_name": "Tress Way",
      "player_first_name": "Tress",
      "player_last_name": "Way",
      "player_position": "P",
      "player_position_group": "SPEC",
      "player_college_name": "Oklahoma",
      "player_height": 73,
      "player_weight": 220,
      "player_birth_date": "1990-04-18",
      "player_draft_club": null,
      "player_draft_number": null,
      "player_team_abbr": "WAS",
      "player_headshot": "https://static.www.nfl.com/image/upload/f_auto,q_auto/league/n5nwpxqj5ex2pc2c2phx"
    },
    {
      "season": 2021,
      "player_id": "00-0035544",
      "team": "DET",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 75,
      "passing_tds": 1,
      "passing_interceptions": 0,
      "qb_epa": 5.069598037283868,
      "passing_cpoe": 50.08513629436493,
      "epa_per_dropback": 5.069598037283868,
      "passing_air_yards": 24,
      "passing_yards_after_catch": 51,
      "pacr": 3.125,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 7,
      "season_type": "REG",
      "player_name": "Tom Kennedy",
      "player_first_name": "Thomas",
      "player_last_name": "Kennedy",
      "player_position": "WR",
      "player_position_group": "WR",
      "player_college_name": "Bryant",
      "player_height": 70,
      "player_weight": 194,
      "player_birth_date": "1996-07-29",
      "player_draft_club": null,
      "player_draft_number": null,
      "player_team_abbr": "DET",
      "player_headshot": "https://static.www.nfl.com/image/upload/f_auto,q_auto/league/i4s3x6uwpru9p2czwode"
    },
    {
      "season": 2016,
      "player_id": "00-0027145",
      "team": "IND",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 35,
      "passing_tds": 0,
      "passing_interceptions": 0,
      "qb_epa": 5.0364618439925835,
      "passing_cpoe": 53.741514682769775,
      "epa_per_dropback": 5.0364618439925835,
      "passing_air_yards": 17,
      "passing_yards_after_catch": 18,
      "pacr": 2.0588235294117645,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 1.4000000000000001,
      "season_type": "REG",
      "player_name": "Pat McAfee",
      "player_first_name": "Pat",
      "player_last_name": "McAfee",
      "player_position": "P",
      "player_position_group": "SPEC",
      "player_college_name": "West Virginia",
      "player_height": 73,
      "player_weight": 233,
      "player_birth_date": "1987-05-02",
      "player_draft_club": "IND",
      "player_draft_number": 222,
      "player_team_abbr": "IND",
      "player_headshot": "https://static.www.nfl.com/image/private/f_auto,q_auto/league/giz54xrdfqrdu9nezfee"
    },
    {
      "season": 2017,
      "player_id": "00-0028954",
      "team": "PIT",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 44,
      "passing_tds": 0,
      "passing_interceptions": 0,
      "qb_epa": 4.970928155875299,
      "passing_cpoe": 67.89935827255249,
      "epa_per_dropback": 4.970928155875299,
      "passing_air_yards": 23,
      "passing_yards_after_catch": 21,
      "pacr": 1.9130434782608696,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 1.76,
      "season_type": "REG",
      "player_name": "Robert Golden",
      "player_first_name": "Robert",
      "player_last_name": "Golden",
      "player_position": "SS",
      "player_position_group": "DB",
      "player_college_name": "Arizona",
      "player_height": 71,
      "player_weight": 202,
      "player_birth_date": "1990-09-13",
      "player_draft_club": null,
      "player_draft_number": null,
      "player_team_abbr": "KC",
      "player_headshot": "https://static.www.nfl.com/image/private/f_auto,q_auto/league/x34qsxf2ita6cddyhped"
    },
    {
      "season": 2016,
      "player_id": "00-0030663",
      "team": "NO",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 50,
      "passing_tds": 1,
      "passing_interceptions": 0,
      "qb_epa": 4.872772316914052,
      "passing_cpoe": 61.50420606136322,
      "epa_per_dropback": 4.872772316914052,
      "passing_air_yards": 22,
      "passing_yards_after_catch": 28,
      "pacr": 2.272727272727273,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 6,
      "season_type": "REG",
      "player_name": "Willie Snead",
      "player_first_name": "Willie",
      "player_last_name": "Snead",
      "player_position": "WR",
      "player_position_group": "WR",
      "player_college_name": "Ball State",
      "player_height": 71,
      "player_weight": 205,
      "player_birth_date": "1992-10-17",
      "player_draft_club": null,
      "player_draft_number": null,
      "player_team_abbr": "MIA",
      "player_headshot": "https://static.www.nfl.com/image/private/f_auto,q_auto/league/fr5xckiwhguuaishnbju"
    },
    {
      "season": 2017,
      "player_id": "00-0024417",
      "team": "BAL",
      "position": "QB",
      "games_played": 2,
      "attempts": 2,
      "completions": 2,
      "completion_pct": 100,
      "passing_yards": 38,
      "passing_tds": 0,
      "passing_interceptions": 0,
      "qb_epa": 9.42508688842645,
      "passing_cpoe": 57.31978118419647,
      "epa_per_dropback": 4.712543444213225,
      "passing_air_yards": 32,
      "passing_yards_after_catch": 6,
      "pacr": 1.1875,
      "passing_first_downs": 2,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 2,
      "fantasy_points": 1.52,
      "season_type": "REG",
      "player_name": "Sam Koch",
      "player_first_name": "Sam",
      "player_last_name": "Koch",
      "player_position": "P",
      "player_position_group": "SPEC",
      "player_college_name": "Nebraska",
      "player_height": 73,
      "player_weight": 222,
      "player_birth_date": "1982-08-13",
      "player_draft_club": "BAL",
      "player_draft_number": 203,
      "player_team_abbr": "BAL",
      "player_headshot": "https://static.www.nfl.com/image/private/f_auto,q_auto/league/dw34lcf6qit1i1knjhpv"
    },
    {
      "season": 2017,
      "player_id": "00-0029632",
      "team": "ATL",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 51,
      "passing_tds": 1,
      "passing_interceptions": 0,
      "qb_epa": 4.45806786720641,
      "passing_cpoe": 75.77771842479706,
      "epa_per_dropback": 4.45806786720641,
      "passing_air_yards": 51,
      "passing_yards_after_catch": 0,
      "pacr": 1,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 6.04,
      "season_type": "REG",
      "player_name": "Mohamed Sanu",
      "player_first_name": "Mohamed",
      "player_last_name": "Sanu",
      "player_position": "WR",
      "player_position_group": "WR",
      "player_college_name": "Rutgers",
      "player_height": 74,
      "player_weight": 215,
      "player_birth_date": "1989-08-22",
      "player_draft_club": "CIN",
      "player_draft_number": 83,
      "player_team_abbr": "MIA",
      "player_headshot": "https://static.www.nfl.com/image/private/f_auto,q_auto/league/a6loroqhta6g65aqnsrk"
    },
    {
      "season": 2024,
      "player_id": "00-0035190",
      "team": "LV",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 34,
      "passing_tds": 0,
      "passing_interceptions": 0,
      "qb_epa": 4.333918029908091,
      "passing_cpoe": 30.53959012031555,
      "epa_per_dropback": 4.333918029908091,
      "passing_air_yards": 4,
      "passing_yards_after_catch": 30,
      "pacr": 8.5,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 1.36,
      "season_type": "REG",
      "player_name": "A.J. Cole",
      "player_first_name": "A.J.",
      "player_last_name": "Cole",
      "player_position": "P",
      "player_position_group": "SPEC",
      "player_college_name": "North Carolina State",
      "player_height": 76,
      "player_weight": 220,
      "player_birth_date": "1995-11-27",
      "player_draft_club": null,
      "player_draft_number": null,
      "player_team_abbr": "LV",
      "player_headshot": "https://static.www.nfl.com/image/upload/f_auto,q_auto/league/iuaqixqidj73rkkwczso"
    },
    {
      "season": 2023,
      "player_id": "00-0036313",
      "team": "PHI",
      "position": "QB",
      "games_played": 1,
      "attempts": 1,
      "completions": 1,
      "completion_pct": 100,
      "passing_yards": 28,
      "passing_tds": 0,
      "passing_interceptions": 0,
      "qb_epa": 4.209326545824297,
      "passing_cpoe": 35.070258378982544,
      "epa_per_dropback": 4.209326545824297,
      "passing_air_yards": 11,
      "passing_yards_after_catch": 17,
      "pacr": 2.5454545454545454,
      "passing_first_downs": 1,
      "passing_2pt_conversions": 0,
      "sacks_suffered": 0,
      "passing_fumbles": 0,
      "passing_fumbles_lost": 0,
      "qb_dropback": 1,
      "fantasy_points": 1.12,
      "season_type": "REG",
      "player_name": "Braden Mann",
      "player_first_name": "Braden",
      "player_last_name": "Mann",
      "player_position": "P",
      "player_position_group": "SPEC",
      "player_college_name": "Texas A&amp;M",
      "player_height": 71,
      "player_weight": 190,
      "player_birth_date": "1997-11-24",
      "player_draft_club": "NYJ",
      "player_draft_number": 191,
      "player_team_abbr": "PHI",
      "player_headshot": "https://static.www.nfl.com/image/upload/f_auto,q_auto/league/z2s9qddgientrtm86fmc"
    }
  ]
}
Response headers
 cache-control: max-age=43200 
 content-length: 10017 
 content-type: application/json 
 date: Fri,02 May 2025 12:59:11 GMT 
 etag: W/534950708629701427 
 server: uvicorn 
 x-fastapi-cache: MISS 
Responses
Code	Description	Links
200	
Successful Response

Media type

application/json
Controls Accept header.
Example Value
Schema
"string"
No links
422	
Validation Error

Media type

application/json
Example Value
Schema
{
  "detail": [
    {
      "loc": [
        "string",
        0
      ],
      "msg": "string",
      "type": "string"
    }
  ]
}
No links

GET
/api/{team}/{name}/stats
Get Player Stats By Team Endpoint


GET
/api/team/{team}
Get Stats For Team


GET
/api/game
Get Game Details


GET
/api/game/outlook
Get Game Analysis


GET
/api/compare
Compare Players


GET
/health
Health Check


GET
/
Root


POST
/api/cache/clear
Clear Cache


GET
/api/cache/status
Get Cache Status


Schemas
AggregationTypeExpand allstring
HTTPValidationErrorExpand allobject
ValidationErrorExpand allobject