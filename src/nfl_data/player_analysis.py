"""Functions for advanced player comparison and analysis."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime
from functools import lru_cache
from .data_loader import load_pbp_data, load_weekly_stats
from fastapi import HTTPException

# Position-specific stat mappings
POSITION_STATS = {
    'QB': [
        'passing_yards', 'passing_tds', 'interceptions', 'completion_percentage',
        'passer_rating', 'rushing_yards', 'rushing_tds', 'sacks', 'qb_hits',
        'pressure_rate', 'time_to_throw', 'air_yards_per_attempt'
    ],
    'RB': [
        'rushing_yards', 'rushing_tds', 'yards_per_carry', 'broken_tackles',
        'receiving_yards', 'receptions', 'targets', 'yards_after_catch',
        'red_zone_touches', 'first_downs'
    ],
    'WR': [
        'receiving_yards', 'receiving_tds', 'receptions', 'targets',
        'yards_per_reception', 'yards_after_catch', 'drops', 'contested_catches',
        'red_zone_targets', 'routes_run', 'average_separation'
    ],
    'TE': [
        'receiving_yards', 'receiving_tds', 'receptions', 'targets',
        'yards_per_reception', 'yards_after_catch', 'drops', 'contested_catches',
        'red_zone_targets', 'blocking_grade'
    ],
    'DEF': [
        'tackles', 'sacks', 'interceptions', 'passes_defended',
        'forced_fumbles', 'fumble_recoveries', 'tackles_for_loss',
        'qb_hits', 'pressures', 'missed_tackles'
    ]
}

# Cache player data to improve performance
@lru_cache(maxsize=1000)
def get_player_info(player_name: str) -> Dict:
    """Get basic player information including position, team, etc."""
    players_df = import_players()
    player = players_df[players_df['player_name'].str.lower() == player_name.lower()].iloc[0]
    return {
        'player_id': player['player_id'],
        'position': player['position'],
        'team': player['current_team'],
        'age': player['age'],
        'experience': player['years_exp']
    }

def get_player_stats(player_name: str, season: Optional[int] = None) -> pd.DataFrame:
    """Get player statistics for a given season."""
    try:
        stats = load_weekly_stats([season] if season else None)
        player_stats = stats[stats['player_name'].str.lower() == player_name.lower()]
        
        if player_stats.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No stats found for player {player_name}"
            )
        
        return player_stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching player stats: {str(e)}"
        )

def get_player_comparison_multi(
    player_names: List[str],
    season: Optional[int] = None,
    week: Optional[int] = None,
    last_n_games: Optional[int] = None,
    **situation_filters
) -> Dict:
    """Compare multiple players with position-specific analysis."""
    # Get player info and group by position
    players_by_pos = {}
    for name in player_names:
        info = get_player_info(name)
        pos = info['position']
        if pos not in players_by_pos:
            players_by_pos[pos] = []
        players_by_pos[pos].append(name)
    
    # Get stats for each player
    all_stats = {}
    for pos, names in players_by_pos.items():
        pos_stats = []
        for name in names:
            stats = get_player_stats(name, season)
            if not stats.empty:
                # Calculate aggregate stats based on position
                agg_stats = {}
                for stat in POSITION_STATS.get(pos, []):
                    if stat in stats.columns:
                        agg_stats[stat] = stats[stat].mean()
                pos_stats.append({
                    'name': name,
                    'stats': agg_stats,
                    'info': get_player_info(name)
                })
        all_stats[pos] = pos_stats
    
    return all_stats

def get_player_on_off_impact(
    player_name: str,
    other_player: str,
    season: Optional[int] = None,
    **filters
) -> Dict:
    """Analyze a player's performance when another player is on/off the field."""
    # Get player info
    player_info = get_player_info(player_name)
    other_info = get_player_info(other_player)
    
    # Import play-by-play data
    if not season:
        season = datetime.now().year
    pbp_data = import_pbp_data([season])
    
    # Get snaps where both players were on field
    on_field_plays = pbp_data[
        (pbp_data['offense_players'].str.contains(player_info['player_id'], na=False)) &
        (pbp_data['offense_players'].str.contains(other_info['player_id'], na=False))
    ]
    
    # Get snaps where only the main player was on field
    off_field_plays = pbp_data[
        (pbp_data['offense_players'].str.contains(player_info['player_id'], na=False)) &
        (~pbp_data['offense_players'].str.contains(other_info['player_id'], na=False))
    ]
    
    # Calculate relevant stats for both scenarios
    stats_with = calculate_position_stats(on_field_plays, player_info['position'])
    stats_without = calculate_position_stats(off_field_plays, player_info['position'])
    
    return {
        'with_player': stats_with,
        'without_player': stats_without,
        'snap_count_with': len(on_field_plays),
        'snap_count_without': len(off_field_plays)
    }

def get_qb_advanced_stats(
    qb_name: str,
    season: Optional[int] = None,
    week: Optional[int] = None,
    **filters
) -> Dict:
    """Get advanced QB statistics broken down by various factors."""
    # Get QB info
    qb_info = get_player_info(qb_name)
    if qb_info['position'] != 'QB':
        raise ValueError(f"{qb_name} is not a quarterback")
    
    # Import play-by-play data
    if not season:
        season = datetime.now().year
    pbp_data = import_pbp_data([season])
    
    # Filter for QB plays
    qb_plays = pbp_data[pbp_data['passer_player_id'] == qb_info['player_id']]
    
    # Calculate stats by dropback type
    dropback_stats = {
        'under_center': calculate_qb_stats(qb_plays[qb_plays['shotgun'] == 0]),
        'shotgun': calculate_qb_stats(qb_plays[qb_plays['shotgun'] == 1])
    }
    
    # Calculate stats by coverage type (if available)
    coverage_stats = {}
    if 'defensive_coverage' in qb_plays.columns:
        for coverage in qb_plays['defensive_coverage'].unique():
            coverage_stats[coverage] = calculate_qb_stats(
                qb_plays[qb_plays['defensive_coverage'] == coverage]
            )
    
    return {
        'by_dropback': dropback_stats,
        'by_coverage': coverage_stats,
        'pressure_stats': {
            'under_pressure': calculate_qb_stats(qb_plays[qb_plays['under_pressure'] == 1]),
            'clean_pocket': calculate_qb_stats(qb_plays[qb_plays['under_pressure'] == 0])
        }
    }

def get_future_schedule_analysis(
    player_name: str,
    weeks_ahead: int = 4
) -> List[Dict]:
    """Analyze upcoming schedule and matchups for a player."""
    # Get player info
    player_info = get_player_info(player_name)
    
    # Get team schedule
    current_season = datetime.now().year
    schedule = import_schedules([current_season])
    
    # Get upcoming games
    team_schedule = schedule[
        ((schedule['home_team'] == player_info['team']) |
         (schedule['away_team'] == player_info['team'])) &
        (schedule['week'] > schedule['week'].max())  # Future games
    ].head(weeks_ahead)
    
    # Analyze each matchup
    matchup_analysis = []
    for _, game in team_schedule.iterrows():
        opponent = game['away_team'] if game['home_team'] == player_info['team'] else game['home_team']
        
        # Get opponent defensive stats
        opp_stats = get_defensive_stats(opponent, current_season)
        
        # Get player's historical performance vs opponent
        historical_stats = get_historical_matchup_stats(player_name, opponent)
        
        matchup_analysis.append({
            'week': game['week'],
            'opponent': opponent,
            'is_home': game['home_team'] == player_info['team'],
            'opponent_defense': opp_stats,
            'historical_performance': historical_stats
        })
    
    return matchup_analysis

def get_game_outlook(
    game_id: str,
    player_name: Optional[str] = None
) -> Dict:
    """Generate game outlook with detailed analysis."""
    # Get game info
    schedule = import_schedules([datetime.now().year])
    game = schedule[schedule['game_id'] == game_id].iloc[0]
    
    # Get team stats
    home_stats = get_team_stats(game['home_team'])
    away_stats = get_team_stats(game['away_team'])
    
    # Get weather data if available
    weather = get_weather_forecast(game_id) if 'weather' in game else None
    
    outlook = {
        'game_info': {
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'kickoff_time': game['gametime'],
            'weather': weather
        },
        'team_analysis': {
            'home': home_stats,
            'away': away_stats
        },
        'key_matchups': analyze_key_matchups(game['home_team'], game['away_team'])
    }
    
    # Add player-specific analysis if requested
    if player_name:
        player_info = get_player_info(player_name)
        outlook['player_analysis'] = analyze_player_matchup(
            player_name,
            game['home_team'],
            game['away_team']
        )
    
    return outlook

# Helper functions for stat calculations
def calculate_position_stats(plays_df: pd.DataFrame, position: str) -> Dict:
    """Calculate relevant statistics for a given position."""
    stats = {}
    for stat in POSITION_STATS.get(position, []):
        if stat in plays_df.columns:
            stats[stat] = plays_df[stat].mean()
    return stats

def calculate_qb_stats(plays_df: pd.DataFrame) -> Dict:
    """Calculate QB statistics from play-by-play data."""
    return {
        'attempts': len(plays_df),
        'completions': plays_df['complete_pass'].sum(),
        'yards': plays_df['passing_yards'].sum(),
        'touchdowns': plays_df['touchdown'].sum(),
        'interceptions': plays_df['interception'].sum(),
        'sacks': plays_df['sack'].sum(),
        'yards_per_attempt': plays_df['passing_yards'].mean(),
        'completion_percentage': (plays_df['complete_pass'].sum() / len(plays_df)) * 100 if len(plays_df) > 0 else 0
    }

def get_defensive_stats(team: str, season: int) -> Dict:
    """Get defensive statistics for a team."""
    # Import play-by-play data
    pbp_data = import_pbp_data([season])
    
    # Filter for defensive plays
    def_plays = pbp_data[pbp_data['defteam'] == team]
    
    if def_plays.empty:
        return {
            'points_allowed_per_game': 0.0,
            'yards_allowed_per_game': 0.0,
            'turnovers_forced': 0,
            'sacks': 0,
            'qb_hits': 0,
            'pressure_rate': 0.0,
            'completion_percentage_allowed': 0.0,
            'yards_per_carry_allowed': 0.0,
            'third_down_stop_rate': 0.0,
            'red_zone_stop_rate': 0.0
        }
    
    # Calculate per-game stats
    games = def_plays['game_id'].nunique()
    points_allowed = def_plays.groupby('game_id')['points_earned'].sum().mean()
    yards_allowed = def_plays.groupby('game_id')['yards_gained'].sum().mean()
    
    # Calculate defensive stats
    stats = {
        'points_allowed_per_game': round(points_allowed, 1),
        'yards_allowed_per_game': round(yards_allowed, 1),
        'turnovers_forced': int(def_plays['interception'].sum() + def_plays['fumble_lost'].sum()),
        'sacks': int(def_plays['sack'].sum()),
        'qb_hits': int(def_plays['qb_hit'].sum()),
        'pressure_rate': round(
            (def_plays['qb_hit'].sum() + def_plays['sack'].sum()) / 
            len(def_plays[def_plays['pass_attempt'] == 1]) * 100, 1
        ) if len(def_plays[def_plays['pass_attempt'] == 1]) > 0 else 0.0,
        'completion_percentage_allowed': round(
            def_plays[def_plays['pass_attempt'] == 1]['complete_pass'].mean() * 100, 1
        ),
        'yards_per_carry_allowed': round(
            def_plays[def_plays['rush_attempt'] == 1]['yards_gained'].mean(), 1
        ) if len(def_plays[def_plays['rush_attempt'] == 1]) > 0 else 0.0,
        'third_down_stop_rate': round(
            (1 - def_plays[def_plays['down'] == 3]['first_down'].mean()) * 100, 1
        ) if len(def_plays[def_plays['down'] == 3]) > 0 else 0.0,
        'red_zone_stop_rate': round(
            (1 - def_plays[def_plays['yardline_100'] <= 20]['touchdown'].mean()) * 100, 1
        ) if len(def_plays[def_plays['yardline_100'] <= 20]) > 0 else 0.0
    }
    
    return stats

def get_historical_matchup_stats(player_name: str, opponent: str) -> Dict:
    """Get player's historical performance against an opponent."""
    # Get last 3 seasons of data
    current_year = datetime.now().year
    seasons = range(current_year - 2, current_year + 1)
    
    # Import weekly stats
    weekly_stats = import_weekly_data(list(seasons))
    
    # Filter for player vs opponent
    player_stats = weekly_stats[
        (weekly_stats['player_name'].str.lower() == player_name.lower()) &
        (weekly_stats['opponent_team'] == opponent)
    ]
    
    if player_stats.empty:
        return {
            'games_played': 0,
            'avg_fantasy_points': 0.0,
            'best_game': None,
            'worst_game': None,
            'total_yards': 0,
            'total_touchdowns': 0
        }
    
    # Get best and worst games
    best_game = player_stats.loc[player_stats['fantasy_points_ppr'].idxmax()]
    worst_game = player_stats.loc[player_stats['fantasy_points_ppr'].idxmin()]
    
    stats = {
        'games_played': len(player_stats),
        'avg_fantasy_points': round(player_stats['fantasy_points_ppr'].mean(), 1),
        'best_game': {
            'week': int(best_game['week']),
            'season': int(best_game['season']),
            'fantasy_points': float(best_game['fantasy_points_ppr']),
            'opponent': opponent
        },
        'worst_game': {
            'week': int(worst_game['week']),
            'season': int(worst_game['season']),
            'fantasy_points': float(worst_game['fantasy_points_ppr']),
            'opponent': opponent
        }
    }
    
    # Add position-specific stats
    player_pos = import_players()[
        import_players()['player_name'].str.lower() == player_name.lower()
    ].iloc[0]['position']
    
    if player_pos == 'QB':
        stats.update({
            'passing_yards_per_game': round(player_stats['passing_yards'].mean(), 1),
            'passing_tds_per_game': round(player_stats['passing_tds'].mean(), 1),
            'interceptions_per_game': round(player_stats['interceptions'].mean(), 1),
            'completion_percentage': round(
                player_stats['completions'].sum() / player_stats['attempts'].sum() * 100, 1
            ) if player_stats['attempts'].sum() > 0 else 0.0
        })
    elif player_pos == 'RB':
        stats.update({
            'rushing_yards_per_game': round(player_stats['rushing_yards'].mean(), 1),
            'rushing_tds_per_game': round(player_stats['rushing_tds'].mean(), 1),
            'receptions_per_game': round(player_stats['receptions'].mean(), 1),
            'receiving_yards_per_game': round(player_stats['receiving_yards'].mean(), 1)
        })
    elif player_pos in ['WR', 'TE']:
        stats.update({
            'targets_per_game': round(player_stats['targets'].mean(), 1),
            'receptions_per_game': round(player_stats['receptions'].mean(), 1),
            'receiving_yards_per_game': round(player_stats['receiving_yards'].mean(), 1),
            'receiving_tds_per_game': round(player_stats['receiving_tds'].mean(), 1)
        })
    
    return stats

def get_team_stats(team: str) -> Dict:
    """Get comprehensive team statistics."""
    # Implementation depends on available data
    pass

def analyze_key_matchups(home_team: str, away_team: str) -> List[Dict]:
    """Analyze key positional matchups between teams."""
    # Get current season
    season = datetime.now().year
    
    # Import depth charts and defensive stats
    depth_charts = import_depth_charts([season])
    home_def_stats = get_defensive_stats(home_team, season)
    away_def_stats = get_defensive_stats(away_team, season)
    
    # Get starters for both teams
    home_starters = depth_charts[
        (depth_charts['team'] == home_team) & 
        (depth_charts['depth_team'] == 1)
    ]
    away_starters = depth_charts[
        (depth_charts['team'] == away_team) & 
        (depth_charts['depth_team'] == 1)
    ]
    
    matchups = []
    
    # QB vs Pass Defense
    home_qb = home_starters[home_starters['position'] == 'QB'].iloc[0] if not home_starters[home_starters['position'] == 'QB'].empty else None
    away_qb = away_starters[away_starters['position'] == 'QB'].iloc[0] if not away_starters[away_starters['position'] == 'QB'].empty else None
    
    if home_qb is not None:
        matchups.append({
            'type': 'QB vs Pass Defense',
            'player': home_qb['player_name'],
            'team': home_team,
            'opponent': away_team,
            'defense_stats': {
                'completion_percentage_allowed': away_def_stats['completion_percentage_allowed'],
                'pressure_rate': away_def_stats['pressure_rate'],
                'sacks': away_def_stats['sacks']
            }
        })
    
    if away_qb is not None:
        matchups.append({
            'type': 'QB vs Pass Defense',
            'player': away_qb['player_name'],
            'team': away_team,
            'opponent': home_team,
            'defense_stats': {
                'completion_percentage_allowed': home_def_stats['completion_percentage_allowed'],
                'pressure_rate': home_def_stats['pressure_rate'],
                'sacks': home_def_stats['sacks']
            }
        })
    
    # RB vs Run Defense
    for team, starters, def_stats in [
        (home_team, home_starters, away_def_stats),
        (away_team, away_starters, home_def_stats)
    ]:
        rb = starters[starters['position'] == 'RB'].iloc[0] if not starters[starters['position'] == 'RB'].empty else None
        if rb is not None:
            matchups.append({
                'type': 'RB vs Run Defense',
                'player': rb['player_name'],
                'team': team,
                'opponent': away_team if team == home_team else home_team,
                'defense_stats': {
                    'yards_per_carry_allowed': def_stats['yards_per_carry_allowed'],
                    'red_zone_stop_rate': def_stats['red_zone_stop_rate']
                }
            })
    
    # WR/TE vs Pass Defense
    for team, starters, def_stats in [
        (home_team, home_starters, away_def_stats),
        (away_team, away_starters, home_def_stats)
    ]:
        for pos in ['WR', 'TE']:
            receivers = starters[starters['position'] == pos]
            for _, receiver in receivers.iterrows():
                matchups.append({
                    'type': f'{pos} vs Pass Defense',
                    'player': receiver['player_name'],
                    'team': team,
                    'opponent': away_team if team == home_team else home_team,
                    'defense_stats': {
                        'completion_percentage_allowed': def_stats['completion_percentage_allowed'],
                        'third_down_stop_rate': def_stats['third_down_stop_rate']
                    }
                })
    
    return matchups

def analyze_player_matchup(player_name: str, home_team: str, away_team: str) -> Dict:
    """Analyze specific player matchup for a game."""
    # Get player info
    players_df = import_players()
    player = players_df[players_df['player_name'].str.lower() == player_name.lower()].iloc[0]
    player_team = player['current_team']
    opponent = away_team if player_team == home_team else home_team
    
    # Get historical performance vs opponent
    historical_stats = get_historical_matchup_stats(player_name, opponent)
    
    # Get opponent defensive stats
    opp_def_stats = get_defensive_stats(opponent, datetime.now().year)
    
    # Get matchup-specific analysis based on position
    matchup_analysis = {}
    if player['position'] == 'QB':
        matchup_analysis = {
            'completion_percentage_allowed': opp_def_stats['completion_percentage_allowed'],
            'pressure_rate': opp_def_stats['pressure_rate'],
            'sacks_allowed': opp_def_stats['sacks'],
            'historical_completion_percentage': historical_stats.get('completion_percentage', 0.0),
            'historical_passing_yards_per_game': historical_stats.get('passing_yards_per_game', 0.0)
        }
    elif player['position'] == 'RB':
        matchup_analysis = {
            'yards_per_carry_allowed': opp_def_stats['yards_per_carry_allowed'],
            'red_zone_stop_rate': opp_def_stats['red_zone_stop_rate'],
            'historical_rushing_yards_per_game': historical_stats.get('rushing_yards_per_game', 0.0),
            'historical_receptions_per_game': historical_stats.get('receptions_per_game', 0.0)
        }
    elif player['position'] in ['WR', 'TE']:
        matchup_analysis = {
            'completion_percentage_allowed': opp_def_stats['completion_percentage_allowed'],
            'third_down_stop_rate': opp_def_stats['third_down_stop_rate'],
            'historical_targets_per_game': historical_stats.get('targets_per_game', 0.0),
            'historical_receiving_yards_per_game': historical_stats.get('receiving_yards_per_game', 0.0)
        }
    
    return {
        'player_info': {
            'name': player_name,
            'position': player['position'],
            'team': player_team
        },
        'opponent': opponent,
        'historical_performance': historical_stats,
        'opponent_defense': opp_def_stats,
        'matchup_analysis': matchup_analysis
    }

def get_weather_forecast(game_id: str) -> Dict:
    """Get weather forecast for a game."""
    # This would require integration with a weather API
    # For now, return a placeholder
    return {
        'temperature': None,
        'conditions': None,
        'wind_speed': None,
        'precipitation': None
    } 