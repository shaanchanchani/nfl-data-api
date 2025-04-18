"""NFL Data API package."""

from .data_import import (
    import_pbp_data,
    import_weekly_data,
    import_players,
    import_schedules,
    import_injuries,
    import_depth_charts
)

from .stats_helpers import (
    get_defensive_stats,
    get_historical_matchup_stats,
    get_team_stats,
    analyze_key_matchups,
    analyze_player_matchup,
    get_player_stats,
    get_player_game_log,
    get_player_career_stats,
    get_player_comparison,
    get_game_stats,
    get_situation_stats,
    get_player_on_field_stats,
    resolve_player,
    get_position_specific_stats_from_pbp
)

from .player_analysis import (
    get_player_on_off_impact,
    get_qb_advanced_stats,
    get_future_schedule_analysis,
    get_game_outlook
)

from .main import app

__all__ = [
    'import_pbp_data',
    'import_weekly_data',
    'import_players',
    'import_schedules',
    'import_injuries',
    'import_depth_charts',
    'get_defensive_stats',
    'get_historical_matchup_stats',
    'analyze_key_matchups',
    'analyze_player_matchup',
    'get_player_stats',
    'get_player_game_log',
    'get_player_career_stats',
    'get_player_comparison',
    'get_game_stats',
    'get_situation_stats',
    'get_player_on_field_stats',
    'get_player_on_off_impact',
    'get_qb_advanced_stats',
    'get_future_schedule_analysis',
    'get_game_outlook',
    'app'
]
