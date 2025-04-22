"""Test suite for NFL Data API endpoints."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from src.nfl_data.main import app

@pytest.fixture
def api_client():
    """Create a test client for the API."""
    return TestClient(app)

@pytest.mark.api
def test_root_endpoint(api_client):
    """Test root endpoint returns welcome message."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "docs_url" in response.json()

@pytest.mark.health
def test_health_endpoint(api_client):
    """Test health check endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.player
@pytest.mark.qb
def test_player_endpoint_qb(api_client):
    """Test player endpoint with a QB."""
    response = api_client.get("/api/player/Patrick Mahomes")
    assert response.status_code == 200
    
    data = response.json()
    assert "player_info" in data
    assert "stats" in data
    
    # Check player info
    player_info = data["player_info"]
    assert player_info["position"] == "QB"
    assert player_info["name"] == "Patrick Mahomes"
    assert "team" in player_info
    assert "player_id" in player_info
    
    # Check QB-specific stats
    stats = data["stats"]
    assert "total_attempts" in stats
    assert "completions" in stats
    assert "passing_yards" in stats
    assert "passing_tds" in stats
    assert "interceptions" in stats
    assert "completion_percentage" in stats
    assert "yards_per_attempt" in stats

@pytest.mark.player
@pytest.mark.rb
def test_player_endpoint_rb(api_client):
    """Test player endpoint with a RB."""
    response = api_client.get("/api/player/Christian McCaffrey")
    assert response.status_code == 200
    
    data = response.json()
    assert "player_info" in data
    assert "stats" in data
    
    # Check player info
    player_info = data["player_info"]
    assert player_info["position"] == "RB"
    assert player_info["name"] == "Christian McCaffrey"
    
    # Check RB-specific stats
    stats = data["stats"]
    assert "total_rush_attempts" in stats
    assert "total_rushing_yards" in stats
    assert "rushing_tds" in stats
    assert "yards_per_carry" in stats
    assert "rushing_first_downs" in stats

@pytest.mark.player
@pytest.mark.wr
def test_player_endpoint_wr(api_client):
    """Test player endpoint with a WR."""
    response = api_client.get("/api/player/Justin Jefferson")
    assert response.status_code == 200
    
    data = response.json()
    assert "player_info" in data
    assert "stats" in data
    
    # Check player info
    player_info = data["player_info"]
    assert player_info["position"] == "WR"
    assert player_info["name"] == "Justin Jefferson"
    
    # Check WR-specific stats
    stats = data["stats"]
    assert "total_targets" in stats
    assert "total_receptions" in stats
    assert "total_receiving_yards" in stats
    assert "receiving_tds" in stats
    assert "catch_rate" in stats
    assert "yards_per_reception" in stats

@pytest.mark.player
@pytest.mark.te
def test_player_endpoint_te(api_client):
    """Test player endpoint with a TE."""
    response = api_client.get("/api/player/Travis Kelce")
    assert response.status_code == 200
    
    data = response.json()
    assert "player_info" in data
    assert "stats" in data
    
    # Check player info
    player_info = data["player_info"]
    assert player_info["position"] == "TE"
    assert player_info["name"] == "Travis Kelce"
    
    # Check TE-specific stats
    stats = data["stats"]
    assert "total_targets" in stats
    assert "total_receptions" in stats
    assert "total_receiving_yards" in stats
    assert "receiving_tds" in stats
    assert "catch_rate" in stats
    assert "yards_per_reception" in stats

@pytest.mark.player
@pytest.mark.filters
def test_player_endpoint_filters(api_client):
    """Test player endpoint with various filters."""
    # Test with season filter
    response = api_client.get("/api/player/Patrick Mahomes?season=2023")
    assert response.status_code == 200
    
    # Test with week filter
    response = api_client.get("/api/player/Patrick Mahomes?week=1")
    assert response.status_code == 200
    
    # Test with multiple filters
    response = api_client.get("/api/player/Patrick Mahomes?season=2023&week=1&is_red_zone=true")
    assert response.status_code == 200
    
    # Test with quarter filter
    response = api_client.get("/api/player/Patrick Mahomes?quarter=4")
    assert response.status_code == 200
    
    # Test with down filter
    response = api_client.get("/api/player/Patrick Mahomes?down=3")
    assert response.status_code == 200

@pytest.mark.player
@pytest.mark.error_handling
def test_player_endpoint_not_found(api_client):
    """Test player endpoint with non-existent player."""
    response = api_client.get("/api/player/XYZ ABC NotARealPlayer")
    assert response.status_code == 404
    assert "detail" in response.json()

@pytest.mark.player
@pytest.mark.error_handling
def test_player_endpoint_invalid_name_format(api_client):
    """Test player endpoint with invalid name format."""
    # Test with only last name
    response = api_client.get("/api/player/Williams")
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "first and last name" in response.json()["detail"].lower()

@pytest.mark.player
@pytest.mark.error_handling
def test_player_endpoint_defensive_player(api_client):
    """Test player endpoint with defensive player."""
    response = api_client.get("/api/player/Aaron Donald")
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "stats not available" in response.json()["detail"].lower()

@pytest.mark.player
@pytest.mark.filters
@pytest.mark.situation
def test_player_endpoint_situation_filters(api_client):
    """Test player endpoint with situation-specific filters."""
    # Test red zone stats
    response = api_client.get("/api/player/Travis Kelce?is_red_zone=true")
    assert response.status_code == 200
    
    # Test third down stats
    response = api_client.get("/api/player/Travis Kelce?is_third_down=true")
    assert response.status_code == 200
    
    # Test fourth quarter clutch stats
    response = api_client.get("/api/player/Travis Kelce?is_fourth_quarter_clutch=true")
    assert response.status_code == 200
    
    # Test when trailing
    response = api_client.get("/api/player/Travis Kelce?is_trailing=true")
    assert response.status_code == 200

@pytest.mark.player
@pytest.mark.filters
@pytest.mark.home_away
def test_player_endpoint_home_away_splits(api_client):
    """Test player endpoint with home/away filters."""
    # Test home games
    response = api_client.get("/api/player/Justin Jefferson?is_home_team=true")
    assert response.status_code == 200
    
    # Test away games
    response = api_client.get("/api/player/Justin Jefferson?is_away_team=true")
    assert response.status_code == 200

@pytest.mark.player
@pytest.mark.error_handling
@pytest.mark.invalid_params
def test_player_endpoint_invalid_params(api_client):
    """Test player endpoint with invalid parameters."""
    # Test invalid season
    response = api_client.get("/api/player/Patrick Mahomes?season=1900")
    assert response.status_code == 500
    
    # Test invalid week
    response = api_client.get("/api/player/Patrick Mahomes?week=25")
    assert response.status_code == 500
    
    # Test invalid quarter
    response = api_client.get("/api/player/Patrick Mahomes?quarter=6")
    assert response.status_code == 500
    
    # Test invalid down
    response = api_client.get("/api/player/Patrick Mahomes?down=5")
    assert response.status_code == 500 