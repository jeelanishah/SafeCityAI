"""Test API endpoints."""

def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_models_endpoint(client):
    """Test models endpoint."""
    response = client.get("/models")
    assert response.status_code == 200
    assert "current_model" in response.json()


def test_predict_missing_file(client):
    """Test predict without file."""
    response = client.post("/predict")
    assert response.status_code == 422
