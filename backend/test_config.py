from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    
    # Verify the new arrays
    assert "flow_backends" in data
    assert "upscale_models" in data
    assert "interpolation_models" in data
    
    print("Response JSON:")
    print(data)
    
    print("\nTests passed!")
    print(f"Flow Backends: {data['flow_backends']}")
    print(f"Upscale Models: {data['upscale_models']}")
    print(f"Interpolation Models: {data['interpolation_models']}")

if __name__ == "__main__":
    test_config()
