"""Test API endpoints."""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("TESTING SAFECITYAI API")
print("=" * 60)

# Test 1: Health Check
print("\n1️⃣  Testing Health Check...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Model Info
print("\n2️⃣  Testing Model Info...")
try:
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ API Tests Complete!")
print("=" * 60)