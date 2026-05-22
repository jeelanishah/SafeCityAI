import requests
import json

with open('traffic video dataset.mp4', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect', files=files)

print(json.dumps(response.json(), indent=2))