import requests
import json

urls = [
    "http://localhost:11434/api/tags",
    "https://mitsubishi-refers-raid-rail.trycloudflare.com/api/tags"
]

for url in urls:
    print(f"Checking {url}...")
    try:
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Ollama is reachable! {response.json()}")
        else:
            print(f"Not reachable. Response: {response.text[:100]}")
    except Exception as e:
        print(f"Error: {e}")
