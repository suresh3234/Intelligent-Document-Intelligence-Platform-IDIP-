import base64
import hmac
import hashlib
import json
import time
import requests

secret = "idip_secret_key_1234567890"
header = {"alg": "HS256", "typ": "JWT"}
payload = {"api_key": "test_client_key", "exp": time.time() + 3600}

def b64_url_encode(d: dict) -> str:
    s = json.dumps(d)
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("utf-8").rstrip("=")
    
h_b64 = b64_url_encode(header)
p_b64 = b64_url_encode(payload)
signing_input = f"{h_b64}.{p_b64}".encode("utf-8")
sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
token = f"{h_b64}.{p_b64}.{sig_b64}"

url = "http://localhost:8000/v1/query/stream"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

doc_id = "956d19df-98be-4483-9cf5-1bda7a7ffab4"
payload = {
    "query": "what are the skills does he have?",
    "top_k": 5,
    "filters": {"doc_id": doc_id}
}

resp = requests.post(url, headers=headers, json=payload, stream=True)
print("Status Code:", resp.status_code)
for line in resp.iter_lines():
    if line:
        print(line.decode("utf-8"))
