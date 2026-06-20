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

headers = {"Authorization": f"Bearer {token}"}

# List documents
resp = requests.get("http://localhost:8000/v1/documents", headers=headers)
print("List Status:", resp.status_code)
print("Documents total:", resp.json().get("total"))
docs = resp.json().get("documents", [])
for d in docs:
    print(f" - ID: {d.get('doc_id')}, Status: {d.get('status')}, Filename: {d.get('filename')}")

# Get details of our doc
doc_id = "956d19df-98be-4483-9cf5-1bda7a7ffab4"
resp_detail = requests.get(f"http://localhost:8000/v1/documents/{doc_id}", headers=headers)
print("Detail Status:", resp_detail.status_code)
if resp_detail.status_code == 200:
    print("Detail Response keys:", list(resp_detail.json().keys()))
    print("Detail Metadata:", resp_detail.json().get("metadata"))
else:
    print("Error detail:", resp_detail.text)
