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

url = "http://localhost:8000/v1/documents/ingest"
headers = {"Authorization": f"Bearer {token}"}
file_path = "C:/Users/devar/Downloads/Suri_dataAnalyst resume (1).pdf"

with open(file_path, "rb") as f:
    files = {"file": ("Suri_dataAnalyst resume (1).pdf", f, "application/pdf")}
    metadata = {
        "source_uri": "upload://Suri_dataAnalyst resume (1).pdf",
        "source_type": "pdf",
        "chunk_strategy": "fixed"
    }
    data = {"metadata": json.dumps(metadata)}

    resp = requests.post(url, headers=headers, files=files, data=data)
    print("Status:", resp.status_code)
    print("Response:", resp.json())
