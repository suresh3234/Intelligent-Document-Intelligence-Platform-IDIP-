import redis
import sqlite3
from config import settings

# 1. Clear Redis keys
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
keys = r.keys("idip:dedup:*")
print("Found Redis keys:", keys)
for k in keys:
    r.delete(k)
print("Redis keys deleted.")

# 2. Clear Database rows
conn = sqlite3.connect("idip_metadata.db")
cursor = conn.cursor()
target_ids = ["69e63934-61bc-4ccf-a687-52ff839c5bf3", "0f68f775-6637-473c-80c8-fd08a808ce07"]
for doc_id in target_ids:
    cursor.execute("DELETE FROM document_catalogue WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM feature_store WHERE doc_id = ?", (doc_id,))
conn.commit()
conn.close()
print("Database entries deleted.")
