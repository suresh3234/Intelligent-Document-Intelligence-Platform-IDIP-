import json
with open("data/faiss_index/id_map.json", "r") as f:
    id_map = json.load(f)

doc_ids = set()
for fid, chunk in id_map.items():
    doc_ids.add(chunk["doc_id"])

print("Doc IDs in FAISS id_map.json:", list(doc_ids))
