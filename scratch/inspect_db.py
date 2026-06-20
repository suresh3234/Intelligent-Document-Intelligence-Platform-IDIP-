import sqlite3
conn = sqlite3.connect("idip_metadata.db")
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", c.fetchall())
c.execute("SELECT * FROM document_catalogue;")
print("document_catalogue rows:")
for r in c.fetchall():
    print(r)
conn.close()
