# index.py
import os
import sqlite3
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

DB = "dirvec.db"
MODEL = SentenceTransformer("./models/all-MiniLM-L6-v2")

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            inode INTEGER PRIMARY KEY,
            mtime REAL,
            path TEXT,
            content TEXT,
            vector BLOB
        )
    """)
    conn.commit()
    return conn

def file_signature(path):
    st = os.stat(path)
    return st.st_ino, st.st_mtime

def read_head(path, size=1024):
    try:
        with open(path, "rb") as f:
            return f.read(size).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def vectorize(text):
    vec = MODEL.encode([text], normalize_embeddings=True)[0]
    return vec.astype(np.float32).tobytes()

def index_directory(root="."):
    conn = init_db()
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            inode, mtime = file_signature(path)
            cur = conn.execute("SELECT mtime FROM files WHERE inode=?", (inode,))
            row = cur.fetchone()
            if row and abs(row[0] - mtime) < 1:
                continue  # unchanged
            content = f"{os.path.relpath(path)} {read_head(path)}"
            vec = vectorize(content)
            conn.execute("REPLACE INTO files(inode, mtime, path, content, vector) VALUES (?,?,?,?,?)",
                         (inode, mtime, path, content, vec))
    conn.commit()
    conn.close()
    print("Indexing complete ✅")

if __name__ == "__main__":
    index_directory()