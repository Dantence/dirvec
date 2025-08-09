# # complete.py
# import os, sys, sqlite3, numpy as np
# from sentence_transformers import SentenceTransformer

# DB = os.path.join(os.path.dirname(__file__), "dirvec.db")
# MODEL = SentenceTransformer("./models/all-MiniLM-L6-v2")  # 离线模型

# def load_vectors():
#     conn = sqlite3.connect(DB)
#     rows = conn.execute("SELECT path, vector FROM files").fetchall()
#     conn.close()
#     if not rows:
#         return [], np.empty((0, 384), dtype=np.float32)
#     paths, vecs = zip(*[(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows])
#     return list(paths), np.vstack(vecs)

# def semantic_search(prefix, top_k=20):
#     """
#     prefix: 用户已输入的字符串，如 'vim src'、'vim src/uti' 等
#     直接把 prefix 向量化，召回最相似的文件路径
#     """
#     paths, vecs = load_vectors()
#     if not paths:
#         return []

#     query_vec = MODEL.encode([prefix], normalize_embeddings=True)
#     scores = (vecs @ query_vec.T).ravel()
#     idx = np.argsort(scores)[::-1][:top_k]
#     return [paths[i] for i in idx]   # ← 不再用 prefix 过滤

# if __name__ == "__main__":
#     prefix = sys.argv[1] if len(sys.argv) > 1 else ""
#     for p in semantic_search(prefix):
#         print(p)


# complete.py
import os
import sys
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB = os.path.join(os.path.dirname(__file__), "dirvec.db")
MODEL = SentenceTransformer("./models/all-MiniLM-L6-v2")  # 离线模型

def load_vectors(db_path=DB):
    if not os.path.exists(db_path):
        return [], np.empty((0, 0), dtype=np.float32)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT path, vector FROM files").fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    paths, vecs = zip(*[(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows if r[1] is not None])
    if not paths:
        return [], np.empty((0, 0), dtype=np.float32)
    dim = vecs[0].shape[0]
    return list(paths), np.vstack(vecs).reshape(-1, dim)

def semantic_search(prefix, top_k=20, db_path=DB):
    paths, vecs = load_vectors(db_path)
    if len(paths) == 0:
        return []
    query_vec = MODEL.encode([prefix], normalize_embeddings=True)
    scores = (vecs @ query_vec.T).ravel()
    idx = np.argsort(scores)[::-1][:top_k]
    return [paths[i] for i in idx]

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    db = sys.argv[2] if len(sys.argv) > 2 else DB
    for p in semantic_search(prefix, db_path=db):
        print(p)