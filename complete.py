# # complete.py
# import os
# import sys
# import sqlite3
# import numpy as np
# from sentence_transformers import SentenceTransformer

# DB = os.path.join(os.path.dirname(__file__), "dirvec.db")
# MODEL = SentenceTransformer("./models/all-MiniLM-L6-v2")  # 离线模型

# def load_vectors(db_path=DB):
#     if not os.path.exists(db_path):
#         return [], np.empty((0, 0), dtype=np.float32)
#     conn = sqlite3.connect(db_path)
#     try:
#         rows = conn.execute("SELECT path, vector FROM files").fetchall()
#     except sqlite3.OperationalError:
#         rows = []
#     finally:
#         conn.close()
#     if not rows:
#         return [], np.empty((0, 0), dtype=np.float32)
#     paths, vecs = zip(*[(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows if r[1] is not None])
#     if not paths:
#         return [], np.empty((0, 0), dtype=np.float32)
#     dim = vecs[0].shape[0]
#     return list(paths), np.vstack(vecs).reshape(-1, dim)

# def semantic_search(prefix, top_k=20, db_path=DB):
#     paths, vecs = load_vectors(db_path)
#     if len(paths) == 0:
#         return []
#     query_vec = MODEL.encode([prefix], normalize_embeddings=True)
#     scores = (vecs @ query_vec.T).ravel()
#     idx = np.argsort(scores)[::-1][:top_k]
#     return [paths[i] for i in idx]

# if __name__ == "__main__":
#     prefix = sys.argv[1] if len(sys.argv) > 1 else ""
#     db = sys.argv[2] if len(sys.argv) > 2 else DB
#     for p in semantic_search(prefix, db_path=db):
#         print(p)

# complete.py
import os
import sys
import sqlite3
import math
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

DB = os.path.join(os.path.dirname(__file__), "dirvec.db")
MODEL = SentenceTransformer("./models/all-MiniLM-L6-v2")  # 离线模型

# ---------- Intent & bias ----------

EXT_GROUPS = {
    "code": {".py", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb", ".php"},
    "text": {".md", ".txt", ".rst", ".csv", ".tsv", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf"},
    "logs": {".log"},
    "web":  {".html", ".css", ".scss", ".less"},
    "shell": {".sh", ".bash", ".zsh"},
}

INTENT_CONFIG = {
    # command: (w_sem, w_lex, ext_weights, prefer_ftype, dir_mode)
    "vim":   dict(w_sem=0.5, w_lex=0.5, ext_pref={"code":0.25,"text":0.2}, ftype_pref="text", dir_mode=False),
    "nvim":  dict(w_sem=0.5, w_lex=0.5, ext_pref={"code":0.25,"text":0.2}, ftype_pref="text", dir_mode=False),
    "vi":    dict(w_sem=0.5, w_lex=0.5, ext_pref={"code":0.25,"text":0.2}, ftype_pref="text", dir_mode=False),
    "code":  dict(w_sem=0.45, w_lex=0.55, ext_pref={"code":0.3,"text":0.15}, ftype_pref="text", dir_mode=False),
    "grep":  dict(w_sem=0.4, w_lex=0.6, ext_pref={"text":0.25,"logs":0.25}, ftype_pref="text", dir_mode=False),
    "rg":    dict(w_sem=0.4, w_lex=0.6, ext_pref={"text":0.25,"logs":0.25}, ftype_pref="text", dir_mode=False),
    "ag":    dict(w_sem=0.4, w_lex=0.6, ext_pref={"text":0.25,"logs":0.25}, ftype_pref="text", dir_mode=False),
    "python":dict(w_sem=0.45, w_lex=0.55, ext_pref={"code":0.2}, ext_only={".py":0.35}, ftype_pref="text", dir_mode=False),
    "python3":dict(w_sem=0.45, w_lex=0.55, ext_pref={"code":0.2}, ext_only={".py":0.35}, ftype_pref="text", dir_mode=False),
    "node":  dict(w_sem=0.45, w_lex=0.55, ext_pref={"code":0.2}, ext_only={".js":0.25,".ts":0.25}, ftype_pref="text", dir_mode=False),
    "deno":  dict(w_sem=0.45, w_lex=0.55, ext_pref={"code":0.2}, ext_only={".ts":0.3,".js":0.2}, ftype_pref="text", dir_mode=False),
    "tail":  dict(w_sem=0.4, w_lex=0.6, ext_pref={"logs":0.35}, ftype_pref="text", dir_mode=False),
    "less":  dict(w_sem=0.4, w_lex=0.6, ext_pref={"logs":0.3,"text":0.15}, ftype_pref="text", dir_mode=False),
    "cat":   dict(w_sem=0.45, w_lex=0.55, ext_pref={"text":0.2,"logs":0.2}, ftype_pref="text", dir_mode=False),
    "cd":    dict(w_sem=0.55, w_lex=0.45, ext_pref={}, ftype_pref=None, dir_mode=True),
    # default fallback
    "_":     dict(w_sem=0.6, w_lex=0.4, ext_pref={"code":0.1,"text":0.1}, ftype_pref="text", dir_mode=False),
}

def parse_intent(prefix: str) -> Tuple[str, List[str]]:
    tokens = prefix.strip().split()
    if not tokens:
        return "_", []
    cmd = tokens[0].lower()
    terms = tokens[1:] if cmd in INTENT_CONFIG else tokens
    return (cmd if cmd in INTENT_CONFIG else "_"), terms

def ext_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def depth_of(path: str) -> int:
    return path.strip(os.sep).count(os.sep)

def group_of_ext(ext: str) -> str:
    for g, exts in EXT_GROUPS.items():
        if ext in exts:
            return g
    return "other"

# ---------- DB / vectors ----------

def load_all(db_path: str):
    if not os.path.exists(db_path):
        return [], np.empty((0, 0), dtype=np.float32), {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT path, vector, mtime, size, ftype FROM files").fetchall()
        meta = {}
        paths = []
        vecs = []
        for p, v, mtime, size, ftype in rows:
            if v is None:
                continue
            arr = np.frombuffer(v, dtype=np.float32)
            paths.append(p)
            vecs.append(arr)
            meta[p] = {"mtime": mtime or 0.0, "size": size or 0, "ftype": ftype or "text"}
        vecs = np.vstack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
        return paths, vecs, meta
    finally:
        conn.close()

def fts5_enabled(db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("SELECT count(*) FROM files_fts;")
            return True
        finally:
            conn.close()
    except Exception:
        return False

def fts_candidates(db_path: str, terms: List[str], limit: int = 200) -> Dict[str, float]:
    if not terms:
        return {}
    if not fts5_enabled(db_path):
        return {}
    # 构造 MATCH 查询："term*" AND "another*"
    def quote_term(t: str) -> str:
        t = t.strip().replace('"', ' ')
        if not t:
            return ""
        # 用前缀匹配提升补全体验
        return f'"{t}*"'
    q_terms = [quote_term(t) for t in terms if t.strip()]
    if not q_terms:
        return {}
    match_query = " AND ".join(q_terms)
    conn = sqlite3.connect(db_path)
    try:
        # bm25(files_fts) 越小越好
        sql = f"""
            SELECT path, bm25(files_fts) AS b
            FROM files_fts
            WHERE files_fts MATCH ?
            ORDER BY b
            LIMIT {int(limit)}
        """
        out = {}
        for p, b in conn.execute(sql, (match_query,)):
            # 归一化：转为 [0,1]，越大越好
            score = 1.0 / (1.0 + float(b))
            out[p] = max(out.get(p, 0.0), score)
        return out
    except sqlite3.OperationalError:
        # 某些系统禁用了 bm25()，做一个退化排序
        try:
            sql = f"""
                SELECT path
                FROM files_fts
                WHERE files_fts MATCH ?
                LIMIT {int(limit)}
            """
            out = {}
            for idx, (p,) in enumerate(conn.execute(sql, (match_query,))):
                out[p] = max(out.get(p, 0.0), 1.0 - idx / max(1, limit))  # 简单衰减
            return out
        except Exception:
            return {}
    finally:
        conn.close()

# ---------- Scoring & hybrid ----------

def normalize_sem(cos_sim: float) -> float:
    # cosine [-1,1] => [0,1]
    return (cos_sim + 1.0) / 2.0

def recency_boost(mtime: float) -> float:
    # 简单时间衰减（最近更高），30 天半衰期
    if not mtime:
        return 0.0
    days = max(0.0, (time.time() - mtime) / 86400.0)
    half_life = 30.0
    return math.exp(-math.log(2) * days / half_life)  # [0,1]

def ext_bias_of(path: str, cfg: Dict[str, Any]) -> float:
    ext = ext_of(path)
    bias = 0.0
    # 具体扩展优先
    for e, w in (cfg.get("ext_only") or {}).items():
        if ext == e:
            bias += w
    # 分组优先
    g = group_of_ext(ext)
    for grp, w in (cfg.get("ext_pref") or {}).items():
        if g == grp:
            bias += w
    return bias

def ftype_bias(ftype: str, cfg: Dict[str, Any]) -> float:
    pref = cfg.get("ftype_pref")
    if pref is None:
        return 0.0
    return 0.1 if ftype == pref else 0.0

def depth_penalty(path: str) -> float:
    # 更短的路径略有优势（上屏更方便）
    d = depth_of(path)
    return -0.03 * d  # 负数，作为微弱惩罚项

def hybrid_search(prefix: str, db_path: str, top_k: int = 20) -> List[str]:
    cmd, terms = parse_intent(prefix)
    cfg = INTENT_CONFIG.get(cmd, INTENT_CONFIG["_"])

    paths, vecs, meta = load_all(db_path)
    if len(paths) == 0:
        return []

    # 语义召回（全量向量，Level 3 再优化）
    q_vec = MODEL.encode([prefix], normalize_embeddings=True)
    sem_scores = (vecs @ q_vec.T).ravel()  # cosine，[-1,1]
    # 取前若干
    sem_top = 200 if sem_scores.shape[0] > 200 else sem_scores.shape[0]
    sem_idx = np.argsort(sem_scores)[::-1][:sem_top]
    sem_candidates = {paths[i]: normalize_sem(float(sem_scores[i])) for i in sem_idx}

    # 词法召回（FTS5）
    lex_candidates = fts_candidates(db_path, terms, limit=200)

    # 合并候选
    all_paths = set(sem_candidates) | set(lex_candidates)

    # 评分融合
    results = []
    for p in all_paths:
        s_sem = sem_candidates.get(p, 0.0)
        s_lex = lex_candidates.get(p, 0.0)
        s = cfg["w_sem"] * s_sem + cfg["w_lex"] * s_lex
        m = meta.get(p, {})
        s += 0.08 * recency_boost(m.get("mtime", 0.0))
        s += ext_bias_of(p, cfg)
        s += ftype_bias(m.get("ftype", "text"), cfg)
        s += depth_penalty(p)
        results.append((p, s))

    # 特殊：cd 模式 -> 汇聚到目录
    if cfg.get("dir_mode"):
        # 将文件映射到父目录，聚合最大分数；再加浅层偏置
        dir_scores: Dict[str, float] = defaultdict(float)
        for p, s in results:
            d = os.path.dirname(p) if os.path.dirname(p) else "."
            dir_scores[d] = max(dir_scores[d], s + 0.05 * (0 if d == "." else 1) + (-0.02 * depth_of(d)))
        ranked = sorted(dir_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [d for d, _ in ranked]

    # 常规：返回文件
    ranked = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return [p for p, _ in ranked]

# ---------- CLI ----------

def semantic_search(prefix, top_k=20, db_path=DB):
    return hybrid_search(prefix, db_path, top_k=top_k)

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    db = sys.argv[2] if len(sys.argv) > 2 else DB
    for p in semantic_search(prefix, db_path=db):
        print(p)