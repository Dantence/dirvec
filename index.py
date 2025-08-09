# index.py
import os
import sys
import argparse
import sqlite3
import stat
import fnmatch
import hashlib
from typing import List, Tuple, Dict, Set

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_DB = "dirvec.db"
DEFAULT_MODEL_PATH = "./models/all-MiniLM-L6-v2"

DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn", ".idea", ".vscode",
    "node_modules", "dist", "build", "out",
    ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".cache", "target"
]
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".so", ".dll", ".dylib", ".o", ".a", ".lib",
    ".ttf", ".otf", ".woff", ".woff2",
    ".mp3", ".wav", ".flac", ".mp4", ".mkv", ".mov", ".avi",
    ".class", ".jar", ".bin", ".iso"
}

READ_HEAD_BYTES = 8192  # 读取文件头部内容用于语义

def init_db(db_path: str) -> Tuple[sqlite3.Connection, bool]:
    """
    初始化主表与（可用时）FTS5 虚表。
    返回: (conn, fts5_enabled)
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=134217728;")

    # 主表（与 Level 1 保持一致并扩展）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            inode INTEGER PRIMARY KEY,
            mtime REAL,
            path TEXT,
            content TEXT,
            vector BLOB
        )
    """)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()}
    if "size" not in existing_cols:
        conn.execute("ALTER TABLE files ADD COLUMN size INTEGER;")
    if "ctime" not in existing_cols:
        conn.execute("ALTER TABLE files ADD COLUMN ctime REAL;")
    if "hash" not in existing_cols:
        conn.execute("ALTER TABLE files ADD COLUMN hash TEXT;")
    if "ftype" not in existing_cols:
        conn.execute("ALTER TABLE files ADD COLUMN ftype TEXT;")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_files_path ON files(path);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);")
    conn.commit()

    # 探测并创建 FTS5
    fts5_ok = False
    try:
        # 注意：即使 IF NOT EXISTS，若当前 SQLite 未编译 FTS5，这行也会抛 "no such module: fts5"
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts
            USING fts5(
                path,
                content,
                tokenize='unicode61',
                prefix='2,3,4'
            );
        """)
        # 探测是否真的可用（若可 SELECT 则 OK）
        conn.execute("SELECT count(*) FROM files_fts;")
        fts5_ok = True
    except Exception:
        fts5_ok = False
    finally:
        conn.commit()

    return conn, fts5_ok

def load_model(model_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_path)

def file_stat(path: str):
    st = os.stat(path, follow_symlinks=False)
    return {
        "inode": getattr(st, "st_ino", None),
        "mtime": st.st_mtime,
        "ctime": st.st_ctime,
        "size": st.st_size,
        "mode": st.st_mode,
    }

def is_probably_text(head: bytes) -> bool:
    if not head:
        return True
    if b"\x00" in head:
        return False
    textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)))
    nontext = sum(1 for b in head if b not in textchars)
    return (nontext / max(1, len(head))) < 0.30

def guess_ftype(path: str, head: bytes) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in BINARY_EXTS:
        return "binary"
    return "text" if is_probably_text(head) else "binary"

def fast_hash(path: str, st: dict, head: str) -> str:
    h = hashlib.sha256()
    h.update(path.encode("utf-8", errors="ignore"))
    h.update(str(st["size"]).encode())
    h.update(str(st["mtime"]).encode())
    h.update(head.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def vectorize_batch(model: SentenceTransformer, texts: List[str], normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    arr = model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

def should_exclude_dir(dir_name: str, excludes: List[str]) -> bool:
    for pat in excludes:
        if pat == dir_name or fnmatch.fnmatch(dir_name, pat):
            return True
    return False

def should_exclude_path(path: str, exclude_globs: List[str]) -> bool:
    for pat in exclude_globs:
        if fnmatch.fnmatch(path, pat):
            return True
    return False

def index_directory(
    root: str = ".",
    db_path: str = DEFAULT_DB,
    model_path: str = DEFAULT_MODEL_PATH,
    exclude: List[str] = None,
    exclude_globs: List[str] = None,
    max_size_mb: int = 5,
    batch_size: int = 64,
    prune_deleted: bool = False,
):
    exclude = exclude or DEFAULT_EXCLUDES
    exclude_globs = exclude_globs or []
    max_size_bytes = max_size_mb * 1024 * 1024

    conn, fts5_enabled = init_db(db_path)
    model = load_model(model_path)

    scanned = 0
    skipped_dirs = 0
    skipped_binary = 0
    skipped_large = 0
    skipped_excluded = 0
    unchanged = 0
    updated = 0
    errors = 0

    seen_paths: Set[str] = set()
    batch_rows: List[Tuple] = []
    batch_texts: List[str] = []

    def flush_batch():
        nonlocal updated
        if not batch_rows:
            return
        vecs = vectorize_batch(model, batch_texts, normalize=True)
        with conn:
            for (inode, mtime, path, size, ctime, fhash, ftype, content), vec in zip(batch_rows, vecs):
                conn.execute(
                    "REPLACE INTO files(inode, mtime, path, content, vector, size, ctime, hash, ftype) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        int(inode) if inode is not None else None,
                        float(mtime),
                        path,
                        content,
                        vec.tobytes(),
                        int(size),
                        float(ctime),
                        fhash,
                        ftype,
                    ),
                )
            # 同步 FTS（若可用）
            if fts5_enabled:
                conn.executemany("DELETE FROM files_fts WHERE path = ?", [(r[2],) for r in batch_rows])
                conn.executemany(
                    "INSERT INTO files_fts(path, content) VALUES (?, ?)",
                    [(r[2], r[7]) for r in batch_rows]
                )
        updated += len(batch_rows)
        batch_rows.clear()
        batch_texts.clear()

    existing_meta: Dict[str, Tuple[float, int]] = {}
    for row in conn.execute("SELECT path, mtime, size FROM files"):
        existing_meta[row[0]] = (row[1], row[2] if row[2] is not None else -1)

    root_abs = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root_abs, followlinks=False):
        original_len = len(dirnames)
        dirnames[:] = [d for d in dirnames if not should_exclude_dir(d, exclude)]
        skipped_dirs += (original_len - len(dirnames))

        for name in filenames:
            full_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(full_path, root_abs)
            scanned += 1

            if should_exclude_path(rel_path, exclude_globs):
                skipped_excluded += 1
                continue

            try:
                st = file_stat(full_path)

                if stat.S_ISLNK(st["mode"]) or not stat.S_ISREG(st["mode"]):
                    continue

                if st["size"] is not None and st["size"] > max_size_bytes:
                    skipped_large += 1
                    continue

                try:
                    with open(full_path, "rb") as f:
                        head_bytes = f.read(READ_HEAD_BYTES)
                except Exception:
                    head_bytes = b""
                ftype = guess_ftype(full_path, head_bytes)
                if ftype == "binary":
                    skipped_binary += 1
                    continue

                try:
                    head_text = head_bytes.decode("utf-8", errors="ignore") if head_bytes else ""
                except Exception:
                    head_text = ""

                content = f"{rel_path}\n{head_text}"
                fhash = fast_hash(rel_path, st, head_text)

                old = existing_meta.get(rel_path)
                if old is not None:
                    old_mtime, old_size = old
                    if abs(old_mtime - st["mtime"]) < 1e-6 and (old_size == st["size"]):
                        unchanged += 1
                        seen_paths.add(rel_path)
                        continue

                batch_rows.append(
                    (
                        st["inode"] if st["inode"] is not None else 0,
                        st["mtime"],
                        rel_path,
                        st["size"] if st["size"] is not None else 0,
                        st["ctime"],
                        fhash,
                        ftype,
                        content,
                    )
                )
                batch_texts.append(content)
                seen_paths.add(rel_path)

                if len(batch_rows) >= batch_size:
                    flush_batch()

            except Exception:
                errors += 1
                continue

    flush_batch()

    if prune_deleted:
        existing_paths = {row[0] for row in conn.execute("SELECT path FROM files")}
        to_delete = [p for p in existing_paths if p not in seen_paths]
        if to_delete:
            with conn:
                conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in to_delete])
                if fts5_enabled:
                    conn.executemany("DELETE FROM files_fts WHERE path = ?", [(p,) for p in to_delete])

    conn.close()

    print("Indexing complete ✅")
    print(f"Scanned: {scanned}, Updated: {updated}, Unchanged: {unchanged}, "
          f"Skipped(binary/large/excluded/dirs): {skipped_binary}/{skipped_large}/{skipped_excluded}/{skipped_dirs}, "
          f"Errors: {errors}")

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DirVec indexer with FTS5 hybrid support")
    p.add_argument("--root", default=".", help="Root directory to index")
    p.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="SentenceTransformer model path")
    p.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDES),
                   help="Comma-separated directory names to exclude (exact or glob)")
    p.add_argument("--exclude-globs", default="", help="Comma-separated path globs to exclude (e.g., *.min.js, */tests/*)")
    p.add_argument("--max-size", type=int, default=5, help="Max file size (MB) to index")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    p.add_argument("--prune", action="store_true", help="Remove DB rows for deleted files under root")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    excludes = [x.strip() for x in args.exclude.split(",") if x.strip()]
    exclude_globs = [x.strip() for x in args.exclude_globs.split(",") if x.strip()]
    index_directory(
        root=args.root,
        db_path=args.db,
        model_path=args.model,
        exclude=excludes,
        exclude_globs=exclude_globs,
        max_size_mb=args.max_size,
        batch_size=args.batch_size,
        prune_deleted=args.prune,
    )

