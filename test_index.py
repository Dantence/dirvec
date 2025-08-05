# test_index.py
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB = "dirvec.db"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def test_index():
    conn = sqlite3.connect(DB)
    cur = conn.execute("SELECT COUNT(*) FROM files")
    count = cur.fetchone()[0]
    print(f"📦 已索引文件数：{count}")

    cur = conn.execute("SELECT path, content, vector FROM files LIMIT 3")
    for path, content, vec_blob in cur.fetchall():
        vec = np.frombuffer(vec_blob, dtype=np.float32)
        print(f"\n🗂️  路径：{path}")
        print(f"📝 内容前 64 字：{content[:64]}...")
        print(f"📏 向量维度：{len(vec)}")
        print("-" * 60)

    conn.close()

def test_incremental():
    import os
    import time
    # 创建一个临时文件，验证增量更新
    tmp_path = "tmp_test_file.txt"
    with open(tmp_path, "w") as f:
        f.write("hello dirvec")
    
    # 重新索引
    import index
    index.index_directory(".")

    conn = sqlite3.connect(DB)
    cur = conn.execute("SELECT path FROM files WHERE path LIKE ?", (f"%{tmp_path}",))
    found = cur.fetchone()
    conn.close()

    if found:
        print("✅ 增量更新测试通过：新文件已入库")
    else:
        print("❌ 增量更新测试失败：新文件未找到")

    # 清理
    os.remove(tmp_path)

if __name__ == "__main__":
    test_index()
    test_incremental()