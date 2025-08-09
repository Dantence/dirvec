#!/usr/bin/env bash
set -euo pipefail

ROOT="dirvec_test"

echo "[1/6] Preparing clean test root: ./${ROOT}"
rm -rf "${ROOT}"
mkdir -p "${ROOT}"

echo "[2/6] Creating directory structure"
mkdir -p "${ROOT}"/{src/src_nested,src/utils,docs,logs,data/"data set",experiments,assets/{images,binary},tests/{unit,integration},deep/a/b/c/d/e/f/g/h/i/j,pipes}
# Ignored dirs by default excludes
mkdir -p "${ROOT}"/{.git,node_modules,dist,build,.venv}

echo "[3/6] Creating text files (UTF-8, Unicode, spaces)"
cat > "${ROOT}/README.md" << 'EOF'
# DirVec Test Project
This is a test corpus for semantic directory-based command completion.
Includes: code, logs, docs, unicode, minified, binaries, large files, symlinks, pipes.
EOF

cat > "${ROOT}/src/main.py" << 'EOF'
import json
from utils.helpers import slugify
def main():
    print("hello dirvec")
    data = {"task": "index", "module": "semantic search"}
    print(json.dumps(data))
if __name__ == "__main__":
    main()
EOF

cat > "${ROOT}/src/utils/helpers.py" << 'EOF'
def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s)
EOF

cat > "${ROOT}/src/utils/parser.py" << 'EOF'
def parse_config(text: str) -> dict:
    return {"keys": [line.split("=")[0] for line in text.splitlines() if "=" in line]}
EOF

cat > "${ROOT}/src/app.js" << 'EOF'
function greet(){ console.log("hello dirvec"); }
greet();
EOF

# 用于 exclude-globs 测试（例如: *.min.js）
echo 'function x(){var a=1;}' > "${ROOT}/src/vendor.min.js"

# 带空格路径
cat > "${ROOT}/data/data set/read me.txt" << 'EOF'
This path contains spaces; useful to test shell completion and semantic retrieval.
EOF

# Unicode 路径与内容
printf "这是一个用于测试语义检索的中文文件。\n包含：路径、内容、编码。\n" > "${ROOT}/项目说明.txt"
mkdir -p "${ROOT}/resume"
printf "Résumé of semantic completion experiments.\n" > "${ROOT}/resume/résumé.txt"

# 非 UTF-8（掺杂非法字节，但仍主要是文本）
printf "partly text\xff\xfe with invalid utf-8 bytes\nmore lines\n" > "${ROOT}/docs/odd_encoding.txt" || true

# 空文件
: > "${ROOT}/docs/empty.txt"

# 日志类文本
cat > "${ROOT}/logs/app.log" << 'EOF'
2024-11-01T12:00:00Z INFO index started
2024-11-01T12:00:01Z INFO scanning files
2024-11-01T12:00:02Z INFO done
EOF

echo "[4/6] Creating binaries, large files, and special files"
# 二进制（会被 BINARY 扩展/启发式过滤）
# 伪 PNG
printf "\x89PNG\r\n\x1a\n" > "${ROOT}/assets/images/logo.png"
head -c 1024 /dev/urandom >> "${ROOT}/assets/images/logo.png"

# .so 二进制
head -c 4096 /dev/urandom > "${ROOT}/assets/binary/engine.so"

# 大文件（默认 >5MB 将被跳过）
dd if=/dev/zero of="${ROOT}/data/large_file.dat" bs=1M count=6 status=none

# 原始二进制数据
head -c 131072 /dev/urandom > "${ROOT}/data/raw.bin"

# minified 版本也放一份到 dist（但 dist 目录默认会被 exclude）
echo '(()=>{let x=0;for(let i=0;i<10;i++)x+=i;console.log(x)})();' > "${ROOT}/dist/app.min.js"

# 符号链接（索引器会跳过）
ln -s "./src/main.py" "${ROOT}/link_to_main.py"
ln -s "./assets" "${ROOT}/symlink_assets_dir"

# 命名管道（特殊文件，索引器会跳过）
mkfifo "${ROOT}/pipes/pipe1"

echo "[5/6] Creating deep nested content"
echo "Deep file for path/recursion test" > "${ROOT}/deep/a/b/c/d/e/f/g/h/i/j/leaf.txt"
echo "Sibling content nested" > "${ROOT}/src/src_nested/nested_readme.md"

echo "[6/6] Creating ignored directories content"
# 默认忽略：.git, node_modules, dist, build, .venv
echo "[core]" > "${ROOT}/.git/config"
mkdir -p "${ROOT}/node_modules/lodash"
echo "{}" > "${ROOT}/node_modules/lodash/package.json"
mkdir -p "${ROOT}/.venv/bin"
echo "# fake venv" > "${ROOT}/.venv/pyvenv.cfg"
mkdir -p "${ROOT}/build"
echo "build artifact" > "${ROOT}/build/app.o"

echo "Done. Test tree created at: ${ROOT}"

# 展示结构（如果安装了 tree）
if command -v tree >/dev/null 2>&1; then
  echo "Directory tree preview:"
  tree -a -L 3 "${ROOT}"
else
  echo "Preview via find (first 60 entries):"
  find "${ROOT}" -maxdepth 3 | head -n 60
fi

cat <<'TIP'

Tips:
- src/vendor.min.js is for --exclude-globs "*.min.js"
- data/large_file.dat (~6MB) should be skipped by --max-size 5 (default)
- assets/images/logo.png & assets/binary/engine.so should be skipped as binary
- .git, node_modules, dist, build, .venv are excluded by default
- link_to_main.py (symlink) and pipes/pipe1 (FIFO) are skipped
- Paths with spaces (data/data set/read me.txt) and Unicode (项目说明.txt, résumé.txt) are included
TIP