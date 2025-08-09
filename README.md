# 基于目录语义的智能终端命令补全工具

## 项目介绍

本项目实现了一个基于目录语义的智能终端命令补全工具，采用词法和语义的“混合检索 + 命令意图偏置”策略，提升命令补全的准确度和实用性。

主要特性包括：

- **高质量检索**  
  在 SQLite 中引入 FTS5（BM25）全文检索，与语义向量相似度融合，提高检索效果。

- **命令意图偏置**  
  根据命令词（如 `vim`、`grep`、`python`、`cd` 等）对文件类型、扩展名、目录进行偏置，更贴近真实命令场景。

- 支持多种命令词意图解析及扩展名权重调整。

## 使用说明

### 索引目录
```bash
python index.py --root /path/to/your/project --db dirvec.db --model ./models/all-MiniLM-L6-v2 --exclude .git,node_modules --max-size 5 --batch-size 64 --prune
```

### 命令补全查询
```bash
python complete.py "vim src/uti"
```
