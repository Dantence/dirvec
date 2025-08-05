from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                  local_dir="./models/all-MiniLM-L6-v2",
                  local_dir_use_symlinks=False)