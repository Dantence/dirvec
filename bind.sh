#!/usr/bin/env bash
# 放入 ~/.bashrc 即可

_dirvec_complete() {
    local prefix="${READLINE_LINE}"
    local choice
    choice="$(/full/path/to/dirvec/venv/bin/python /full/path/to/dirvec/complete.py "$prefix" \
              | fzf --height=40% --reverse --query="$prefix" --prompt='DirVec> ')"
    [[ -n "$choice" ]] && READLINE_LINE="${choice}"
    READLINE_POINT=${#READLINE_LINE}
}

# Ctrl-h 触发
bind -m emacs-standard -x '"\C-h": _dirvec_complete'