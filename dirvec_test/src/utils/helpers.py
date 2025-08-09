def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s)
