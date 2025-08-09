def parse_config(text: str) -> dict:
    return {"keys": [line.split("=")[0] for line in text.splitlines() if "=" in line]}
