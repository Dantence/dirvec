import json
from utils.helpers import slugify
def main():
    print("hello dirvec")
    data = {"task": "index", "module": "semantic search"}
    print(json.dumps(data))
if __name__ == "__main__":
    main()
