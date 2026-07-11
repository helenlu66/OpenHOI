import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.path_config import grab_data_root

file_path = grab_data_root() / "text.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

cnt = 0
for i in data.items():
    print(i)
    cnt += 1
print(cnt, len(data))
