import json
from pathlib import Path

a = Path('./name_info.json')
b = json.loads(a.read_text())
for i in b:
    Path(i).rename(b[i][0])
