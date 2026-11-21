"""Index the byte offsets in the already verified ImageNet-100 PNG archive."""
import json
import sys
import tarfile
from pathlib import Path

archive = Path(sys.argv[1])
splits = {"train": [], "validation": []}
with tarfile.open(archive, "r:") as handle:
    for entry in handle:
        parts = entry.name.split("/")
        if entry.isfile() and len(parts) == 4 and parts[0] == "rgb256" and parts[1] in splits:
            if entry.name.lower().endswith(".png"):
                splits[parts[1]].append([entry.name, entry.offset_data, entry.size])
            handle.members.clear()

assert [len(splits[key]) for key in splits] == [126689, 5000]
classes = [{row[0].split("/")[2] for row in rows} for rows in splits.values()]
assert len(classes[0]) == 100 and classes[0] == classes[1]
path = archive.with_suffix(".index.json")
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps({"archive_bytes": archive.stat().st_size, "splits": splits}))
temp.replace(path)
print("INDEX_READY train=126689 validation=5000 classes=100", flush=True)
