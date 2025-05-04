import gdown
from pipeline.data.constants import DRIVE_IDS

start = False
for key, value in DRIVE_IDS.items():
    if key == "hausa_eval_dataset.jsonl":
        start = True

    if not start:
        continue

    print(f"Key: {key}, Value: {value}")
    name = gdown.download(id=value, quiet=True)
    assert name == key, f"Downloaded file name {name} does not match expected name {key}"