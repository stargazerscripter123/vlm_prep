"""
NeMo AutoModel dataset builder for the converted package (referenced by the recipe as
`_target_: person_rock_data.make_dataset`; train.sh puts this directory on PYTHONPATH).

Returns a plain Python list of {"conversation": [...]} dicts. Never goes through HF datasets:
Arrow would add "image": None to text items and the Qwen3-VL template would render an extra
image placeholder for each of them.
"""
import json
from pathlib import Path


def make_dataset(path_or_dataset, split=None, **kwargs):
    rows = [json.loads(line) for line in Path(path_or_dataset).read_text().splitlines() if line.strip()]
    if not rows or "conversation" not in rows[0]:
        raise ValueError(f"{path_or_dataset}: expected rows with a 'conversation' key (run nemo/prep_data.py)")
    return rows
