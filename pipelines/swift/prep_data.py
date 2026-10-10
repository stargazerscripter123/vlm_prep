"""
ms-swift trains on the data package as-is (train.jsonl + ROOT_IMAGE_DIR). This only materializes
derived splits that are not package files -- the fixed audit32 subset -- in the same ms-swift
messages+images format, with absolute image paths.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from data_lib import IMAGE_TAG, prep_main  # noqa: E402


def convert(sample):
    return {"messages": [{"role": "user", "content": IMAGE_TAG + sample.prompt},
                         {"role": "assistant", "content": sample.answer}],
            "images": [sample.image]}


if __name__ == "__main__":
    prep_main("swift", convert)
