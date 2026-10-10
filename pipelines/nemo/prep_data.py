"""
Package -> NeMo AutoModel VLM "conversation" rows (the format its dataset builders return),
image as an absolute path string (loaded by the HF processor inside qwen2_5_collate_fn).
Loaded by person_rock_data.make_dataset as a plain Python list (never via HF datasets: Arrow
would add "image": None to text items and the Qwen3-VL template would render extra image pads).

  {"conversation": [{"role": "user", "content": [{"type": "image", "image": "/abs/path.jpg"},
                                                 {"type": "text", "text": PROMPT}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ANSWER}]}]}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from data_lib import prep_main  # noqa: E402


def convert(sample):
    return {"conversation": [
        {"role": "user", "content": [{"type": "image", "image": sample.image},
                                     {"type": "text", "text": sample.prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]}]}


if __name__ == "__main__":
    prep_main("nemo", convert)
