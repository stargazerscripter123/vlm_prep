"""
Package -> TRL conversational prompt/completion rows (vision). Loss is on the completion only
(completion_only_loss). The prompt is a plain string: TRL inserts the image placeholder before
the text, matching the package's "<image>"-first layout. Images: absolute path strings.

  {"prompt": [{"role": "user", "content": PROMPT}],
   "completion": [{"role": "assistant", "content": ANSWER}],
   "images": ["/abs/path.jpg"]}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from data_lib import prep_main  # noqa: E402


def convert(sample):
    return {"prompt": [{"role": "user", "content": sample.prompt}],
            "completion": [{"role": "assistant", "content": sample.answer}],
            "images": [sample.image]}


if __name__ == "__main__":
    prep_main("trl", convert)
