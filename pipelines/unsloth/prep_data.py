"""
Package -> Unsloth vision rows. The image goes in the top-level "images" column (decoded as-is
by UnslothVisionDataCollator; message-embedded paths would go through Unsloth's own Qwen2-constant
resize), the user content keeps a bare {"type": "image"} placeholder before the text.

  {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [{"type": "text", "text": ANSWER}]}],
   "images": ["/abs/path.jpg"]}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from data_lib import prep_main  # noqa: E402


def convert(sample):
    return {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample.prompt}]},
                         {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]}],
            "images": [sample.image]}


if __name__ == "__main__":
    prep_main("unsloth", convert)
