"""
Package -> Axolotl multimodal chat_template rows. Images are referenced inside the user message
by ABSOLUTE path (Axolotl resolves relative paths against the CWD); no top-level "images"
column (the processing strategy would insert the image a second time).

  {"messages": [{"role": "user", "content": [{"type": "image", "path": "/abs/path.jpg"},
                                             {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [{"type": "text", "text": ANSWER}]}]}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from data_lib import prep_main  # noqa: E402


def convert(sample):
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "path": sample.image},
                                     {"type": "text", "text": sample.prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]}]}


if __name__ == "__main__":
    prep_main("axolotl", convert)
