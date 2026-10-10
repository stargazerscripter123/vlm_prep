"""render_template.py <template> <output>: fill string.Template placeholders from the environment.
Fails (and writes nothing) if any placeholder is missing."""
import os
import string
import sys

text = string.Template(open(sys.argv[1]).read()).substitute(os.environ)
with open(sys.argv[2], "w") as f:
    f.write(text)
