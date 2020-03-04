import os
from os.path import expanduser
from pathlib import Path


home = expanduser("~")
DATA_DIR = Path(home)/'.local'/'share'/'alpaca'


if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
