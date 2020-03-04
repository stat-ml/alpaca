import os


DATA_DIR = '~/.local/share/alpaca'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

