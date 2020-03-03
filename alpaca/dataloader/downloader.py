import os
from os import path

import requests


def download(cache_dir, name, url):
    if not path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_path = path.join(cache_dir, name)
    if not path.exists(data_path):
        print("Loading dataset...")
        response = requests.get(url)
        with open(data_path, 'wb') as f:
            f.write(response.content)

    return data_path