import math
import os

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)