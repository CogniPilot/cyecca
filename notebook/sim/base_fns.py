import os
import pathlib

def get_local_folder():
    return pathlib.Path(__file__).parent.resolve()