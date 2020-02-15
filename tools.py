import os


def getRootPath():
    root_path = os.path.dirname(os.path.abspath(__file__))
    return root_path
