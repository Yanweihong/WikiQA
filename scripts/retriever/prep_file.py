import codecs
import os
import re
import tools
from tqdm import tqdm


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield f, os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def preprocess(input_file,output_file):
    """remove useless special character"""
    p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
    p2 = re.compile(r'[（][，；。？！\s]∗[）]')
    p3 = re.compile(r'[「『]')
    p4 = re.compile(r'[」』]')
    with codecs.open(output_file, 'w', 'utf-8') as outf:
        with codecs.open(input_file, 'r', 'utf-8') as inf:
            for line in inf:
                line = p1.sub(r'\2', line)
                line = p2.sub(r'', line)
                line = p3.sub(r'“', line)
                line = p4.sub(r'”', line)
                outf.write(line)

def convert_all(data_path):
    """preprocess all files"""
    root_path = tools.getRootPath()
    files = [f for f in iter_files(data_path)]
    for f in tqdm(files):

        file_name = f[0]  # wiki_00
        input_file_path = os.path.join(root_path, f[1])  # D:\workspace\pycharm\myQA\data1\AA\wiki_01
        output_file_path = input_file_path.replace('data1', 'data2')  # D:\workspace\pycharm\myQA\data2\AA\wiki_01

        path = os.path.split(output_file_path)[0]  # D:\workspace\pycharm\myQA\data2\AA
        if not os.path.exists(path):
            os.makedirs(path)

        with open(output_file_path, 'w'):
            preprocess(input_file_path, output_file_path)


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    convert_all(r'D:\workspace\pycharm\myQA\data1')
