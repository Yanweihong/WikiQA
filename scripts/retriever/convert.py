import os

import win32api
from tqdm import tqdm

import tools


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield f, os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def t2s(input_file, output_file):
    """Convert traditional Chinese to simplified Chinese"""
    param_i = '-i ' + input_file + ' '
    param_o = '-o ' + output_file + ' '
    param_c = '-c D:\\workspace\\FinalProject\\opencc-1.0.4\\share\\opencc\\t2s.json'
    param = param_i + param_o + param_c
    opencc_exe = 'D:\\workspace\\FinalProject\\opencc-1.0.4\\bin\\opencc.exe'
    win32api.ShellExecute(0, 'open', opencc_exe, param, '', 0)


def convert_all(data_path):
    """Convert all files from traditional Chinese to simplified Chinese"""
    root_path = tools.getRootPath()
    files = [f for f in iter_files(data_path)]
    for f in tqdm(files):

        file_name = f[0]  # wiki_00
        input_file_path = os.path.join(root_path, f[1])  # D:\workspace\pycharm\myQA\data\AA\wiki_01
        output_file_path = input_file_path.replace('data', 'data1')  # D:\workspace\pycharm\myQA\data1\AA\wiki_01

        path = os.path.split(output_file_path)[0]  # D:\workspace\pycharm\myQA\data1\AA
        if not os.path.exists(path):
            os.makedirs(path)

        with open(output_file_path, 'w'):
            t2s(input_file_path, output_file_path)


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    convert_all(r'D:\workspace\pycharm\myQA\data')
