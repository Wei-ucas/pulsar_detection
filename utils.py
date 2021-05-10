import logging
import time
import os
import numpy as np
import os.path as osp
import tempfile
import platform
import sys
from importlib import import_module
import shutil


def create_logger(name, save_path):
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_path, time.strftime('%Y%m%d%H%M%S', time.localtime()) + '_{}.log'.format(name)),
        filemode='a',
        format='%(asctime)s -  %(levelname)s: %(message)s'
    )
    chlr = logging.StreamHandler()
    chlr.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    logger = logging.getLogger(name)
    logger.addHandler(chlr)
    return logger


def config_from_py(filename):
    '''
    use .py save configs, codes come from mmcv packages
    :param filename: .py file
    :return: configs
    '''
    filename = osp.abspath(osp.expanduser(filename))
    # check_file_exist(filename)
    if not osp.exists(filename):
        raise FileNotFoundError('file "{}" does not exist'.format(filename))
    fileExtname = osp.splitext(filename)[1]
    if fileExtname != '.py':
        raise IOError('Only py type are supported')
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=fileExtname)
        if platform.system() == 'Windows':
            temp_config_file.close()
        shutil.copyfile(filename, temp_config_file.name)
        temp_config_name = osp.basename(temp_config_file.name)
        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        del sys.modules[temp_module_name]

    return mod


def tnormalize(data):
    for i in range(data.shape[0]):
        mean = np.median(data[i, :])
        var = np.std(data[i, :])
        if var > 1:
            data[i, :] = (data[i, :] - mean) / var
        else:
            data[i, :] = (data[i, :] - mean)
    return data


def fnormalize(data):
    global_max = np.maximum.reduce(np.maximum.reduce(data))
    min_parts = np.minimum.reduce(data, 1)
    data = (data - min_parts[:, np.newaxis]) / global_max
    return data


def data2pic(data, f_downsamp, t_downsamp, downsamp=True):
    if downsamp == True:
        l, m = data.shape
        data = data.reshape((int(l / f_downsamp), f_downsamp,
                             int(m / t_downsamp), t_downsamp)).sum(axis=1).sum(axis=2)
    else:
        l, m = data.shape
        data = data.reshape((int(l / f_downsamp), f_downsamp, m)).sum(axis=1)
    data = tnormalize(data)
    dst = data

    dst = fnormalize(dst)
    chan_mean = dst.mean(axis=1)
    dst = (dst.T - chan_mean).T
    mi = np.min(dst)
    ma = np.max(dst)
    dst = (dst - mi)/(ma - mi) * 255
    dst = 255 - dst
    # dst = dst.clip(0,255)
    return dst


def create_folder(path):
    root_path = os.path.split(path)[0]
    if not os.path.exists(root_path):
        create_folder(root_path)
    os.mkdir(path)


if __name__ == '__main__':
    config_from_py('configs/default.py')
