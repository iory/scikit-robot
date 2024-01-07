import os
import os.path as osp

import gdown


data_dir = osp.abspath(osp.dirname(__file__))
_default_cache_dir = osp.expanduser('~/.skrobot')


def get_cache_dir():
    return os.environ.get('SKROBOT_CACHE_DIR', _default_cache_dir)


def bunny_objpath():
    target_path = osp.join(get_cache_dir(), 'mesh', 'bunny.obj')
    gdown.cached_download(
        url='https://raw.githubusercontent.com/iory/scikit-robot-models/master/data/bunny.obj',  # NOQA
        path=target_path,
        md5='19bd31bde1fcf5242a8a82ed4ac03c72',
        quiet=True,
    )
    return target_path


def fetch_urdfpath():
    gdown.cached_download(
        url='https://github.com/iory/scikit-robot-models/raw/master/fetch_description.tar.gz',  # NOQA
        path=osp.join(get_cache_dir(), 'fetch_description.tar.gz'),
        md5='fbe29ab5f3d029d165a625175b43a265',
        postprocess=gdown.extractall,
        quiet=True,
    )
    return osp.join(get_cache_dir(), 'fetch_description', 'fetch.urdf')


def kuka_urdfpath():
    return osp.join(data_dir, 'kuka_description', 'kuka.urdf')


def panda_urdfpath():
    gdown.cached_download(
        url='https://github.com/iory/scikit-robot-models/raw/master/franka_description.tar.gz',  # NOQA
        path=osp.join(get_cache_dir(), 'franka_description.tar.gz'),
        md5='3de5bd15262b519e3beb88f1422032ac',
        postprocess=gdown.extractall,
        quiet=True,
    )
    return osp.join(get_cache_dir(), 'franka_description', 'panda.urdf')


def pr2_urdfpath():
    gdown.cached_download(
        url='https://github.com/iory/scikit-robot-models/raw/master/pr2_description.tar.gz',  # NOQA
        path=osp.join(get_cache_dir(), 'pr2_description.tar.gz'),
        md5='6e6e2d4f38e2c5c0a93f44b67962b98a',
        postprocess=gdown.extractall,
        quiet=True,
    )
    return osp.join(get_cache_dir(), 'pr2_description', 'pr2.urdf')
