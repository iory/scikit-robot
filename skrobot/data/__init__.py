import os.path as osp

import gdown


download_dir = osp.expanduser('~/.skrobot')
data_dir = osp.abspath(osp.dirname(__file__))


def bunny_objpath():
    target_path = osp.join(download_dir, 'mesh', 'bunny.obj')
    gdown.cached_download(
        url='https://drive.google.com/uc?id=18aAYzBglAGaSwes6oAENcufD8KC5XDph',
        path=target_path,
        md5='19bd31bde1fcf5242a8a82ed4ac03c72',
        quiet=True,
    )
    return target_path


def fetch_urdfpath():
    gdown.cached_download(
        url='https://drive.google.com/uc?id=1y7Jc3QoVW6J072CrSNupfKpyLp4NNxuH',
        path=osp.join(download_dir, 'fetch_description.tar.gz'),
        md5='fbe29ab5f3d029d165a625175b43a265',
        postprocess=gdown.extractall,
        quiet=True,
    )
    return osp.join(download_dir, 'fetch_description', 'fetch.urdf')


def kuka_urdfpath():
    return osp.join(data_dir, 'kuka_description', 'kuka.urdf')


def r2d2_urdfpath():
    return osp.join(data_dir, 'r2d2_description', 'r2d2.urdf')


def panda_urdfpath():
    gdown.cached_download(
        url='https://drive.google.com/uc?id=1h6ib9jpEUNa1xB2DNrnRQtqpSD2Rj9bz',
        path=osp.join(download_dir, 'franka_description.tar.gz'),
        md5='3de5bd15262b519e3beb88f1422032ac',
        postprocess=gdown.extractall,
        quiet=True,
    )
    return osp.join(download_dir, 'franka_description', 'panda.urdf')


def pr2_urdfpath():
    gdown.cached_download(
        url='https://drive.google.com/uc?id=1zy4C665o6efPko7eMk4XBdHbvgFfdC-6',
        path=osp.join(download_dir, 'pr2_description.tar.gz'),
        md5='e4fb915accdb3568a5524c92e9c35c9a',
        postprocess=gdown.extractall,
        quiet=True,
    )
    return osp.join(download_dir, 'pr2_description', 'pr2.urdf')
