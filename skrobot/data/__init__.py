import os.path as osp


data_dir = osp.abspath(osp.dirname(__file__))


def fetch_urdfpath():
    return osp.join(data_dir, 'fetch_description/fetch.urdf')


def kuka_urdfpath():
    return osp.join(data_dir, 'kuka_description/kuka.urdf')
