import os
import os.path as osp

from packaging.version import Version


data_dir = osp.abspath(osp.dirname(__file__))
_default_cache_dir = osp.expanduser('~/.skrobot')

_gdown = None
_gdown_version = None


def _lazy_gdown():
    global _gdown
    if _gdown is None:
        import gdown
        _gdown = gdown
    return _gdown


def _lazy_gdown_version():
    global _gdown_version
    if _gdown_version is None:
        from skrobot import determine_version
        _gdown_version = determine_version('gdown')

    return _gdown_version


def get_cache_dir():
    return os.environ.get('SKROBOT_CACHE_DIR', _default_cache_dir)


def _download(url, path, md5, postprocess=None, quiet=False):
    gdown = _lazy_gdown()
    if postprocess == 'extractall':
        postprocess = gdown.extractall
    if Version(_lazy_gdown_version()) < Version("5.1.0"):
        gdown.cached_download(
            url=url, path=path, md5=md5, quiet=quiet,
            postprocess=postprocess,
        )
    else:
        gdown.cached_download(
            url=url,
            path=path,
            hash="md5:{}".format(md5),
            quiet=quiet,
            postprocess=postprocess,
        )


def bunny_objpath():
    path = osp.join(get_cache_dir(), 'mesh', 'bunny.obj')
    if osp.exists(path):
        return path
    _download(
        url='https://raw.githubusercontent.com/iory/scikit-robot-models/main/data/bunny.obj',  # NOQA
        path=path,
        md5='19bd31bde1fcf5242a8a82ed4ac03c72',
        quiet=True,
    )
    return path


def fetch_urdfpath():
    path = osp.join(get_cache_dir(),
                    'fetch_description', 'fetch.urdf')
    if osp.exists(path):
        return path
    _download(
        url='https://github.com/iory/scikit-robot-models/raw/main/fetch_description.tar.gz',  # NOQA
        path=osp.join(get_cache_dir(), 'fetch_description.tar.gz'),
        md5='fbe29ab5f3d029d165a625175b43a265',
        postprocess='extractall',
        quiet=True,
    )
    return path


def kuka_urdfpath():
    return osp.join(data_dir, 'kuka_description', 'kuka.urdf')


def nextage_urdfpath():
    path = osp.join(get_cache_dir(), 'nextage_description', 'urdf', 'NextageOpen.urdf')
    if osp.exists(path):
        return path
    _download(
        url='https://github.com/iory/scikit-robot-models/raw/refs/heads/main/nextage_description.tar.gz',  # NOQA
        path=osp.join(get_cache_dir(), 'nextage_description.tar.gz'),
        md5='9805ac9cd97b67056dde31aa88762ec7',
        postprocess='extractall',
        quiet=True,
    )
    return path


def panda_urdfpath():
    path = osp.join(get_cache_dir(),
                    'franka_description', 'panda.urdf')
    _download(
        url='https://github.com/iory/scikit-robot-models/raw/main/franka_description.tar.gz',  # NOQA
        path=osp.join(get_cache_dir(), 'franka_description.tar.gz'),
        md5='3de5bd15262b519e3beb88f1422032ac',
        postprocess='extractall',
        quiet=True,
    )
    return path


def pr2_urdfpath():
    path = osp.join(get_cache_dir(), 'pr2_description', 'pr2.urdf')
    if osp.exists(path):
        return path
    _download(
        url='https://github.com/iory/scikit-robot-models/raw/main/pr2_description.tar.gz',  # NOQA
        path=osp.join(get_cache_dir(), 'pr2_description.tar.gz'),
        md5='6e6e2d4f38e2c5c0a93f44b67962b98a',
        postprocess='extractall',
        quiet=True,
    )
    return path


def r8_6_urdfpath():
    return osp.join(data_dir, 'robot_descriptions', 'urdf', 'r8_6.urdf')


def rover_armed_tycoon_urdfpath():
    path = osp.join(get_cache_dir(),
                    'tycoon_description', 'urdf', 'tycoon_arm_assem.urdf')
    if osp.exists(path):
        return path
    _download(
        url='https://github.com/iory/scikit-robot-models/raw/main/tycoon_description.tar.gz',
        path=osp.join(get_cache_dir(), 'tycoon_description.tar.gz'),
        md5='166ab45ed6e9b24bfb8d0f7a7eb19186',
        postprocess='extractall',
        quiet=True,
    )
    return path
