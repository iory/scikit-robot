import os
import os.path as osp
import sys


data_dir = osp.abspath(osp.dirname(__file__))
_default_cache_dir = osp.expanduser('~/.skrobot')

_pooch = None


def _lazy_pooch():
    global _pooch
    if _pooch is None:
        import pooch
        _pooch = pooch
    return _pooch


def get_cache_dir():
    return os.environ.get('SKROBOT_CACHE_DIR', _default_cache_dir)


def _print_download_error(url, fname, cache_dir, exc):
    """Write a clearly-formatted, coloured download failure notice to stderr."""
    use_color = sys.stderr.isatty()
    red = '\033[31m' if use_color else ''
    bold = '\033[1m' if use_color else ''
    reset = '\033[0m' if use_color else ''
    target = osp.join(cache_dir, fname)
    sys.stderr.write(
        '\n{red}{bold}[skrobot] Failed to download robot-description data.{reset}\n'
        '  {bold}File:{reset}   {fname}\n'
        '  {bold}URL:{reset}    {url}\n'
        '  {bold}Target:{reset} {target}\n'
        '  {bold}Reason:{reset} {cls}: {msg}\n'
        '  {bold}Hint:{reset}   check network connection, DNS, proxy or firewall,\n'
        '          then retry. Cached files under {cache_dir} are reused if intact.\n\n'
        .format(
            red=red, bold=bold, reset=reset,
            fname=fname, url=url, target=target, cache_dir=cache_dir,
            cls=type(exc).__name__, msg=str(exc) or '(no detail)',
        )
    )


def _retrieve(url, fname, md5, extract=False):
    """Download (and optionally extract) a file under the cache dir.

    Parameters
    ----------
    url : str
        Download URL.
    fname : str
        File name relative to the cache directory. May contain subdirectories,
        e.g. ``"mesh/bunny.obj"``.
    md5 : str
        Expected MD5 hash of the downloaded file.
    extract : bool
        If True, the downloaded file is treated as a tar archive and extracted
        into the cache directory (the archive's parent directory).

    Notes
    -----
    A per-archive file lock is used so that concurrent processes (e.g. pytest
    workers running via pytest-xdist) cannot race on the same archive. Without
    the lock, one worker can observe a half-extracted tarball written by
    another worker and raise ``ValueError: ... is not a file``.

    On download failure the original exception is re-raised unchanged; a
    human-readable summary is additionally written to stderr so that users
    running scripts interactively are not left reading a bare stack trace.
    """
    from filelock import FileLock

    pooch = _lazy_pooch()
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    lock_path = osp.join(cache_dir, '.{}.lock'.format(fname.replace(os.sep, '_')))
    processor = pooch.Untar(extract_dir='.') if extract else None
    try:
        with FileLock(lock_path):
            pooch.retrieve(
                url=url,
                known_hash='md5:{}'.format(md5),
                path=cache_dir,
                fname=fname,
                processor=processor,
            )
    except Exception as e:
        _print_download_error(url, fname, cache_dir, e)
        raise


def bunny_objpath():
    path = osp.join(get_cache_dir(), 'mesh', 'bunny.obj')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://raw.githubusercontent.com/iory/scikit-robot-models/main/data/bunny.obj',  # NOQA
        fname=osp.join('mesh', 'bunny.obj'),
        md5='19bd31bde1fcf5242a8a82ed4ac03c72',
    )
    return path


def fetch_urdfpath():
    path = osp.join(get_cache_dir(), 'fetch_description', 'fetch.urdf')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/main/fetch_description.tar.gz',  # NOQA
        fname='fetch_description.tar.gz',
        md5='fbe29ab5f3d029d165a625175b43a265',
        extract=True,
    )
    return path


def griphis_urdfpath():
    path = osp.join(get_cache_dir(),
                    'griphis_description', 'urdf', 'griphis.urdf')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/main/griphis_description.tar.gz',  # NOQA
        fname='griphis_description.tar.gz',
        md5='e74a5e9887b51a918227b4fc185a6a33',
        extract=True,
    )
    return path


def kuka_urdfpath():
    return osp.join(data_dir, 'kuka_description', 'kuka.urdf')


def nextage_urdfpath():
    path = osp.join(get_cache_dir(), 'nextage_description', 'urdf', 'NextageOpen.urdf')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/refs/heads/main/nextage_description.tar.gz',  # NOQA
        fname='nextage_description.tar.gz',
        md5='9805ac9cd97b67056dde31aa88762ec7',
        extract=True,
    )
    return path


def panda_urdfpath():
    path = osp.join(get_cache_dir(), 'franka_description', 'panda.urdf')
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/main/franka_description.tar.gz',  # NOQA
        fname='franka_description.tar.gz',
        md5='3de5bd15262b519e3beb88f1422032ac',
        extract=True,
    )
    return path


def pr2_urdfpath():
    path = osp.join(get_cache_dir(), 'pr2_description', 'pr2.urdf')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/main/pr2_description.tar.gz',  # NOQA
        fname='pr2_description.tar.gz',
        md5='6e6e2d4f38e2c5c0a93f44b67962b98a',
        extract=True,
    )
    return path


def r8_6_urdfpath():
    return osp.join(data_dir, 'robot_descriptions', 'urdf', 'r8_6.urdf')


def rover_armed_tycoon_urdfpath():
    path = osp.join(get_cache_dir(),
                    'tycoon_description', 'urdf', 'tycoon_arm_assem.urdf')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/main/tycoon_description.tar.gz',  # NOQA
        fname='tycoon_description.tar.gz',
        md5='166ab45ed6e9b24bfb8d0f7a7eb19186',
        extract=True,
    )
    return path


def differential_wrist_sample_urdfpath():
    path = osp.join(get_cache_dir(),
                    'differential_wrist_sample', 'urdf',
                    'differential_wrist.urdf')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/main/differential_wrist_sample.tar.gz',  # NOQA
        fname='differential_wrist_sample.tar.gz',
        md5='d8b3c1b4fef700d6e772265fe68867b5',
        extract=True,
    )
    return path


def differential_wrist_sample_joint_limit_table_path():
    path = osp.join(get_cache_dir(),
                    'differential_wrist_sample', 'config',
                    'joint_limit_table.yaml')
    if osp.exists(path):
        return path
    _retrieve(
        url='https://github.com/iory/scikit-robot-models/raw/main/differential_wrist_sample.tar.gz',  # NOQA
        fname='differential_wrist_sample.tar.gz',
        md5='d8b3c1b4fef700d6e772265fe68867b5',
        extract=True,
    )
    return path
