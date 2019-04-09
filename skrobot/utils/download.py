import filelock
import hashlib
import os
import shutil
import tempfile

import gdown


_default_cache_dir = os.environ.get(
    'SKROBOT_CACHE_ROOT',
    os.path.expanduser('~/.skrobot'))


def cached_gdown_download(url, target_path=None, md5sum=None):
    try:
        os.makedirs(_default_cache_dir)
    except OSError:
        if not os.path.exists(_default_cache_dir) and \
           not os.path.isdir(_default_cache_dir):
            raise
    cache_root = os.path.join(_default_cache_dir, '_dl_cache')
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.exists(cache_root) and \
           not os.path.isdir(_default_cache_dir):
            raise

    urlhash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if target_path is None:
        target_path = os.path.join(cache_root, urlhash)
    lock_path = target_path + ".lock"
    print(lock_path)

    with filelock.FileLock(lock_path):
        if os.path.exists(target_path):
            return target_path

    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = os.path.join(temp_root, 'download.cache')
        gdown.download(url, temp_path, quiet=False)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, target_path)
    finally:
        shutil.rmtree(temp_root)

    if md5sum is not None:
        if check_md5sum(target_path, md5sum) is False:
            raise ValueError("Downloaded/Cached file's md5sum is odd.")

    return target_path


def check_md5sum(path, md5):
    """Check md5sum"""
    # validate md5 string length if it is specified
    if md5 and len(md5) != 32:
        raise ValueError('md5 must be 32 charactors\n'
                         'actual: {} ({} charactors)'.format(md5, len(md5)))
    print('[%s] Checking md5sum (%s)' % (path, md5))
    is_same = md5sum(path) == md5
    print('[%s] Finished checking md5sum' % path)
    return is_same


def md5sum(path, chunks=8192):
    """Return md5sum.

    Reading 8192(default) byte chunks

    """
    m = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            data = f.read(chunks)
            if not data:
                break
            m.update(data)
    return m.hexdigest()
