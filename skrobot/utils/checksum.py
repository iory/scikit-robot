import hashlib


def checksum_md5(filename, blocksize=8192):
    """Calculate md5sum.

    Parameters
    ----------
    filename : str or pathlib.Path
        input filename.
    blocksize : int
        MD5 has 128-byte digest blocks (default: 8192 is 128x64).

    Returns
    -------
    md5 : str
        calculated md5sum.
    """
    filename = str(filename)
    hash_factory = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(blocksize), b''):
            hash_factory.update(chunk)
    return hash_factory.hexdigest()
