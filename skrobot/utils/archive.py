from pathlib import Path
import tarfile


def make_tarfile(source_dir, output_filename=None, arcname=None):
    """Create a tar.gz archive from a directory.

    Parameters
    ----------
    source_dir : str or pathlib.Path
        The directory to archive
    output_filename : str or pathlib.Path, optional
        Output tar.gz filename. If None, uses source_dir.tar.gz
    arcname : str, optional
        Alternative name for the top-level directory in the archive

    Returns
    -------
    str
        Path to created tar.gz file
    """
    source_dir = Path(source_dir)
    if output_filename is None:
        output_filename = source_dir.with_suffix('.tar.gz')
    else:
        output_filename = Path(output_filename)
        if not str(output_filename).endswith('.tar.gz'):
            output_filename = output_filename.with_suffix('.tar.gz')

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=arcname or source_dir.name)

    return str(output_filename)
