from __future__ import print_function

import distutils.spawn
import os
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '0.0.3'


if sys.argv[-1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist',
        'twine upload dist/scikit-robot-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


def listup_package_data():
    data_files = []
    for root, _, files in os.walk('skrobot/data'):
        for filename in files:
            data_files.append(
                os.path.join(
                    root[len('skrobot/'):],
                    filename))
    return data_files


setup_requires = []

with open('requirements.txt') as f:
    install_requires = []
    for line in f:
        req = line.split('#')[0].strip()
        install_requires.append(req)

# Python 2.7 and 3.4 support has been dropped from packages
# version lock those packages here so install succeeds
if (sys.version_info.major, sys.version_info.minor) <= (3, 4):
    # packages that no longer support old Python
    lock = [('pyglet', '1.4.10')]
    for name, version in lock:
        # remove version-free requirements
        install_requires.remove(name)
        # add working version locked requirements
        install_requires.append('{}=={}'.format(name, version))

setup(
    name='scikit-robot',
    version=version,
    description='A Flexible Framework for Robot Control in Python',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/scikit-robot',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=find_packages(),
    package_data={'skrobot': listup_package_data()},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require={
        'all': ['pybullet>=2.1.9'],
    },
)
