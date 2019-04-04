#!/usr/bin/env python

import os

from setuptools import find_packages
from setuptools import setup


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
install_requires = [
    'cached-property',
    'cvxopt',
    'future',
    'lxml',
    'networkx==2.2.0',
    'numpy>=1.9.0',
    'ordered_set',
    'pillow',
    'pybullet>=2.1.9',
    'pyyaml',
    'quadprog',
    'six',
    'sympy',
    'trimesh',
]

setup(
    name='skrobot',
    version='0.0.1',
    description='A python robot programming library',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/scikit-robot',
    license='MIT License',
    packages=find_packages(),
    package_data={'skrobot': listup_package_data()},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
