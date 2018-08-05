#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import find_packages
from setuptools import setup


def listup_package_data():
    data_files = []
    for root, _, files in os.walk('robot/models_robot'):
        for filename in files:
            data_files.append(
                os.path.join(
                    root[len('robot/'):],
                    filename))
    return data_files


setup_requires = []
install_requires = [
    'cached-property',
    'cvxopt',
    'future',
    'numpy>=1.9.0',
    'ordered_set',
    'pybullet',
    'quadprog',
    'sympy',
]

setup(
    name='robot',
    version='0.0.1',
    description='A python robot programming library',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/robot',
    license='MIT License',
    packages=find_packages(),
    package_data={'robot': listup_package_data()},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
