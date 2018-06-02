#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import find_packages
from setuptools import setup


setup_requires = []
install_requires = [
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
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
