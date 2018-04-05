#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import setup


setup_requires = []
install_requires = [
    'cvxopt',
    'numpy>=1.9.0',
    'ordered_set',
    'pybullet',
    'pyyaml',
    'quadprog',
]

setup(
    name='robot',
    version='0.0.1',
    description='A python robot programming library',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/robot',
    license='MIT License',
    packages=['robot',
              'robot.misc',
              'robot.optimizers',
              'robot.utils',
              ],
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
