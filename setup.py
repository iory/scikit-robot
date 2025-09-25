from __future__ import print_function

import os
import re
import sys

from setuptools import find_packages
from setuptools import setup


version = '0.2.21'


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

docs_install_requires = []
with open('requirements_docs.txt') as f:
    for line in f:
        req = line.split('#')[0].strip()
        docs_install_requires.append(req)

opt_install_requires = []
with open('requirements_opt.txt') as f:
    for line in f:
        req = line.split('#')[0].strip()
        opt_install_requires.append(req)


def remove_from_requirements(install_requires, remove_req):
    # If remove_req is "pyglet", requirement name with or without
    # version specification will be removed.
    # For example, either "pyglet" or "pyglet<2.0", "pyglet==1.4" will
    # be removed.
    delete_requirement = []
    for req in install_requires:
        req_without_version = re.split("[<>=]", req)[0]
        if req_without_version == remove_req:
            delete_requirement.append(req)
    assert len(delete_requirement) == 1, "expect only one match"
    install_requires.remove(delete_requirement.pop())


extra_all_requires = ['pybullet>=2.1.9']
if (sys.version_info.major > 2):
    # open3d doesn't support Python 3.13 yet
    if (sys.version_info.major, sys.version_info.minor) < (3, 13):
        extra_all_requires.append('open3d')
    extra_all_requires.append('fast-simplification')

# Python 2.7 and 3.4 support has been dropped from packages
# version lock those packages here so install succeeds
if (sys.version_info.major, sys.version_info.minor) <= (3, 7):
    # packages that no longer support old Python
    lock = [('pyglet', '1.4.10', install_requires),
            ('cvxopt', '1.2.7', opt_install_requires)]
    for name, version, requires in lock:
        remove_from_requirements(requires, name)

        # add working version locked requirements
        requires.append('{}=={}'.format(name, version))

extra_all_requires += docs_install_requires + opt_install_requires


console_scripts = ["visualize-urdf=skrobot.apps.visualize_urdf:main"]
if (sys.version_info.major, sys.version_info.minor) >= (3, 6):
    console_scripts.append(
        "convert-urdf-mesh=skrobot.apps.convert_urdf_mesh:main")
console_scripts.append("modularize-urdf=skrobot.apps.modularize_urdf:main")
console_scripts.append("change-urdf-root=skrobot.apps.change_urdf_root:main")
console_scripts.append("visualize-mesh=skrobot.apps.visualize_mesh:main")
console_scripts.append("urdf-hash=skrobot.apps.urdf_hash:main")


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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=find_packages(),
    package_data={'skrobot': listup_package_data()},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    entry_points={
        "console_scripts": console_scripts,
    },
    extras_require={
        'opt': opt_install_requires,
        'docs': docs_install_requires,
        'all': extra_all_requires,
    },
)
