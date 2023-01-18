from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='libraries',
    version='0.1',
    packages=find_packages(where='libraries'),
    package_dir={'': 'libraries'},
    py_modules=[splitext(basename(path))[0] for path in glob('libraries/*.py')],
    description='Uses neural networks  to make predictions ground-state properties of NN scattering using NCSM data',
    author='Alberto J. Garcia',
    author_email='garcia.823@osu.edu',
    zip_safe=False
)
