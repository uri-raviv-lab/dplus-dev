_author__ = 'yaels'

import sys


import numpy
import os
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

GRID_CPP_DIR = "../../../Backend/Backend"
DPLUS_DIR = "../../../Common"

extra_compile_args = []
extra_link_args = []
if sys.platform == 'win32':
    extra_compile_args = ['/Ox']
    # extra_link_args = ['/debug']
elif sys.platform in ['linux', 'linux2']:
    extra_compile_args = ['-fPIC']

setup(
    name='CythonGrid',
    version='0.1',
    description='dplus c++ grid',
    author='Chelem',
    author_email='contact@chelem.co.il',
    ext_modules=cythonize(
        [Extension(
            "CythonGrid",
            ["CythonGrid.pyx"],
            language='c++',
            include_dirs=[GRID_CPP_DIR,DPLUS_DIR,
                          numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)]
    )
)
