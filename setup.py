"""Build script for Chess C extension.

Usage (from project root):
    python chess/setup.py build_ext --inplace
    pip install -e chess/

Usage (from chess/ directory):
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import numpy as np
import os
import sys

# Find pufferlib ocean include path (for env_binding.h)
import pufferlib
pufferlib_ocean = os.path.join(os.path.dirname(pufferlib.__file__), 'ocean')

# Resolve paths relative to this file's directory
here = os.path.dirname(os.path.abspath(__file__))

extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-std=c11']
extra_link_args = []
if sys.platform == 'darwin':
    extra_compile_args.append('-stdlib=libc++')
elif sys.platform.startswith('linux'):
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

if os.environ.get('DEBUG'):
    extra_compile_args = ['-g', '-O0', '-fsanitize=address', '-std=c11']
    extra_link_args = []

extension = Extension(
    'csrc.binding',
    sources=[os.path.join(here, 'csrc', 'binding.c')],
    include_dirs=[
        os.path.join(here, 'csrc'),
        np.get_include(),
        pufferlib_ocean,
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)

setup(
    name='chess-env',
    version='0.1.0',
    description='High-performance Chess self-play RL environment',
    ext_modules=[extension],
    install_requires=[
        'numpy>=1.20.0',
        'gymnasium>=0.29.0',
        'pufferlib>=2.0.0',
        'torch>=2.0.0',
    ],
)
