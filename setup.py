#!/usr/bin/env python3

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='nerf',
      version='0.1.0',
      description='Neural Radiance Fields',
      author='Alessandro Benetti',
      license='Apache 2.0',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['nerf'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software"
      ],
      install_requires=['numpy', 'requests', 'pillow', 'tqdm', 'networkx'],
      python_requires='>=3.8',
      extras_require={
        'linting': [
            "flake8",
            "pylint",
            "mypy",
            "pre-commit",
        ],
      },
      include_package_data=True)