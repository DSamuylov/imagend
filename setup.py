#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# read long description from README file:
with open("readme.md", "rb") as f:
    long_descr = f.read().decode('utf-8')

setup(
    version="0.1.0",
    name="imagend",
    description="Python package to visualize 3 and 4 dimensional image data.",
    long_description=long_descr,
    author="Denis K. Samuylov",
    author_email="denis.samuylov@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
