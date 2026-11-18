#!/usr/bin/env python3
"""Setup script for aetheris-microbrains."""
from setuptools import setup, find_packages

setup(
    name="aetheris-microbrains",
    version="0.7.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["*.json", "*.npz", "*.md"],
    },
)
