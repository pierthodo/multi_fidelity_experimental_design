import re
from setuptools import find_packages, setup


setup(
    name="mfed",
    version="0.1",
    description="A package for multi-fidelity experimental design on PDE",
    author="Pierre Thodoroff",
    author_email="pierthodo@gmail.com",
    packages=find_packages(include=["mfed", "mfed.*"]),
)
