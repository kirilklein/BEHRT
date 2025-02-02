#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(name='behrt',
      version='1.0',
      description='BEHRT original implementation at git@github.com:deepmedicine/BEHRT.git',
      long_description=readme,
      author='Kiril Klein',
      author_email='kikl@di.ku.dk',
      url="https://github.com/kirilklein/BEHRT.git",
      packages=['behrt'],
     )