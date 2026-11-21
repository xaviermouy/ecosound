#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_namespace_packages

exec(open("ecosound/_version.py").read())

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# automatically captured required modules for install_requires in requirements.txt
with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().split('\n')

setup(
    name="ecosound", # Replace with your own username
    version=__version__,
	author="Xavier Mouy",
    author_email="xaviermouy@uvic.ca",
    description="Python toolkit for analysing passive acoustic data",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/x-rst",
    keywords='ecosound',
	url="https://github.com/xaviermouy/ecosound",    
    include_package_data=True,
	#package_data={'': ['core/*.json']},
    packages=find_namespace_packages(include=['ecosound', 'ecosound.*'],exclude=['docs', 'tests','resources']),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ecosound = ecosound.cli:main",
        ],
    },
    license="BSD-3-Clause",
	classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires='>=3.6.0',
	zip_safe=False,
)