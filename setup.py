#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
		'dask==2.30.0',
        'xarray==0.16.1',
        'pandas==1.1.3',
        'numba==0.51.2',
        'PySoundFile==0.9.0.post1',
        'dask_image==0.4.0',
        'matplotlib==3.3.1',
        'scipy==1.5.2',
        'numpy==1.19.1',
        'scikit_learn==0.23.2',
        'soundfile==0.10.3.post1',]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    name="ecosound", # Replace with your own username
    version="0.0.1", # PEP440
    author="Xavier Mouy",
    author_email="xaviermouy@uvic.ca",
    description="Python toolkit for analysing passive acoustic data",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
	keywords='ecosound',
	url="https://github.com/xaviermouy/ecosound",
    packages=find_packages(include=['ecosound', 'ecosound.*'],exclude=['docs', 'tests','resources']),
    install_requires=requirements,
	setup_requires=setup_requirements,
	test_suite='tests',
	tests_require=test_requirements,
    license="BSD license",
	classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License", 
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires='>=3.6.0,<3.8.0',
	zip_safe=False,
)