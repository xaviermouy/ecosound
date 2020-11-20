import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecosound", # Replace with your own username
    version="0.0.1", # PEP440
    author="Xavier Mouy",
    author_email="xaviermouy@uvic.ca",
    description="Python toolkit for analysing passive acoustic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xaviermouy/ecosound",
    packages=setuptools.find_packages(exclude=['docs', 'tests','resources']),
    install_requires=[
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
        'soundfile==0.10.3.post1',
        ],
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
)