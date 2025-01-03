#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#    history = history_file.read()

requirements = ["setuptools>=61.0", "astropy", "numpy"]


# build-backend = "setuptools.build_meta"

test_requirements = [
    "pytest>=3",
]

setup(
    author="Henry Best",
    author_email="hbest@gradcenter.cuny.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Amoeba",
    install_requires=requirements,
    license="MIT license",
    # long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="amoeba",
    name="amoeba",
    packages=find_packages(),
    test_suite="amoeba/tests",
    tests_require=test_requirements,
    url="https://github.com/Henry-Best-01/Amoeba",
    version="0.1.0",
    zip_safe=False,
)
