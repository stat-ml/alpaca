import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()


setuptools.setup(
    name="alpaca-ml",
    version="0.0.2",
    author="Maxim Panov and Evgenii Tsymbalov and Kirill Fedyanin",
    author_email="k.fedyanin@skoltech.ru",
    description="Active learning utilities for machine learning applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/premolab/alpaca",
    packages=setuptools.find_packages(),
    license="Apache License 2.0",
    install_requires=load_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)