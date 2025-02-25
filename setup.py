# coding=utf-8

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

reqs = [
    "scikit-learn>=1.1.0, <=1.2.1",
    "pandas>=1.0.5, <=1.3.5",
    "numpy>=1.18.5, <=1.21.6",
    "scipy>=1.4.1, <=1.7.3",
    "attrs>=19.3.0",
    "flatten_dict>=0.3.0",
    "tensorboardX>=2.1.0",
    "gcsfs==0.6.2",
    "google-auth>=2.15.0",
    "fsspec <= 2023.1.0",
    "wandb"
]

setup(
    name="allRank",
    version="1.4.3",
    description="allRank is a framework for training learning-to-rank neural models",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache 2",
    url="https://github.com/allegro/allRank",
    install_requires=reqs,
    author_email="allrank@allegro.pl",
    packages=find_packages(exclude=["tests"]),
    package_data={"allrank": ["config.json"]},
    entry_points={"console_scripts": ['allRank = allrank.main:run']},
    zip_safe=False,
)
