#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="missing_data_imputation",
    version="1.0.0",
    description="Methods and experiments for data imputation.",
    author=["Arber Qoku", "Alexander Hanf"],
    author_email=["arber.qoku@campus.lmu.de", "a.hanf@campus.lmu.de"],
    url="https://github.com/arberqoku/missing-data-imputation",
    packages=find_packages(),
    install_requires=[
        "fancyimpute>=0.5.4",
        "missingpy>= 0.2.0",
        "pandas>=0.25.3",
        "pandas-profiling>=2.3.0",
        "scikit-learn>=0.21.3",
    ],
    include_package_data=True,
)
