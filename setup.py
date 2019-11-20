#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="missing_data_imputation",
    version="1.0.0",
    description="Methods and experiments for data imputation.",
    author=["Arber Qoku", "Alexander Hanf"],
    author_email=["arber.qoku@campus.lmu.de", ""],
    url="https://github.com/arberqoku/missing-data-imputation",
    packages=find_packages(),
    install_requires=[
        "autoimpute>=0.11.6",
        "datawig>=0.1.10",
        "fancyimpute>=0.5.4",
        "impyute>=0.0.8",
        "missingno>= 0.4.2",
        "missingpy>= 0.2.0",
        "pandas>=0.25.3",
        "pandas-profiling>=2.3.0",
        "scikit-learn>=0.21.3",
    ],
    include_package_data=True,
)
