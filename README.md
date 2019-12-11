# Missing Data Imputation

This repository provides an overview of most common missing data imputation strategies along with a notebook demo showcasing different python implementations of these strategies.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites
* Minimal requirements
    * [Python 3.X (**64 bit**)](https://www.python.org/downloads/)
    * [Python package manager (PIP)](https://pypi.org/project/pip/)
* Recommended
    * [Conda](https://www.anaconda.com/distribution/) for python 3.7

### Installing

* Clone repository 
```bash
git clone git@github.com:arberqoku/missing-data-imputation.git
```

* Change directory to project
```bash
cd /path/to/missing-data-imputation
```

* (Optional) create conda environment
```bash
conda create -n mdi python=3.7 anaconda
```

* Install as python package
```bash
pip install -e .
```

The `-e` flag installs the package in editable/development mode, allowing changes made to the source code and updating the python package accordingly.

* (Alternative) install requirements directly
```bash
pip install -r requirements.txt
```

* Open notebook on browser
```bash
jupyter notebook
```

## Authors

* **[Alexander Hanf](mailto:a.hanf@campus.lmu.de)**
* **[Arber Qoku](mailto:arber.qoku@campus.lmu.de)**

## License

This project is licensed under the [MIT license](LICENSE.md).
