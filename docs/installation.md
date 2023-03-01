## Setting up a conda environment

To avoid dependency conflicts, we recommend the use of a dedicated [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment. To do so, enter the following in terminal.

```
conda create -n lma
source activate lma
```

## Installation

Installation with PyPI using pip is not currently supported. To install the LMA package directly from GitHub, enter the following in terminal.

```
python -m pip install -e 'https://github.com/tranmartin45/lma.git'
```

Note that the installation is done with `-e`, meaning that the package will be editable (primarily for pre-release testing).