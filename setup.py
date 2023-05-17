# +
from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()


with open("linmo/__init__.py", "r") as f:
    init = f.readlines()

for line in init:
    if '__author__' in line:
        __author__ = line.split("'")[-2]
    if '__email__' in line:
        __email__ = line.split("'")[-2]
    if '__version__' in line:
        __version__ = line.split("'")[-2]

requirements = ["numpy", "pandas", "tqdm", "colorcet", "seaborn", "matplotlib", "statsmodels", "jupyter", "ipywidgets == 7.7.2", "jupyterlab_widgets == 1.1.1", "widgetsnbextension" , "pandas-profiling", "numba"]

setup(
    name='linmo',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='Package for Lineage Motif Analysis. Extracts statistically over- or under- represented cell fate patterns within a set of lineage trees.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
# -


