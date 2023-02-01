"""Perform motif analysis on provided tree dataset.

Modules exported by this package:

- **'resample'**: Resamples tree dataset and exports count of doublet, triplet, or quartet across all resamples, the original
    trees, and the expected number (solved analytically).
- **'plot'**: Visualize DataFrame outputs from 'resample' module as frequency or deviation plots.
"""

# +
from . import resample
from . import plot

__author__ = 'Martin Tran'
__email__ = 'mtran@caltech.edu'
__version__ = '0.0.1'
