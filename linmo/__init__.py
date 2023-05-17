"""The modules exported by this package include the following.

- **'resample'**: Resamples tree dataset and exports count of various kinds of cell fate patterns across all resamples, the original
    trees, and the expected number (solved analytically).
- **'plot'**: Visualize DataFrame outputs from 'resample' module as frequency or deviation plots.
- **'simulate'**: Simulate lineage trees with input progenitor/cell types and given transition matrix.
"""

# +
from . import resample
from . import plot
from . import simulate

__author__ = 'Martin Tran'
__email__ = 'mtran@caltech.edu'
__version__ = '0.0.1'
