"""
Topological Hypervector Sheaves (THS)

A framework for energy-efficient concept drift detection using
Cellular Sheaf Cohomology and Hyperdimensional Computing.
"""

from ths.drift.ths_drift import THSDrift
from ths.core.hypervector import (
    Hypervector,
    bind,
    bundle,
    permute,
    similarity,
    hamming_distance,
    random_hypervector,
)
from ths.sheaf.cellular_sheaf import CellularSheaf
from ths.core.encoders import (
    RandomProjectionEncoder,
    IDLevelEncoder,
    SequenceEncoder,
)

__version__ = "0.1.0"
__all__ = [
    "THSDrift",
    "Hypervector",
    "bind",
    "bundle", 
    "permute",
    "similarity",
    "hamming_distance",
    "random_hypervector",
    "CellularSheaf",
    "RandomProjectionEncoder",
    "IDLevelEncoder",
    "SequenceEncoder",
]
