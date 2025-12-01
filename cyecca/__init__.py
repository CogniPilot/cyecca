"""
Cyecca - Symbolic Estimation and Control with Computer Algebra

A Python library providing IR and analysis backends for the Rumoca compiler.
"""

from beartype.claw import beartype_package

beartype_package(__name__)

__version__ = "0.4.0"

from cyecca import ir
from cyecca import analysis

__all__ = ["ir", "analysis", "__version__"]
