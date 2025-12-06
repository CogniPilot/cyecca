"""
Cyecca - Symbolic Estimation and Control with Computer Algebra

A Python library providing IR and analysis backends for the Rumoca compiler.
"""

from beartype.claw import beartype_package

beartype_package(__name__)

__version__ = "0.4.0"

from . import ir
from . import analysis

__all__ = ["ir", "analysis", "__version__", "load_ipython_extension"]


def load_ipython_extension(ipython):
    """
    Load Cyecca magic commands for Jupyter notebooks.

    Usage in a notebook:
        %load_ext cyecca

        from cyecca.ir import Model
        ir_model: Model | None = None

        %%modelica_rumoca ir_model
        model MyModel
            Real x(start=1);
        equation
            der(x) = -x;
        end MyModel;

        # ir_model is now a cyecca.ir.Model
        from cyecca.backends.casadi import CasadiBackend
        backend = CasadiBackend(ir_model).compile()
        t, sol = backend.simulate(10.0)
    """
    from .magic import load_ipython_extension as _load

    _load(ipython)
