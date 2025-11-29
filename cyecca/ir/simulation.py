"""
Simulation utilities for the Cyecca DSL.

Provides a unified interface for simulation results,
independent of the compute backend used.

================================================================================
PROTOTYPE MODE - API IS IN FLUX
================================================================================

This DSL is in active prototype development. The API may change significantly
between versions. Do NOT maintain backward compatibility - iterate rapidly.

================================================================================
DESIGN PRINCIPLES - DO NOT REMOVE OR IGNORE
================================================================================

1. BACKEND-AGNOSTIC: This module defines interfaces that any backend must implement.
   The SimulationResult works with any conforming backend.

2. NUMPY ARRAYS: Results are numpy arrays for direct use with matplotlib/numpy.
   Users plot with: plt.plot(result.t, result(model.theta))

3. TYPE SAFETY: All functions MUST use beartype for runtime type checking.
   - All public functions decorated with @beartype
   - WHEN ADDING NEW FUNCTIONS: Always add @beartype decorator
   - Import: from beartype import beartype

================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from beartype import beartype

# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class SimulationResult:
    """
    Result of a model simulation.

    Provides convenient access to simulation data as numpy arrays.
    Variables are accessed using the callable interface with model variables
    for autocomplete support.

    Example
    -------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> from cyecca.dsl import model, var, der, sin  # doctest: +SKIP
    >>> from cyecca.backends import CasadiBackend  # doctest: +SKIP
    >>>
    >>> @model  # doctest: +SKIP
    ... class Pendulum:
    ...     g = var(9.81, parameter=True)
    ...     theta = var(start=0.5, state=True)
    ...     omega = var(state=True)
    ...
    ...     @equations
    ...     def _(m):
    ...         der(m.theta) == m.omega
    ...         der(m.omega) == -m.g * sin(m.theta)
    >>>
    >>> pend = Pendulum()  # doctest: +SKIP
    >>> compiled = CasadiBackend.compile(pend.flatten())  # doctest: +SKIP
    >>> result = compiled.simulate(tf=10.0)  # doctest: +SKIP
    >>>
    >>> # Plot using model variables (autocomplete works on pend.theta)
    >>> plt.plot(result.t, result(pend.theta))  # doctest: +SKIP
    >>> plt.plot(result.t, result(pend.omega))  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    """

    # Time vector
    t: np.ndarray

    # Trajectory data: name -> array
    _data: Dict[str, np.ndarray] = field(default_factory=dict)

    # Metadata
    model_name: str = ""
    state_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    discrete_names: List[str] = field(default_factory=list)

    @beartype
    def __call__(self, var: Any) -> np.ndarray:
        """
        Get trajectory data for a variable.

        Parameters
        ----------
        var : SymbolicVar, Expr, or str
            The variable to get data for. Can be:
            - A model variable (e.g., pend.theta) - provides autocomplete
            - A string name (e.g., "theta")

        Returns
        -------
        np.ndarray
            The trajectory data for this variable

        Example
        -------
        >>> result(pend.theta)  # doctest: +SKIP
        >>> result("theta")     # doctest: +SKIP
        """
        # Get the variable name
        if isinstance(var, str):
            name = var
        elif hasattr(var, "_name"):
            # SymbolicVar
            name = var._name
        elif hasattr(var, "name"):
            # Expr or other object with .name
            name = var.name
        else:
            raise TypeError(
                f"Expected model variable or string, got {type(var).__name__}. "
                f"Use result(model.var_name) or result('var_name')"
            )

        if name not in self._data:
            raise KeyError(f"Variable '{name}' not in result. " f"Available: {self.available_names}")
        return self._data[name]

    def __getitem__(self, key: str) -> np.ndarray:
        """Get trajectory by string name (dict-style access)."""
        if key == "t":
            return self.t
        if key not in self._data:
            raise KeyError(f"No trajectory named '{key}'. Available: {self.available_names}")
        return self._data[key]

    @property
    def available_names(self) -> List[str]:
        """List all available trajectory names."""
        return list(self._data.keys())

    @property
    def states(self) -> Dict[str, np.ndarray]:
        """State trajectories as a dict."""
        return {name: self._data[name] for name in self.state_names if name in self._data}

    @property
    def outputs(self) -> Dict[str, np.ndarray]:
        """Output trajectories as a dict."""
        return {name: self._data[name] for name in self.output_names if name in self._data}

    @property
    def inputs(self) -> Dict[str, np.ndarray]:
        """Input trajectories as a dict."""
        return {name: self._data[name] for name in self.input_names if name in self._data}

    @property
    def discrete(self) -> Dict[str, np.ndarray]:
        """Discrete variable trajectories as a dict."""
        return {name: self._data[name] for name in self.discrete_names if name in self._data}

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """All trajectories as a single dict."""
        result = {"t": self.t}
        result.update(self._data)
        return result


# =============================================================================
# Simulator Protocol (what backends must implement)
# =============================================================================


class Simulator(ABC):
    """
    Abstract base class for model simulators.

    Backends should implement this interface to provide simulation capabilities.
    """

    @abstractmethod
    def simulate(
        self,
        t0: float = 0.0,
        tf: float = 10.0,
        dt: float = 0.01,
        x0: Optional[Dict[str, Any]] = None,
        u: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        u_func: Optional[Callable[[float], Dict[str, Any]]] = None,
    ) -> SimulationResult:
        """
        Simulate the model.

        Parameters
        ----------
        t0 : float
            Initial time
        tf : float
            Final time
        dt : float
            Time step
        x0 : dict, optional
            Initial state values
        u : dict, optional
            Constant input values
        params : dict, optional
            Parameter values
        u_func : callable, optional
            Function u_func(t) -> dict for time-varying inputs

        Returns
        -------
        SimulationResult
            Simulation results
        """
        pass

    @property
    @abstractmethod
    def state_names(self) -> List[str]:
        """Names of state variables."""
        pass

    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        """Names of input variables."""
        pass

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """Names of output variables."""
        pass

    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """Names of parameters."""
        pass
