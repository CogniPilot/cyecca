"""
Base backend interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np

from cyecca.ir.model import Model


class Backend(ABC):
    """
    Abstract base class for all backends.

    A backend converts the IR Model into executable code using a specific
    symbolic framework (CasADi, SymPy, JAX, etc.).
    """

    def __init__(self, model: Model):
        """
        Initialize the backend with a model.

        Args:
            model: The IR model to compile
        """
        self.model = model
        self._compiled = False

    @abstractmethod
    def compile(self) -> None:
        """
        Compile the model to the backend representation.

        This should generate all necessary functions for simulation,
        linearization, etc.
        """
        pass

    @abstractmethod
    def simulate(
        self,
        t_final: float,
        dt: float = 0.01,
        input_func: Optional[Callable[[float], dict[str, float]]] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Simulate the model.

        Args:
            t_final: Final simulation time
            dt: Time step
            input_func: Optional function that returns input values at time t

        Returns:
            (t, sol) where:
                t: Time array of shape (n_steps,)
                sol: Dictionary mapping variable names to arrays of shape (n_steps,)
        """
        pass

    @abstractmethod
    def linearize(
        self, x0: Optional[dict[str, float]] = None, u0: Optional[dict[str, float]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the model at an operating point.

        Args:
            x0: State values at operating point (uses start values if None)
            u0: Input values at operating point (uses 0 if None)

        Returns:
            (A, B, C, D) state-space matrices
        """
        pass

    @abstractmethod
    def get_rhs_function(self) -> Callable:
        """
        Get the right-hand side function for the ODEs.

        Returns:
            Function f(t, x, u, p) -> xdot
        """
        pass

    def _ensure_compiled(self) -> None:
        """Raise an error if the model hasn't been compiled yet."""
        if not self._compiled:
            raise RuntimeError("Model must be compiled before use. Call compile() first.")
