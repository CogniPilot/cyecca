"""
Base backend interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Type, cast
import tempfile

import numpy as np

from cyecca.ir.model import Model

# Type variable for generic class methods
T = TypeVar("T", bound="Backend")


class Backend(ABC):
    """
    Abstract base class for all backends.

    A backend converts the IR Model into executable code using a specific
    symbolic framework (CasADi, SymPy, JAX, etc.).
    """

    def __init__(self, model: Model) -> None:
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

    @classmethod
    def from_file(cls: Type[T], modelica_file: str, **backend_kwargs: Any) -> T:
        """
        Compile a Modelica file and return a compiled backend instance.

        This is a convenience method that handles the full pipeline:
        1. Compile Modelica file with Rumoca
        2. Export to Base Modelica JSON
        3. Import into Cyecca IR
        4. Create and compile the backend

        Args:
            modelica_file: Path to the Modelica (.mo) file
            **backend_kwargs: Additional keyword arguments to pass to the backend constructor
                             (e.g., sym_type='MX' for CasadiBackend)

        Returns:
            Compiled backend instance ready for simulation

        Example:
            >>> from cyecca.backends.casadi import CasadiBackend
            >>> backend = CasadiBackend.from_file("model.mo", sym_type='SX')
            >>> t, result = backend.simulate(10.0, dt=0.01)
        """
        try:
            import rumoca
        except ImportError:
            raise ImportError(
                "Rumoca is required to compile Modelica files. "
                "Install it with: pip install rumoca"
            )

        from cyecca.io.base_modelica import import_base_modelica

        # Create temporary file for JSON export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json_path = tmp.name

        try:
            # Compile Modelica file with Rumoca
            try:
                result = rumoca.compile(modelica_file)
            except Exception as e:
                # Re-raise with clearer error message
                error_msg = str(e)
                if "CompilationError" in type(e).__name__:
                    # Miette diagnostics already contain file context, just pass through
                    raise RuntimeError(error_msg) from None
                else:
                    raise

            # Export to Base Modelica JSON
            result.export_base_modelica_json(json_path)

            # Import into Cyecca IR
            model = import_base_modelica(json_path)

            # Create backend instance using normal constructor
            # This works fine with beartype - the issue was with manual instantiation
            backend = cls(model, **backend_kwargs)

            # Compile the backend
            backend.compile()

            return cast(T, backend)
        finally:
            # Clean up temporary file
            Path(json_path).unlink(missing_ok=True)

    @classmethod
    def from_string(cls: Type[T], modelica_code: str, **backend_kwargs: Any) -> T:
        """
        Compile Modelica code from a string and return a compiled backend instance.

        This is a convenience method that handles the full pipeline:
        1. Write Modelica code to temporary file
        2. Compile with Rumoca
        3. Export to Base Modelica JSON
        4. Import into Cyecca IR
        5. Create and compile the backend

        Args:
            modelica_code: Modelica source code as a string
            **backend_kwargs: Additional keyword arguments to pass to the backend constructor
                             (e.g., sym_type='MX' for CasadiBackend)

        Returns:
            Compiled backend instance ready for simulation

        Example:
            >>> modelica_code = '''
            ... model SimpleModel
            ...   Real x(start=1.0);
            ... equation
            ...   der(x) = -x;
            ... end SimpleModel;
            ... '''
            >>> from cyecca.backends.casadi import CasadiBackend
            >>> backend = CasadiBackend.from_string(modelica_code, sym_type='SX')
            >>> t, result = backend.simulate(5.0, dt=0.1)
        """
        try:
            import rumoca
        except ImportError:
            raise ImportError(
                "Rumoca is required to compile Modelica code. "
                "Install it with: pip install rumoca"
            )

        from cyecca.io.base_modelica import import_base_modelica

        # Create temporary files for Modelica and JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mo", delete=False) as mo_tmp:
            mo_path = mo_tmp.name
            mo_tmp.write(modelica_code)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as json_tmp:
            json_path = json_tmp.name

        try:
            # Compile Modelica code with Rumoca
            try:
                result = rumoca.compile(mo_path)
            except Exception as e:
                # Re-raise with error message
                error_msg = str(e)
                if "CompilationError" in type(e).__name__:
                    # Miette diagnostics are already beautifully formatted, just pass through
                    raise RuntimeError(error_msg) from None
                else:
                    raise

            # Export to Base Modelica JSON
            result.export_base_modelica_json(json_path)

            # Import into Cyecca IR
            model = import_base_modelica(json_path)

            # Create backend instance using normal constructor
            # This works fine with beartype - the issue was with manual instantiation
            backend = cls(model, **backend_kwargs)

            # Compile the backend
            backend.compile()

            return cast(T, backend)
        finally:
            # Clean up temporary files
            Path(mo_path).unlink(missing_ok=True)
            Path(json_path).unlink(missing_ok=True)
