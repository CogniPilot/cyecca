"""
Compute backends for compiling and simulating models.

Backends compile FlatModel representations into executable functions
for simulation, optimization, and analysis.

Available backends:
- casadi: CasADi-based backend for symbolic computation and simulation
  - SX (default): Scalar symbolic expressions - expands arrays
  - MX: Matrix symbolic expressions - keeps array structure for efficiency
- sympy: SymPy-based backend for symbolic analysis
  - Jacobian computation and linearization
  - LaTeX output
  - Code generation (NumPy, C, Fortran)

Integrators:
- RK4: Fixed-step 4th-order Runge-Kutta (simple, fast)
- CVODES: SUNDIALS variable-step BDF/Adams method (accurate, handles stiff systems)
- IDAS: SUNDIALS variable-step BDF for DAEs
"""

from cyecca.backends.casadi import CasadiBackend, CompiledModel, Integrator, SymbolicType
from cyecca.backends.sympy_backend import CompiledSymPyModel, SymPyBackend
from cyecca.ir.simulation import SimulationResult, Simulator

__all__ = [
    "CasadiBackend",
    "CompiledModel",
    "Integrator",
    "SymbolicType",
    "SymPyBackend",
    "CompiledSymPyModel",
    "SimulationResult",
    "Simulator",
]
