"""
Compute backends for the Cyecca DSL.

Backends compile FlatModel representations into executable functions
for simulation, optimization, and analysis.

Available backends:
- casadi: CasADi-based backend for symbolic computation and simulation
  - SX (default): Scalar symbolic expressions - expands arrays
  - MX: Matrix symbolic expressions - keeps array structure for efficiency
"""

from cyecca.dsl.backends.casadi import CasadiBackend, CompiledModel, SymbolicType
from cyecca.dsl.simulation import SimulationResult, Simulator

__all__ = ["CasadiBackend", "CompiledModel", "SymbolicType", "SimulationResult", "Simulator"]
