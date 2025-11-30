"""
Intermediate Representation (IR) for dynamical systems.

This module provides the core data structures that Rumoca compiles to.
It's designed to be easy to generate programmatically while providing
enough structure for backend code generation.
"""

from cyecca.ir.types import VariableType, Causality, Variability, PrimitiveType
from cyecca.ir.variable import Variable
from cyecca.ir.expr import (
    Expr,
    Literal,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    ComponentRef,
    ComponentRefPart,
    VarRef,
    ArrayRef,
    ArrayLiteral,
    Slice,
    IfExpr,
    der,
    pre,
    edge,
)
from cyecca.ir.equation import Equation, EquationType
from cyecca.ir.statement import (
    Statement,
    Assignment,
    IfStatement,
    ForStatement,
    WhileStatement,
    WhenStatement,
    ReinitStatement,
    BreakStatement,
    ReturnStatement,
    FunctionCallStatement,
)
from cyecca.ir.algorithm import AlgorithmSection
from cyecca.ir.event import Event
from cyecca.ir.model import Model

__all__ = [
    # Types
    "VariableType",
    "Causality",
    "Variability",
    "PrimitiveType",
    # Core structures
    "Variable",
    "Expr",
    "Literal",
    "BinaryOp",
    "UnaryOp",
    "FunctionCall",
    "ComponentRef",
    "ComponentRefPart",
    "ArrayLiteral",
    "Slice",
    "IfExpr",
    # Backward compatibility (deprecated)
    "VarRef",
    "ArrayRef",
    # Modelica operators
    "der",
    "pre",
    "edge",
    # Equations
    "Equation",
    "EquationType",
    # Statements
    "Statement",
    "Assignment",
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "WhenStatement",
    "ReinitStatement",
    "BreakStatement",
    "ReturnStatement",
    "FunctionCallStatement",
    # Algorithms
    "AlgorithmSection",
    # Events
    "Event",
    # Model
    "Model",
]
