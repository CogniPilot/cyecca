"""
Model validation utilities.

Provides comprehensive validation for Cyecca IR models, including:
- Undefined variable detection in expressions
- Array bounds checking
- Parameter and start value validation
- Structural analysis integration (algebraic loops, balance checks)
- Type consistency checks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from cyecca.ir.expr import (
    Expr,
    Literal,
    VarRef,
    ComponentRef,
    ArrayRef,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    IfExpr,
    ArrayLiteral,
    Slice,
)
from cyecca.ir.equation import Equation, EquationType
from cyecca.ir.types import VariableType


class ValidationSeverity(Enum):
    """Severity level of a validation issue."""

    ERROR = "error"  # Critical issue that prevents correct execution
    WARNING = "warning"  # Issue that may cause problems
    INFO = "info"  # Informational note


class ValidationCategory(Enum):
    """Category of validation issue."""

    UNDEFINED_VARIABLE = "undefined_variable"
    ARRAY_BOUNDS = "array_bounds"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_VALUE = "missing_value"
    STRUCTURAL = "structural"
    EQUATION_BALANCE = "equation_balance"
    DERIVATIVE = "derivative"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    location: Optional[str] = None  # e.g., "equation 3", "variable x"
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value.upper()}] {self.category.value}: {self.message}{loc}"


@dataclass
class ValidationResult:
    """Result of model validation."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """True if there are any errors."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True if there are any warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    @property
    def is_valid(self) -> bool:
        """True if there are no errors (warnings are OK)."""
        return not self.has_errors

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all errors."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warnings."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add(self, issue: ValidationIssue) -> None:
        """Add an issue."""
        self.issues.append(issue)

    def add_error(
        self,
        category: ValidationCategory,
        message: str,
        location: Optional[str] = None,
        **details,
    ) -> None:
        """Add an error."""
        self.add(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=category,
                message=message,
                location=location,
                details=details,
            )
        )

    def add_warning(
        self,
        category: ValidationCategory,
        message: str,
        location: Optional[str] = None,
        **details,
    ) -> None:
        """Add a warning."""
        self.add(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=category,
                message=message,
                location=location,
                details=details,
            )
        )

    def add_info(
        self,
        category: ValidationCategory,
        message: str,
        location: Optional[str] = None,
        **details,
    ) -> None:
        """Add an info message."""
        self.add(
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                category=category,
                message=message,
                location=location,
                details=details,
            )
        )

    def summary(self) -> str:
        """Get a summary of validation results."""
        n_errors = len(self.errors)
        n_warnings = len(self.warnings)
        n_info = len([i for i in self.issues if i.severity == ValidationSeverity.INFO])

        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Validation Result: {status}",
            f"  Errors: {n_errors}",
            f"  Warnings: {n_warnings}",
            f"  Info: {n_info}",
        ]

        if self.issues:
            lines.append("\nIssues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def collect_variable_refs_with_indices(
    expr: Expr, var_shapes: dict[str, Optional[list[int]]]
) -> list[tuple[str, Optional[tuple[int, ...]]]]:
    """
    Collect all variable references in an expression, including array indices.

    Args:
        expr: Expression to analyze
        var_shapes: Map of variable name to shape (None for scalars)

    Returns:
        List of (variable_name, indices) tuples. indices is None for scalars.
    """
    refs: list[tuple[str, Optional[tuple[int, ...]]]] = []

    if expr is None:
        return refs

    if isinstance(expr, Literal):
        return refs

    elif isinstance(expr, VarRef):
        refs.append((expr.name, None))

    elif isinstance(expr, ComponentRef):
        # Build the full variable name
        parts = []
        for i, part in enumerate(expr.parts):
            if i < len(expr.parts) - 1:
                parts.append(str(part))
            else:
                parts.append(part.name)

        var_name = ".".join(parts)
        last_part = expr.parts[-1]

        # Extract indices from subscripts
        if last_part.subscripts:
            indices = []
            for sub in last_part.subscripts:
                if isinstance(sub, Literal) and isinstance(sub.value, (int, float)):
                    indices.append(int(sub.value))
                else:
                    # Symbolic index - can't check bounds statically
                    indices = None
                    break
            refs.append((var_name, tuple(indices) if indices is not None else None))
        else:
            refs.append((var_name, None))

    elif isinstance(expr, ArrayRef):
        indices = []
        for idx in expr.indices:
            if isinstance(idx, Literal) and isinstance(idx.value, (int, float)):
                indices.append(int(idx.value))
            else:
                indices = None
                break
        refs.append((expr.name, tuple(indices) if indices is not None else None))

    elif isinstance(expr, BinaryOp):
        refs.extend(collect_variable_refs_with_indices(expr.left, var_shapes))
        refs.extend(collect_variable_refs_with_indices(expr.right, var_shapes))

    elif isinstance(expr, UnaryOp):
        refs.extend(collect_variable_refs_with_indices(expr.operand, var_shapes))

    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            refs.extend(collect_variable_refs_with_indices(arg, var_shapes))

    elif isinstance(expr, IfExpr):
        refs.extend(collect_variable_refs_with_indices(expr.condition, var_shapes))
        refs.extend(collect_variable_refs_with_indices(expr.true_expr, var_shapes))
        refs.extend(collect_variable_refs_with_indices(expr.false_expr, var_shapes))

    elif isinstance(expr, ArrayLiteral):
        for elem in expr.elements:
            refs.extend(collect_variable_refs_with_indices(elem, var_shapes))

    elif isinstance(expr, Slice):
        if expr.start:
            refs.extend(collect_variable_refs_with_indices(expr.start, var_shapes))
        if expr.stop:
            refs.extend(collect_variable_refs_with_indices(expr.stop, var_shapes))
        if expr.step:
            refs.extend(collect_variable_refs_with_indices(expr.step, var_shapes))

    return refs


def _check_undefined_variables(
    model: Any,
    result: ValidationResult,
    var_names: set[str],
    var_shapes: dict[str, Optional[list[int]]],
) -> None:
    """Check for undefined variable references in equations."""

    def check_expr(expr: Expr, location: str) -> None:
        refs = collect_variable_refs_with_indices(expr, var_shapes)
        for var_name, indices in refs:
            if var_name not in var_names:
                result.add_error(
                    ValidationCategory.UNDEFINED_VARIABLE,
                    f"Reference to undefined variable '{var_name}'",
                    location=location,
                    variable=var_name,
                )

    for eq_idx, eq in enumerate(model.equations):
        loc = f"equation {eq_idx}"

        if eq.eq_type == EquationType.SIMPLE:
            if eq.lhs is not None:
                check_expr(eq.lhs, loc)
            if eq.rhs is not None:
                check_expr(eq.rhs, loc)

        elif eq.eq_type == EquationType.WHEN:
            if eq.condition is not None:
                check_expr(eq.condition, f"{loc} (when condition)")
            if eq.when_equations:
                for sub_eq in eq.when_equations:
                    if sub_eq.lhs is not None:
                        check_expr(sub_eq.lhs, f"{loc} (when body)")
                    if sub_eq.rhs is not None:
                        check_expr(sub_eq.rhs, f"{loc} (when body)")

        elif eq.eq_type == EquationType.FOR:
            if eq.for_equations:
                for sub_eq in eq.for_equations:
                    if sub_eq.lhs is not None:
                        check_expr(sub_eq.lhs, f"{loc} (for body)")
                    if sub_eq.rhs is not None:
                        check_expr(sub_eq.rhs, f"{loc} (for body)")


def _check_array_bounds(
    model: Any, result: ValidationResult, var_shapes: dict[str, Optional[list[int]]]
) -> None:
    """Check array index bounds in expressions."""

    def check_expr(expr: Expr, location: str) -> None:
        refs = collect_variable_refs_with_indices(expr, var_shapes)
        for var_name, indices in refs:
            if indices is None:
                continue  # No static indices or symbolic index

            shape = var_shapes.get(var_name)
            if shape is None:
                # Scalar variable with index
                result.add_error(
                    ValidationCategory.ARRAY_BOUNDS,
                    f"Cannot index scalar variable '{var_name}'",
                    location=location,
                    variable=var_name,
                    indices=indices,
                )
                continue

            # Check each dimension
            # Note: Modelica uses 1-based indexing
            for dim, (idx, size) in enumerate(zip(indices, shape)):
                if idx < 1:
                    result.add_error(
                        ValidationCategory.ARRAY_BOUNDS,
                        f"Index {idx} is below lower bound (1) for '{var_name}' dimension {dim + 1}",
                        location=location,
                        variable=var_name,
                        index=idx,
                        dimension=dim + 1,
                        size=size,
                    )
                elif idx > size:
                    result.add_error(
                        ValidationCategory.ARRAY_BOUNDS,
                        f"Index {idx} exceeds upper bound ({size}) for '{var_name}' dimension {dim + 1}",
                        location=location,
                        variable=var_name,
                        index=idx,
                        dimension=dim + 1,
                        size=size,
                    )

    for eq_idx, eq in enumerate(model.equations):
        loc = f"equation {eq_idx}"

        if eq.eq_type == EquationType.SIMPLE:
            if eq.lhs is not None:
                check_expr(eq.lhs, loc)
            if eq.rhs is not None:
                check_expr(eq.rhs, loc)

        elif eq.eq_type == EquationType.WHEN:
            if eq.condition is not None:
                check_expr(eq.condition, f"{loc} (when condition)")
            if eq.when_equations:
                for sub_eq in eq.when_equations:
                    if sub_eq.lhs is not None:
                        check_expr(sub_eq.lhs, f"{loc} (when body)")
                    if sub_eq.rhs is not None:
                        check_expr(sub_eq.rhs, f"{loc} (when body)")


def _check_missing_values(model: Any, result: ValidationResult) -> None:
    """Check for missing start values and parameter values."""
    for var in model.variables:
        if var.var_type == VariableType.STATE:
            # States should have start values for simulation
            if var.start is None:
                result.add_warning(
                    ValidationCategory.MISSING_VALUE,
                    f"State variable '{var.name}' has no start value (defaulting to 0)",
                    location=f"variable {var.name}",
                )

        elif var.var_type == VariableType.PARAMETER:
            # Parameters should have values
            if var.value is None and var.start is None:
                result.add_warning(
                    ValidationCategory.MISSING_VALUE,
                    f"Parameter '{var.name}' has no value assigned",
                    location=f"variable {var.name}",
                )

        elif var.var_type == VariableType.CONSTANT:
            # Constants must have values
            if var.value is None:
                result.add_error(
                    ValidationCategory.MISSING_VALUE,
                    f"Constant '{var.name}' has no value assigned",
                    location=f"variable {var.name}",
                )


def _check_derivative_equations(model: Any, result: ValidationResult) -> None:
    """Check that all states have derivative equations."""
    # Find all state variables
    state_names = {v.name for v in model.variables if v.var_type == VariableType.STATE}

    # Find all variables that have der() on them
    der_vars: set[str] = set()

    def find_der_in_expr(expr: Expr) -> None:
        if expr is None:
            return

        if isinstance(expr, FunctionCall) and expr.func == "der":
            if len(expr.args) > 0:
                arg = expr.args[0]
                if isinstance(arg, VarRef):
                    der_vars.add(arg.name)
                elif isinstance(arg, ComponentRef):
                    name = ".".join(p.name for p in arg.parts)
                    der_vars.add(name)
        elif isinstance(expr, BinaryOp):
            find_der_in_expr(expr.left)
            find_der_in_expr(expr.right)
        elif isinstance(expr, UnaryOp):
            find_der_in_expr(expr.operand)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                find_der_in_expr(arg)
        elif isinstance(expr, IfExpr):
            find_der_in_expr(expr.condition)
            find_der_in_expr(expr.true_expr)
            find_der_in_expr(expr.false_expr)

    for eq in model.equations:
        if eq.eq_type == EquationType.SIMPLE:
            if eq.lhs is not None:
                find_der_in_expr(eq.lhs)
            if eq.rhs is not None:
                find_der_in_expr(eq.rhs)

    # Check for states without derivatives
    for state_name in state_names:
        if state_name not in der_vars:
            result.add_error(
                ValidationCategory.DERIVATIVE,
                f"State variable '{state_name}' has no derivative equation (no der({state_name}) found)",
                location=f"variable {state_name}",
            )

    # Check for derivatives on non-states
    for der_name in der_vars:
        if der_name not in state_names:
            # Check if the variable exists at all
            if not model.has_variable(der_name):
                result.add_error(
                    ValidationCategory.UNDEFINED_VARIABLE,
                    f"der({der_name}) references undefined variable '{der_name}'",
                )
            else:
                result.add_warning(
                    ValidationCategory.DERIVATIVE,
                    f"der({der_name}) used on non-state variable (will be promoted to state)",
                    location=f"variable {der_name}",
                )


def _check_equation_balance(model: Any, result: ValidationResult) -> None:
    """Check equation/variable balance."""
    # Count equations (simplified - doesn't expand for loops)
    n_equations = len(model.equations)

    # Count unknowns (states + algebraic)
    n_states = len([v for v in model.variables if v.var_type == VariableType.STATE])
    n_algebraic = len([v for v in model.variables if v.var_type == VariableType.ALGEBRAIC])
    n_unknowns = n_states + n_algebraic

    if n_equations < n_unknowns:
        result.add_error(
            ValidationCategory.EQUATION_BALANCE,
            f"System is under-determined: {n_equations} equations for {n_unknowns} unknowns",
            n_equations=n_equations,
            n_unknowns=n_unknowns,
            missing=n_unknowns - n_equations,
        )
    elif n_equations > n_unknowns and n_unknowns > 0:
        result.add_warning(
            ValidationCategory.EQUATION_BALANCE,
            f"System may be over-determined: {n_equations} equations for {n_unknowns} unknowns",
            n_equations=n_equations,
            n_unknowns=n_unknowns,
            excess=n_equations - n_unknowns,
        )


def validate_model(
    model: Any,
    check_undefined_vars: bool = True,
    check_array_bounds: bool = True,
    check_missing_values: bool = True,
    check_derivatives: bool = True,
    check_balance: bool = True,
) -> ValidationResult:
    """
    Perform comprehensive validation of a Cyecca IR model.

    Args:
        model: The model to validate
        check_undefined_vars: Check for undefined variable references
        check_array_bounds: Check array index bounds
        check_missing_values: Check for missing start/parameter values
        check_derivatives: Check derivative equations for states
        check_balance: Check equation/variable balance

    Returns:
        ValidationResult containing all issues found

    Note:
        BLT (Block Lower Triangular) analysis is handled by Rumoca, not Cyecca.
    """
    result = ValidationResult()

    # Build variable info
    var_names = {v.name for v in model.variables}
    var_shapes: dict[str, Optional[list[int]]] = {}
    for var in model.variables:
        var_shapes[var.name] = var.shape if var.is_array else None

    # Run checks
    if check_undefined_vars:
        _check_undefined_variables(model, result, var_names, var_shapes)

    if check_array_bounds:
        _check_array_bounds(model, result, var_shapes)

    if check_missing_values:
        _check_missing_values(model, result)

    if check_derivatives:
        _check_derivative_equations(model, result)

    if check_balance:
        _check_equation_balance(model, result)

    return result
