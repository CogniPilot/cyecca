"""
Tests for array variables, indexing, and slicing.

This is Phase 3.1 of the IR migration - adding support for Modelica arrays
and array operations.
"""

import pytest
from cyecca.ir import (
    Expr,
    Literal,
    ComponentRef,
    ComponentRefPart,
    Slice,
    ArrayLiteral,
    Variable,
    VariableType,
    Model,
    Equation,
    der,
)


def test_array_variable_creation():
    """Test creating array variables."""
    # Vector: Real x[3]
    x = Variable(name="x", var_type=VariableType.STATE, shape=[3])
    assert x.is_array
    assert not x.is_scalar
    assert x.shape == [3]

    # Matrix: Real A[3,3]
    A = Variable(name="A", var_type=VariableType.PARAMETER, shape=[3, 3])
    assert A.is_array
    assert A.shape == [3, 3]

    # Scalar for comparison
    s = Variable(name="s", var_type=VariableType.STATE)
    assert s.is_scalar
    assert not s.is_array
    assert s.shape is None


def test_simple_array_indexing():
    """Test simple array indexing: x[i]."""
    i = Expr.literal(1)
    x_i = Expr.component_ref(("x", [i]))

    assert isinstance(x_i, ComponentRef)
    assert len(x_i.parts) == 1
    assert x_i.parts[0].name == "x"
    assert len(x_i.parts[0].subscripts) == 1
    assert str(x_i) == "x[1]"


def test_matrix_indexing():
    """Test matrix indexing: A[i,j]."""
    i = Expr.literal(1)
    j = Expr.literal(2)
    A_ij = Expr.component_ref(("A", [i, j]))

    assert len(A_ij.parts) == 1
    assert A_ij.parts[0].name == "A"
    assert len(A_ij.parts[0].subscripts) == 2
    assert str(A_ij) == "A[1,2]"


def test_slice_creation():
    """Test creating slice expressions."""
    # Full slice: :
    s1 = Expr.slice()
    assert isinstance(s1, Slice)
    assert s1.start is None
    assert s1.stop is None
    assert s1.step is None
    assert str(s1) == ":"

    # Range slice: 1:3
    s2 = Expr.slice(Expr.literal(1), Expr.literal(3))
    assert s2.start is not None
    assert s2.stop is not None
    assert s2.step is None
    assert str(s2) == "1:3"

    # Step slice: 1:2:10
    s3 = Expr.slice(Expr.literal(1), Expr.literal(10), Expr.literal(2))
    assert s3.start is not None
    assert s3.stop is not None
    assert s3.step is not None
    assert str(s3) == "1:2:10"

    # Open-ended slices
    s4 = Expr.slice(None, Expr.literal(5))
    assert str(s4) == ":5"

    s5 = Expr.slice(Expr.literal(3), None)
    assert str(s5) == "3:"


def test_array_slicing_basic():
    """Test basic array slicing: x[1:3]."""
    slice_expr = Expr.slice(Expr.literal(1), Expr.literal(3))
    x_slice = Expr.component_ref(("x", [slice_expr]))

    assert len(x_slice.parts) == 1
    assert x_slice.parts[0].name == "x"
    assert len(x_slice.parts[0].subscripts) == 1
    assert isinstance(x_slice.parts[0].subscripts[0], Slice)
    assert str(x_slice) == "x[1:3]"


def test_array_slicing_full():
    """Test full array slicing: x[:]."""
    slice_expr = Expr.slice()
    x_all = Expr.component_ref(("x", [slice_expr]))

    assert str(x_all) == "x[:]"


def test_matrix_slicing():
    """Test matrix slicing: A[:,2] and A[1,:]."""
    # A[:,2] - all rows, column 2
    col_slice = Expr.slice()
    col_idx = Expr.literal(2)
    A_col2 = Expr.component_ref(("A", [col_slice, col_idx]))

    assert len(A_col2.parts[0].subscripts) == 2
    assert isinstance(A_col2.parts[0].subscripts[0], Slice)
    assert isinstance(A_col2.parts[0].subscripts[1], Literal)
    assert str(A_col2) == "A[:,2]"

    # A[1,:] - row 1, all columns
    row_idx = Expr.literal(1)
    row_slice = Expr.slice()
    A_row1 = Expr.component_ref(("A", [row_idx, row_slice]))

    assert str(A_row1) == "A[1,:]"


def test_matrix_submatrix_slicing():
    """Test extracting submatrix: A[1:2, 2:4]."""
    row_slice = Expr.slice(Expr.literal(1), Expr.literal(2))
    col_slice = Expr.slice(Expr.literal(2), Expr.literal(4))
    A_sub = Expr.component_ref(("A", [row_slice, col_slice]))

    assert str(A_sub) == "A[1:2,2:4]"


def test_hierarchical_array_access():
    """Test hierarchical + array: vehicle.positions[i].x."""
    i = Expr.literal(1)
    ref = Expr.component_ref("vehicle", ("positions", [i]), "x")

    assert len(ref.parts) == 3
    assert ref.parts[0].name == "vehicle"
    assert ref.parts[1].name == "positions"
    assert len(ref.parts[1].subscripts) == 1
    assert ref.parts[2].name == "x"
    assert str(ref) == "vehicle.positions[1].x"


def test_array_literal():
    """Test array literals: [1, 2, 3]."""
    arr = Expr.array_literal(Expr.literal(1), Expr.literal(2), Expr.literal(3))

    assert isinstance(arr, ArrayLiteral)
    assert len(arr.elements) == 3
    assert str(arr) == "[1, 2, 3]"


def test_array_of_expressions():
    """Test array containing expressions: [x, y, z]."""
    x = Expr.var_ref("x")
    y = Expr.var_ref("y")
    z = Expr.var_ref("z")

    arr = Expr.array_literal(x, y, z)
    assert str(arr) == "[x, y, z]"


def test_nested_array_literals():
    """Test nested array literals for matrices: [[1,2], [3,4]]."""
    row1 = Expr.array_literal(Expr.literal(1), Expr.literal(2))
    row2 = Expr.array_literal(Expr.literal(3), Expr.literal(4))
    matrix = Expr.array_literal(row1, row2)

    assert str(matrix) == "[[1, 2], [3, 4]]"


def test_der_with_array_element():
    """Test der() on array elements: der(x[i])."""
    i = Expr.literal(1)
    x_i = Expr.component_ref(("x", [i]))
    dx_i = der(x_i)

    assert dx_i.func == "der"
    assert len(dx_i.args) == 1
    assert str(dx_i.args[0]) == "x[1]"


def test_equation_with_array_slicing():
    """Test equation with array slicing: y = x[1:3]."""
    y = Expr.var_ref("y")
    x_slice = Expr.component_ref(("x", [Expr.slice(Expr.literal(1), Expr.literal(3))]))

    eq = Equation.simple(y, x_slice)
    assert eq.lhs == y
    assert str(eq.rhs) == "x[1:3]"


def test_array_variable_in_model():
    """Test adding array variables to a model."""
    model = Model(name="ArrayModel")

    # Add vector state: Real x[3]
    x = Variable(name="x", var_type=VariableType.STATE, shape=[3], start=0.0)
    model.add_variable(x)

    # Add matrix parameter: Real A[3,3]
    A = Variable(name="A", var_type=VariableType.PARAMETER, shape=[3, 3])
    model.add_variable(A)

    assert model.has_variable("x")
    assert model.has_variable("A")
    assert model.get_variable("x").is_array
    assert model.get_variable("A").shape == [3, 3]


def test_for_equation_basic():
    """Test basic for equation: for i in 1:3 loop der(x[i]) = v[i]; end for."""
    # Create loop body equation: der(x[i]) = v[i]
    i_var = Expr.var_ref("i")
    x_i = Expr.component_ref(("x", [i_var]))
    v_i = Expr.component_ref(("v", [i_var]))
    eq_body = Equation.simple(der(x_i), v_i)

    # Create for loop
    range_expr = Expr.slice(Expr.literal(1), Expr.literal(3))
    for_eq = Equation.for_loop("i", range_expr, [eq_body])

    assert for_eq.eq_type.name == "FOR"
    assert for_eq.index_var == "i"
    assert isinstance(for_eq.range_expr, Slice)
    assert len(for_eq.for_equations) == 1
    assert str(for_eq) == "for i in 1:3 loop der(x[i]) = v[i] end for"


def test_for_equation_multiple_statements():
    """Test for equation with multiple statements in body."""
    i_var = Expr.var_ref("i")

    # der(x[i]) = v[i]
    x_i = Expr.component_ref(("x", [i_var]))
    v_i = Expr.component_ref(("v", [i_var]))
    eq1 = Equation.simple(der(x_i), v_i)

    # y[i] = 2 * x[i]
    y_i = Expr.component_ref(("y", [i_var]))
    eq2 = Equation.simple(y_i, Expr.mul(Expr.literal(2), x_i))

    # for i in 1:n loop ... end for
    n = Expr.var_ref("n")
    range_expr = Expr.slice(Expr.literal(1), n)
    for_eq = Equation.for_loop("i", range_expr, [eq1, eq2])

    assert len(for_eq.for_equations) == 2
    assert "der(x[i]) = v[i]" in str(for_eq)
    assert "y[i]" in str(for_eq)


def test_nested_for_equations():
    """Test nested for loops for matrix operations."""
    # for i in 1:3 loop
    #   for j in 1:3 loop
    #     A[i,j] = B[i,j] + C[i,j]
    #   end for
    # end for

    i_var = Expr.var_ref("i")
    j_var = Expr.var_ref("j")

    # Inner loop body: A[i,j] = B[i,j] + C[i,j]
    A_ij = Expr.component_ref(("A", [i_var, j_var]))
    B_ij = Expr.component_ref(("B", [i_var, j_var]))
    C_ij = Expr.component_ref(("C", [i_var, j_var]))
    inner_eq = Equation.simple(A_ij, Expr.add(B_ij, C_ij))

    # Inner for loop: for j in 1:3
    inner_range = Expr.slice(Expr.literal(1), Expr.literal(3))
    inner_for = Equation.for_loop("j", inner_range, [inner_eq])

    # Outer for loop: for i in 1:3
    outer_range = Expr.slice(Expr.literal(1), Expr.literal(3))
    outer_for = Equation.for_loop("i", outer_range, [inner_for])

    assert outer_for.index_var == "i"
    assert len(outer_for.for_equations) == 1
    assert outer_for.for_equations[0].eq_type.name == "FOR"
    assert outer_for.for_equations[0].index_var == "j"


def test_for_equation_in_model():
    """Test adding for equation to a model."""
    model = Model(name="ForLoopModel")

    # Add array variables
    model.add_variable(Variable(name="x", var_type=VariableType.STATE, shape=[3]))
    model.add_variable(Variable(name="v", var_type=VariableType.STATE, shape=[3]))

    # Create for equation: for i in 1:3 loop der(x[i]) = v[i]; end for
    i_var = Expr.var_ref("i")
    x_i = Expr.component_ref(("x", [i_var]))
    v_i = Expr.component_ref(("v", [i_var]))
    eq_body = Equation.simple(der(x_i), v_i)

    range_expr = Expr.slice(Expr.literal(1), Expr.literal(3))
    for_eq = Equation.for_loop("i", range_expr, [eq_body])

    model.add_equation(for_eq)

    assert len(model.equations) == 1
    assert model.equations[0].eq_type.name == "FOR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
