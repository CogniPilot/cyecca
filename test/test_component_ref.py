"""
Tests for ComponentRef (hierarchical component references).

This is Phase 2 of the IR migration - adding support for Modelica-style
hierarchical component references like vehicle.engine.temp[i].
"""

import pytest
from cyecca.ir import (
    Expr,
    ComponentRef,
    ComponentRefPart,
    VarRef,
    ArrayRef,
    Equation,
    der,
)


def test_simple_component_ref():
    """Test simple variable reference."""
    x = Expr.var_ref("x")

    assert isinstance(x, ComponentRef)
    assert x.is_simple
    assert x.simple_name == "x"
    assert str(x) == "x"
    assert len(x.parts) == 1
    assert x.parts[0].name == "x"
    assert len(x.parts[0].subscripts) == 0


def test_hierarchical_component_ref():
    """Test hierarchical reference."""
    ref = Expr.component_ref("vehicle", "engine", "temp")

    assert isinstance(ref, ComponentRef)
    assert len(ref.parts) == 3
    assert str(ref) == "vehicle.engine.temp"
    assert not ref.is_simple
    assert ref.parts[0].name == "vehicle"
    assert ref.parts[1].name == "engine"
    assert ref.parts[2].name == "temp"


def test_component_ref_with_subscripts():
    """Test array indexing in hierarchy."""
    i = Expr.literal(1)
    ref = Expr.component_ref(("positions", [i]), "x")

    assert isinstance(ref, ComponentRef)
    assert len(ref.parts) == 2
    assert ref.parts[0].name == "positions"
    assert len(ref.parts[0].subscripts) == 1
    assert ref.parts[1].name == "x"
    assert str(ref) == "positions[1].x"
    assert not ref.is_simple


def test_component_ref_with_multiple_subscripts():
    """Test multi-dimensional array indexing."""
    i = Expr.literal(1)
    j = Expr.literal(2)
    ref = Expr.component_ref(("matrix", [i, j]))

    assert isinstance(ref, ComponentRef)
    assert len(ref.parts) == 1
    assert ref.parts[0].name == "matrix"
    assert len(ref.parts[0].subscripts) == 2
    assert str(ref) == "matrix[1,2]"
    assert not ref.is_simple  # Not simple because it has subscripts


def test_component_ref_complex_hierarchy():
    """Test complex hierarchical reference with multiple subscripts."""
    i = Expr.literal(0)
    ref = Expr.component_ref("vehicle", ("wheels", [i]), "pressure")

    assert len(ref.parts) == 3
    assert ref.parts[0].name == "vehicle"
    assert ref.parts[1].name == "wheels"
    assert len(ref.parts[1].subscripts) == 1
    assert ref.parts[2].name == "pressure"
    assert str(ref) == "vehicle.wheels[0].pressure"


def test_varref_backward_compatibility():
    """Test old VarRef still works."""
    x = VarRef("x")

    assert x.name == "x"
    assert str(x) == "x"

    # Can convert to ComponentRef
    comp_ref = x.to_component_ref()
    assert isinstance(comp_ref, ComponentRef)
    assert comp_ref.is_simple
    assert comp_ref.simple_name == "x"


def test_arrayref_backward_compatibility():
    """Test old ArrayRef still works."""
    i = Expr.literal(1)
    j = Expr.literal(2)
    arr = ArrayRef("matrix", (i, j))

    assert arr.name == "matrix"
    assert len(arr.indices) == 2
    assert str(arr) == "matrix[1, 2]"

    # Can convert to ComponentRef
    comp_ref = arr.to_component_ref()
    assert isinstance(comp_ref, ComponentRef)
    assert comp_ref.parts[0].name == "matrix"
    assert len(comp_ref.parts[0].subscripts) == 2


def test_componentref_to_varref():
    """Test converting simple ComponentRef to legacy VarRef."""
    x = Expr.var_ref("x")

    # Can convert simple refs to VarRef
    var_ref = x.to_varref()
    assert isinstance(var_ref, VarRef)
    assert var_ref.name == "x"

    # Cannot convert hierarchical refs
    hierarchical = Expr.component_ref("a", "b")
    with pytest.raises(ValueError, match="Cannot convert hierarchical"):
        hierarchical.to_varref()


def test_simple_name_error():
    """Test that simple_name raises on hierarchical ref."""
    hierarchical = Expr.component_ref("a", "b")
    with pytest.raises(ValueError, match="Not a simple reference"):
        _ = hierarchical.simple_name


def test_equation_with_hierarchical_ref():
    """Test equation with hierarchical reference."""
    temp = Expr.component_ref("vehicle", "engine", "temp")
    ambient = Expr.var_ref("T_ambient")
    k = Expr.var_ref("k")

    # der(vehicle.engine.temp) = k * (T_ambient - vehicle.engine.temp)
    eq = Equation.simple(der(temp), Expr.mul(k, Expr.sub(ambient, temp)))

    assert eq.lhs.func == "der"
    assert isinstance(eq.lhs.args[0], ComponentRef)
    assert str(eq.lhs.args[0]) == "vehicle.engine.temp"


def test_der_with_hierarchical_ref():
    """Test der() with hierarchical references."""
    pos = Expr.component_ref("vehicle", "position", "x")
    dpos = der(pos)

    assert dpos.func == "der"
    assert len(dpos.args) == 1
    assert isinstance(dpos.args[0], ComponentRef)
    assert str(dpos.args[0]) == "vehicle.position.x"


def test_der_with_subscripted_ref():
    """Test der() with array subscripts."""
    i = Expr.literal(1)
    pos_i = Expr.component_ref(("positions", [i]))
    dpos_i = der(pos_i)

    assert dpos_i.func == "der"
    assert str(dpos_i.args[0]) == "positions[1]"


def test_component_ref_part_str():
    """Test string representation of ComponentRefPart."""
    # Without subscripts
    part1 = ComponentRefPart("x")
    assert str(part1) == "x"

    # With single subscript
    i = Expr.literal(1)
    part2 = ComponentRefPart("arr", (i,))
    assert str(part2) == "arr[1]"

    # With multiple subscripts
    j = Expr.literal(2)
    part3 = ComponentRefPart("matrix", (i, j))
    assert str(part3) == "matrix[1,2]"


def test_backward_compatible_var_ref():
    """Test that Expr.var_ref now returns ComponentRef but is backward compatible."""
    x = Expr.var_ref("x")

    # Should be ComponentRef, not VarRef
    assert isinstance(x, ComponentRef)

    # But should act like a simple variable reference
    assert x.is_simple
    assert x.simple_name == "x"
    assert str(x) == "x"

    # Can be used in equations just like before
    v = Expr.var_ref("v")
    eq = Equation.simple(der(x), v)
    assert eq.lhs.func == "der"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
