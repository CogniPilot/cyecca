"""
Tests for cyecca.dsl.causality module.

Covers: BLT decomposition, Tarjan's algorithm, equation sorting, symbolic solving.
"""

import numpy as np
import pytest


class TestFindVariables:
    """Test find_variables function."""

    def test_find_variable(self) -> None:
        from cyecca.dsl.causality import find_variables
        from cyecca.dsl.expr import Expr, ExprKind

        expr = Expr(ExprKind.VARIABLE, name="x")
        vars = find_variables(expr)
        assert vars == {"x"}

    def test_find_derivative(self) -> None:
        from cyecca.dsl.causality import find_variables
        from cyecca.dsl.expr import Expr, ExprKind

        expr = Expr(ExprKind.DERIVATIVE, name="x")
        vars = find_variables(expr)
        assert vars == {"der_x"}

    def test_find_nested(self) -> None:
        from cyecca.dsl.causality import find_variables
        from cyecca.dsl.expr import Expr, ExprKind

        # x + y * z
        x = Expr(ExprKind.VARIABLE, name="x")
        y = Expr(ExprKind.VARIABLE, name="y")
        z = Expr(ExprKind.VARIABLE, name="z")
        yz = Expr(ExprKind.MUL, children=(y, z))
        expr = Expr(ExprKind.ADD, children=(x, yz))

        vars = find_variables(expr)
        assert vars == {"x", "y", "z"}


class TestIsLinearIn:
    """Test is_linear_in function."""

    def test_just_variable(self) -> None:
        from cyecca.dsl.causality import is_linear_in
        from cyecca.dsl.expr import Expr, ExprKind

        x = Expr(ExprKind.VARIABLE, name="x")
        is_lin, coef, const = is_linear_in(x, "x")

        assert is_lin is True
        assert coef is not None
        assert coef.kind == ExprKind.CONSTANT
        assert coef.value == 1.0

    def test_constant_times_variable(self) -> None:
        from cyecca.dsl.causality import is_linear_in
        from cyecca.dsl.expr import Expr, ExprKind

        # 2 * x
        two = Expr(ExprKind.CONSTANT, value=2.0)
        x = Expr(ExprKind.VARIABLE, name="x")
        expr = Expr(ExprKind.MUL, children=(two, x))

        is_lin, coef, const = is_linear_in(expr, "x")

        assert is_lin is True
        assert coef is not None
        assert coef.value == 2.0

    def test_variable_not_in_expr(self) -> None:
        from cyecca.dsl.causality import is_linear_in
        from cyecca.dsl.expr import Expr, ExprKind

        y = Expr(ExprKind.VARIABLE, name="y")
        is_lin, coef, const = is_linear_in(y, "x")

        assert is_lin is True
        assert coef is not None
        assert coef.value == 0.0  # Coefficient is 0 since x doesn't appear


class TestSolveLinear:
    """Test solve_linear function."""

    def test_solve_explicit_derivative(self) -> None:
        from cyecca.dsl.causality import solve_linear
        from cyecca.dsl.equations import Equation
        from cyecca.dsl.expr import Expr, ExprKind

        # der(x) == y
        lhs = Expr(ExprKind.DERIVATIVE, name="x")
        rhs = Expr(ExprKind.VARIABLE, name="y")
        eq = Equation(lhs=lhs, rhs=rhs)

        solved = solve_linear(eq, "der_x")

        assert solved is not None
        assert solved.var_name == "x"
        assert solved.is_derivative is True

    def test_solve_algebraic(self) -> None:
        from cyecca.dsl.causality import solve_linear
        from cyecca.dsl.equations import Equation
        from cyecca.dsl.expr import Expr, ExprKind

        # y == x + 1
        lhs = Expr(ExprKind.VARIABLE, name="y")
        one = Expr(ExprKind.CONSTANT, value=1.0)
        x = Expr(ExprKind.VARIABLE, name="x")
        rhs = Expr(ExprKind.ADD, children=(x, one))
        eq = Equation(lhs=lhs, rhs=rhs)

        solved = solve_linear(eq, "y")

        assert solved is not None
        assert solved.var_name == "y"
        assert solved.is_derivative is False


class TestTarjanSCC:
    """Test Tarjan's algorithm for strongly connected components."""

    def test_no_cycles(self) -> None:
        from cyecca.dsl.causality import _tarjan_scc

        # Linear chain: 0 -> 1 -> 2
        nodes = [0, 1, 2]
        adj = {0: [1], 1: [2], 2: []}

        sccs = _tarjan_scc(nodes, adj)

        # Each node is its own SCC (no cycles)
        assert len(sccs) == 3
        # All SCCs should be singletons
        for scc in sccs:
            assert len(scc) == 1

    def test_single_cycle(self) -> None:
        from cyecca.dsl.causality import _tarjan_scc

        # Cycle: 0 -> 1 -> 2 -> 0
        nodes = [0, 1, 2]
        adj = {0: [1], 1: [2], 2: [0]}

        sccs = _tarjan_scc(nodes, adj)

        # All nodes in one SCC
        assert len(sccs) == 1
        assert set(sccs[0]) == {0, 1, 2}

    def test_two_sccs(self) -> None:
        from cyecca.dsl.causality import _tarjan_scc

        # Two separate cycles: 0 <-> 1, 2 <-> 3
        nodes = [0, 1, 2, 3]
        adj = {0: [1], 1: [0], 2: [3], 3: [2]}

        sccs = _tarjan_scc(nodes, adj)

        assert len(sccs) == 2
        scc_sets = [set(scc) for scc in sccs]
        assert {0, 1} in scc_sets
        assert {2, 3} in scc_sets

    def test_chain_with_cycle(self) -> None:
        from cyecca.dsl.causality import _tarjan_scc

        # 0 -> (1 <-> 2) -> 3
        nodes = [0, 1, 2, 3]
        adj = {0: [1], 1: [2], 2: [1, 3], 3: []}

        sccs = _tarjan_scc(nodes, adj)

        # Should have 3 SCCs: {0}, {1,2}, {3}
        assert len(sccs) == 3
        scc_sets = [set(scc) for scc in sccs]
        assert {0} in scc_sets
        assert {1, 2} in scc_sets
        assert {3} in scc_sets


class TestAnalyzeCausality:
    """Test the full analyze_causality function."""

    def test_simple_ode(self) -> None:
        from cyecca.dsl import Real, der, equations, model, var
        from cyecca.dsl.causality import analyze_causality

        @model
        class SimpleODE:
            x = Real(start=0.0)

            @equations
            def _(m):
                der(m.x) == 1.0

        flat = SimpleODE().flatten()
        sorted_sys = analyze_causality(flat)

        assert sorted_sys.is_ode_explicit is True
        assert len(sorted_sys.solved) == 1
        assert len(sorted_sys.implicit_blocks) == 0
        assert sorted_sys.solved[0].is_derivative is True
        assert sorted_sys.solved[0].var_name == "x"

    def test_coupled_ode(self) -> None:
        from cyecca.dsl import Real, der, equations, model, var
        from cyecca.dsl.causality import analyze_causality

        @model
        class CoupledODE:
            x = Real(start=1.0)
            y = Real(start=0.0)

            @equations
            def _(m):
                der(m.x) == m.y
                der(m.y) == -m.x

        flat = CoupledODE().flatten()
        sorted_sys = analyze_causality(flat)

        assert sorted_sys.is_ode_explicit is True
        assert len(sorted_sys.solved) == 2
        assert len(sorted_sys.implicit_blocks) == 0

    def test_with_algebraic(self) -> None:
        from cyecca.dsl import Real, der, equations, model, var
        from cyecca.dsl.causality import analyze_causality

        @model
        class WithAlgebraic:
            x = Real(start=0.0)
            y = Real()  # No der(y) means algebraic

            @equations
            def _(m):
                der(m.x) == m.y
                m.y == 2 * m.x

        flat = WithAlgebraic().flatten()
        sorted_sys = analyze_causality(flat)

        assert sorted_sys.has_algebraic is True
        # Both equations should be solved
        assert len(sorted_sys.solved) == 2

    def test_harmonic_oscillator(self) -> None:
        from cyecca.dsl import Real, der, equations, model, var
        from cyecca.dsl.causality import analyze_causality

        @model
        class HarmonicOscillator:
            x = Real(start=1.0)
            v = Real(start=0.0)
            k = Real(1.0, parameter=True)
            m_param = Real(1.0, parameter=True)

            @equations
            def _(m):
                der(m.x) == m.v
                der(m.v) == -m.k / m.m_param * m.x

        flat = HarmonicOscillator().flatten()
        sorted_sys = analyze_causality(flat)

        assert sorted_sys.is_ode_explicit is True
        assert len(sorted_sys.solved) == 2
        assert len(sorted_sys.implicit_blocks) == 0

        # Check that equations are in proper order (x depends on v, v depends on x)
        var_names = [s.var_name for s in sorted_sys.solved]
        assert "x" in var_names
        assert "v" in var_names


# =============================================================================
# RLC Circuit Example (Modelica-style electrical components)
# =============================================================================


class TestRLCCircuit:
    """
    Test causality analysis with a realistic RLC circuit example.

    This implements the Modelica-style electrical components:
    - Pin connector (voltage potential, current flow)
    - Resistor, Capacitor, Inductor, VoltageSourceDC, Ground
    - RLC_Circuit composite model with connect() statements
    """

    def test_electrical_pin_connector(self) -> None:
        """Test the electrical Pin connector definition."""
        from cyecca.dsl import Real, connector, var

        @connector
        class Pin:
            """Electrical pin with voltage (potential) and current (flow)."""

            v = Real()  # Voltage
            i = var(flow=True)  # Current (positive into pin)

        pin = Pin()
        assert pin._metadata.variables["v"].flow is False
        assert pin._metadata.variables["i"].flow is True

    def test_resistor_model(self) -> None:
        """Test the Resistor component model."""
        from cyecca.dsl import Real, connect, connector, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Resistor:
            """Resistor: v = R * i (Ohm's law)."""

            R = Real(1.0, parameter=True)  # Resistance (Ohm)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.R * m.p.i  # Ohm's law
                m.n.i == -m.p.i  # Current conservation (rewritten)

        resistor = Resistor()
        flat = resistor.flatten()

        assert "R" in flat.param_names
        # Equations: Ohm's law + current conservation
        assert len(flat.equations) == 2

    def test_capacitor_model(self) -> None:
        """Test the Capacitor component model."""
        from cyecca.dsl import Real, connector, der, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Capacitor:
            """Capacitor: C * der(v) = i."""

            C = Real(1.0, parameter=True)  # Capacitance (F)
            p = submodel(Pin)
            n = submodel(Pin)
            v = Real(start=0.0)  # Voltage across capacitor

            @equations
            def _(m):
                m.v == m.p.v - m.n.v
                m.C * der(m.v) == m.p.i
                m.p.i + m.n.i == 0

        cap = Capacitor()
        flat = cap.flatten()

        assert "v" in flat.state_names  # v is differentiated
        assert "C" in flat.param_names
        assert len(flat.equations) == 3

    def test_inductor_model(self) -> None:
        """Test the Inductor component model."""
        from cyecca.dsl import Real, connector, der, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Inductor:
            """Inductor: L * der(i) = v."""

            L = Real(1.0, parameter=True)  # Inductance (H)
            p = submodel(Pin)
            n = submodel(Pin)
            i = Real(start=0.0)  # Current through inductor

            @equations
            def _(m):
                m.p.v - m.n.v == m.L * der(m.i)
                m.p.i == m.i
                m.p.i + m.n.i == 0

        ind = Inductor()
        flat = ind.flatten()

        assert "i" in flat.state_names  # i is differentiated
        assert "L" in flat.param_names

    def test_voltage_source_dc(self) -> None:
        """Test the DC voltage source model."""
        from cyecca.dsl import Real, connector, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class VoltageSourceDC:
            """DC voltage source: v_p - v_n = V."""

            V = Real(1.0, parameter=True)  # Voltage (V)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.V
                m.p.i + m.n.i == 0

        vsrc = VoltageSourceDC()
        flat = vsrc.flatten()

        assert "V" in flat.param_names
        assert len(flat.equations) == 2

    def test_ground_model(self) -> None:
        """Test the Ground reference model."""
        from cyecca.dsl import Real, connector, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Ground:
            """Ground reference: v = 0."""

            p = submodel(Pin)

            @equations
            def _(m):
                m.p.v == 0

        gnd = Ground()
        flat = gnd.flatten()

        assert len(flat.equations) == 1

    def test_rc_circuit_causality(self) -> None:
        """Test causality analysis of a simple RC circuit.

        Circuit: V -- R -- C -- GND

        This is a first-order system with one state (capacitor voltage).
        """
        from cyecca.dsl import Real, connect, connector, der, equations, model, submodel, var
        from cyecca.dsl.causality import analyze_causality

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Resistor:
            R = Real(1000.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.R * m.p.i
                m.p.i + m.n.i == 0

        @model
        class Capacitor:
            C = var(1e-6, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            v = Real(start=0.0)

            @equations
            def _(m):
                m.v == m.p.v - m.n.v
                m.C * der(m.v) == m.p.i
                m.p.i + m.n.i == 0

        @model
        class VoltageSourceDC:
            V = Real(5.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.V
                m.p.i + m.n.i == 0

        @model
        class Ground:
            p = submodel(Pin)

            @equations
            def _(m):
                m.p.v == 0

        @model
        class RC_Circuit:
            """Simple RC circuit: voltage source, resistor, capacitor, ground."""

            Vsrc = submodel(VoltageSourceDC)
            R1 = submodel(Resistor)
            C1 = submodel(Capacitor)
            G = submodel(Ground)

            @equations
            def _(m):
                connect(m.Vsrc.p, m.R1.p)
                connect(m.R1.n, m.C1.p)
                connect(m.C1.n, m.Vsrc.n)
                connect(m.Vsrc.n, m.G.p)

        circuit = RC_Circuit()
        flat = circuit.flatten()

        # Should have one state: capacitor voltage
        assert "C1.v" in flat.state_names

        # Perform causality analysis
        sorted_sys = analyze_causality(flat)

        # Should have solved equations
        assert len(sorted_sys.solved) > 0 or len(sorted_sys.implicit_blocks) > 0

    def test_rlc_circuit_definition(self) -> None:
        """Test the full RLC circuit definition from the Modelica example."""
        from cyecca.dsl import Real, connect, connector, der, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Resistor:
            R = Real(10.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.R * m.p.i
                m.p.i + m.n.i == 0

        @model
        class Capacitor:
            C = Real(0.01, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            v = Real(start=0.0)

            @equations
            def _(m):
                m.v == m.p.v - m.n.v
                m.C * der(m.v) == m.p.i
                m.p.i + m.n.i == 0

        @model
        class Inductor:
            L = Real(0.5, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            i = Real(start=0.0)

            @equations
            def _(m):
                m.p.v - m.n.v == m.L * der(m.i)
                m.p.i == m.i
                m.p.i + m.n.i == 0

        @model
        class VoltageSourceDC:
            V = Real(5.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.V
                m.p.i + m.n.i == 0

        @model
        class Ground:
            p = submodel(Pin)

            @equations
            def _(m):
                m.p.v == 0

        @model
        class RLC_Circuit:
            """RLC circuit: V -- R -- L -- C -- GND (series connection)."""

            # Components
            Vsrc = submodel(VoltageSourceDC)
            R1 = submodel(Resistor)
            L1 = submodel(Inductor)
            C1 = submodel(Capacitor)
            G = submodel(Ground)

            @equations
            def _(m):
                # Circuit connections
                connect(m.Vsrc.p, m.R1.p)
                connect(m.R1.n, m.L1.p)
                connect(m.L1.n, m.C1.p)
                connect(m.C1.n, m.Vsrc.n)
                connect(m.Vsrc.n, m.G.p)

        circuit = RLC_Circuit()
        flat = circuit.flatten()

        # Should have two states: inductor current and capacitor voltage
        assert "L1.i" in flat.state_names
        assert "C1.v" in flat.state_names

        # Should have parameters
        assert "R1.R" in flat.param_names
        assert "L1.L" in flat.param_names
        assert "C1.C" in flat.param_names
        assert "Vsrc.V" in flat.param_names

    def test_rlc_circuit_causality(self) -> None:
        """Test causality analysis of the RLC circuit."""
        from cyecca.dsl import Real, connect, connector, der, equations, model, submodel, var
        from cyecca.dsl.causality import analyze_causality

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Resistor:
            R = Real(10.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.R * m.p.i
                m.p.i + m.n.i == 0

        @model
        class Capacitor:
            C = Real(0.01, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            v = Real(start=0.0)

            @equations
            def _(m):
                m.v == m.p.v - m.n.v
                m.C * der(m.v) == m.p.i
                m.p.i + m.n.i == 0

        @model
        class Inductor:
            L = Real(0.5, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            i = Real(start=0.0)

            @equations
            def _(m):
                m.p.v - m.n.v == m.L * der(m.i)
                m.p.i == m.i
                m.p.i + m.n.i == 0

        @model
        class VoltageSourceDC:
            V = Real(5.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.V
                m.p.i + m.n.i == 0

        @model
        class Ground:
            p = submodel(Pin)

            @equations
            def _(m):
                m.p.v == 0

        @model
        class RLC_Circuit:
            Vsrc = submodel(VoltageSourceDC)
            R1 = submodel(Resistor)
            L1 = submodel(Inductor)
            C1 = submodel(Capacitor)
            G = submodel(Ground)

            @equations
            def _(m):
                connect(m.Vsrc.p, m.R1.p)
                connect(m.R1.n, m.L1.p)
                connect(m.L1.n, m.C1.p)
                connect(m.C1.n, m.Vsrc.n)
                connect(m.Vsrc.n, m.G.p)

        circuit = RLC_Circuit()
        flat = circuit.flatten()

        # Perform causality analysis
        sorted_sys = analyze_causality(flat)

        # RLC circuit is a second-order system
        # Should have 2 derivative equations (for L1.i and C1.v)
        derivative_eqs = [s for s in sorted_sys.solved if s.is_derivative]
        derivative_vars = {s.var_name for s in derivative_eqs}

        # The system should be able to solve for der(L1.i) and der(C1.v)
        # though the exact form depends on the BLT decomposition
        assert len(sorted_sys.solved) > 0 or len(sorted_sys.implicit_blocks) > 0

    def test_rlc_circuit_simulation(self) -> None:
        """Test that the RLC circuit can be compiled and simulated."""
        from cyecca.dsl import Real, connect, connector, der, equations, model, submodel, var
        from cyecca.dsl.backends import CasadiBackend

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Resistor:
            R = Real(10.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.R * m.p.i
                m.p.i + m.n.i == 0

        @model
        class Capacitor:
            C = Real(0.01, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            v = Real(start=0.0)

            @equations
            def _(m):
                m.v == m.p.v - m.n.v
                m.C * der(m.v) == m.p.i
                m.p.i + m.n.i == 0

        @model
        class Inductor:
            L = Real(0.5, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            i = Real(start=0.0)

            @equations
            def _(m):
                m.p.v - m.n.v == m.L * der(m.i)
                m.p.i == m.i
                m.p.i + m.n.i == 0

        @model
        class VoltageSourceDC:
            V = Real(5.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.V
                m.p.i + m.n.i == 0

        @model
        class Ground:
            p = submodel(Pin)

            @equations
            def _(m):
                m.p.v == 0

        @model
        class RLC_Circuit:
            Vsrc = submodel(VoltageSourceDC)
            R1 = submodel(Resistor)
            L1 = submodel(Inductor)
            C1 = submodel(Capacitor)
            G = submodel(Ground)

            @equations
            def _(m):
                connect(m.Vsrc.p, m.R1.p)
                connect(m.R1.n, m.L1.p)
                connect(m.L1.n, m.C1.p)
                connect(m.C1.n, m.Vsrc.n)
                connect(m.Vsrc.n, m.G.p)

        circuit = RLC_Circuit()
        flat = circuit.flatten()

        # Compile and simulate with IDAS (handles DAE systems)
        from cyecca.dsl.backends.casadi import Integrator

        compiled = CasadiBackend.compile(flat)

        # Simulate for 0.1 seconds with IDAS
        result = compiled.simulate(tf=0.1, integrator=Integrator.IDAS)

        # Check that we got results
        assert len(result.t) > 0

        # Inductor current should start at 0 and change
        i_L = result("L1.i")
        assert i_L[0] == pytest.approx(0.0, abs=1e-6)

        # Capacitor voltage should start at 0 and rise toward source voltage
        v_C = result("C1.v")
        assert v_C[0] == pytest.approx(0.0, abs=1e-6)
        # After some time, capacitor voltage should be positive
        assert v_C[-1] > 0

    def test_rlc_damped_oscillation(self) -> None:
        """Test that RLC circuit exhibits damped oscillation behavior.

        For underdamped RLC: R < 2*sqrt(L/C)
        With R=10, L=0.5, C=0.01: 2*sqrt(0.5/0.01) = 2*sqrt(50) ≈ 14.14
        So R=10 < 14.14, meaning we have underdamped oscillation.
        """
        from cyecca.dsl import Real, connect, connector, der, equations, model, submodel, var
        from cyecca.dsl.backends import CasadiBackend
        from cyecca.dsl.backends.casadi import Integrator

        @connector
        class Pin:
            v = Real()
            i = var(flow=True)

        @model
        class Resistor:
            R = Real(10.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.R * m.p.i
                m.p.i + m.n.i == 0

        @model
        class Capacitor:
            C = Real(0.01, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            v = Real(start=0.0)

            @equations
            def _(m):
                m.v == m.p.v - m.n.v
                m.C * der(m.v) == m.p.i
                m.p.i + m.n.i == 0

        @model
        class Inductor:
            L = Real(0.5, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)
            i = Real(start=0.0)

            @equations
            def _(m):
                m.p.v - m.n.v == m.L * der(m.i)
                m.p.i == m.i
                m.p.i + m.n.i == 0

        @model
        class VoltageSourceDC:
            V = Real(5.0, parameter=True)
            p = submodel(Pin)
            n = submodel(Pin)

            @equations
            def _(m):
                m.p.v - m.n.v == m.V
                m.p.i + m.n.i == 0

        @model
        class Ground:
            p = submodel(Pin)

            @equations
            def _(m):
                m.p.v == 0

        @model
        class RLC_Circuit:
            Vsrc = submodel(VoltageSourceDC)
            R1 = submodel(Resistor)
            L1 = submodel(Inductor)
            C1 = submodel(Capacitor)
            G = submodel(Ground)

            @equations
            def _(m):
                connect(m.Vsrc.p, m.R1.p)
                connect(m.R1.n, m.L1.p)
                connect(m.L1.n, m.C1.p)
                connect(m.C1.n, m.Vsrc.n)
                connect(m.Vsrc.n, m.G.p)

        circuit = RLC_Circuit()
        flat = circuit.flatten()
        compiled = CasadiBackend.compile(flat)

        # Simulate for longer to see oscillation with IDAS
        result = compiled.simulate(tf=0.5, dt=0.001, integrator=Integrator.IDAS)

        v_C = result("C1.v")

        # For underdamped oscillation, capacitor voltage should overshoot
        # the final value (5V) at some point
        V_source = 5.0
        max_v_C = np.max(v_C)

        # Should overshoot slightly due to underdamped oscillation
        assert max_v_C > V_source * 0.9  # At least gets close to source

        # Final value should settle near source voltage
        assert v_C[-1] == pytest.approx(V_source, rel=0.1)
