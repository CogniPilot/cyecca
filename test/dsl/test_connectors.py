"""
Tests for Modelica connector support (MLS Chapter 9).

Covers:
- @connector decorator
- flow=True variable prefix
- connect() function for connection equations
- Connector balancing restriction
"""

import pytest


class TestConnectorDecorator:
    """Test the @connector decorator."""

    def test_basic_connector(self) -> None:
        """Test basic connector with potential and flow variable."""
        from cyecca.dsl import Real, connector, var

        @connector
        class Pin:
            """Electrical pin."""

            v = Real()  # Potential (voltage)
            i = Real(flow=True)  # Flow (current)

        # Should have _is_connector marker
        assert getattr(Pin, "_is_connector", False) is True

        # Create instance and check metadata
        pin = Pin()
        assert "v" in pin._metadata.variables
        assert "i" in pin._metadata.variables
        assert pin._metadata.variables["i"].flow is True
        assert pin._metadata.variables["v"].flow is False

    def test_mechanical_connector(self) -> None:
        """Test mechanical flange connector."""
        from cyecca.dsl import Real, connector, var

        @connector
        class Flange:
            """Mechanical translational flange."""

            s = Real()  # Position (potential)
            f = Real(flow=True)  # Force (flow)

        flange = Flange()
        assert flange._metadata.variables["s"].flow is False
        assert flange._metadata.variables["f"].flow is True

    def test_connector_with_parameters(self) -> None:
        """Test that parameters don't count in balancing."""
        from cyecca.dsl import Real, connector, var

        @connector
        class ParameterizedPin:
            """Pin with a parameter."""

            v = Real()
            i = Real(flow=True)
            R_internal = Real(0.01, parameter=True)  # Parameters don't affect balance

        pin = ParameterizedPin()
        assert pin._metadata.variables["R_internal"].parameter is True

    def test_connector_balancing_violation(self) -> None:
        """Test that unbalanced connectors raise TypeError."""
        from cyecca.dsl import Real, connector, var

        with pytest.raises(TypeError, match="balancing violation"):

            @connector
            class UnbalancedPin:
                """Invalid: 1 potential, 2 flow."""

                v = Real()
                i1 = Real(flow=True)
                i2 = Real(flow=True)

    def test_connector_no_equations(self) -> None:
        """Test that connectors cannot have @equations."""
        from cyecca.dsl import Real, connector, equations, var

        with pytest.raises(TypeError, match="cannot have @equations"):

            @connector
            class BadConnector:
                v = Real()
                i = Real(flow=True)

                @equations
                def _(m):
                    pass

    def test_connector_no_algorithm(self) -> None:
        """Test that connectors cannot have @algorithm."""
        from cyecca.dsl import Real, algorithm, connector, var

        with pytest.raises(TypeError, match="cannot have @algorithm"):

            @connector
            class BadConnector:
                v = Real()
                i = Real(flow=True)

                @algorithm
                def _(m):
                    pass

    def test_connector_no_submodels(self) -> None:
        """Test that connectors cannot have submodels."""
        from cyecca.dsl import Real, connector, model, submodel, var

        @model
        class Inner:
            x = Real()

        with pytest.raises(TypeError, match="cannot have submodels"):

            @connector
            class BadConnector:
                v = Real()
                i = Real(flow=True)
                inner = submodel(Inner)

    def test_connector_array_balancing(self) -> None:
        """Test that array variables are properly counted for balancing."""
        from cyecca.dsl import Real, connector, var

        @connector
        class VectorPin:
            """3D mechanical connector."""

            x = Real(shape=(3,))  # 3 potential variables
            f = Real(shape=(3,), flow=True)  # 3 flow variables

        pin = VectorPin()
        assert pin._metadata.variables["x"].size == 3
        assert pin._metadata.variables["f"].size == 3

    def test_connector_array_imbalance(self) -> None:
        """Test array size mismatch causes balancing error."""
        from cyecca.dsl import Real, connector, var

        with pytest.raises(TypeError, match="balancing violation"):

            @connector
            class BadVectorPin:
                x = Real(shape=(3,))  # 3 potential
                f = Real(shape=(2,), flow=True)  # Only 2 flow


class TestConnectFunction:
    """Test the connect() function."""

    def test_connect_generates_equations(self) -> None:
        """Test that connect() generates proper equations."""
        from cyecca.dsl import Real, connect, connector, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = Real(flow=True)

        @model
        class TwoPins:
            p1 = submodel(Pin)
            p2 = submodel(Pin)

            @equations
            def _(m):
                connect(m.p1, m.p2)

        model_instance = TwoPins()
        eqs = model_instance.get_equations()

        # Should generate 2 equations: v equality and i sum-to-zero
        assert len(eqs) == 2

    def test_connect_outside_equations_raises(self) -> None:
        """Test that connect() outside @equations raises RuntimeError."""
        from cyecca.dsl import Real, connect, connector, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = Real(flow=True)

        @model
        class TwoPins:
            p1 = submodel(Pin)
            p2 = submodel(Pin)

        m = TwoPins()
        with pytest.raises(RuntimeError, match="inside an @equations block"):
            connect(m.p1, m.p2)

    def test_connect_non_connector_raises(self) -> None:
        """Test that connecting non-connectors raises TypeError."""
        from cyecca.dsl import Real, connect, equations, model, submodel, var

        @model
        class NotAConnector:
            x = Real()

        @model
        class System:
            a = submodel(NotAConnector)
            b = submodel(NotAConnector)

            @equations
            def _(m):
                connect(m.a, m.b)

        system = System()
        with pytest.raises(TypeError, match="not a connector"):
            system.get_equations()

    def test_connect_mismatched_connectors_raises(self) -> None:
        """Test that connecting different connector types raises TypeError."""
        from cyecca.dsl import Real, connect, connector, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = Real(flow=True)

        @connector
        class OtherPin:
            voltage = Real()  # Different name!
            current = Real(flow=True)

        @model
        class System:
            p1 = submodel(Pin)
            p2 = submodel(OtherPin)

            @equations
            def _(m):
                connect(m.p1, m.p2)

        system = System()
        with pytest.raises(TypeError, match="different variables"):
            system.get_equations()


class TestConnectorIntegration:
    """Integration tests for connectors in complete models."""

    def test_resistor_model(self) -> None:
        """Test a simple resistor model with two pins."""
        from cyecca.dsl import Real, connector, der, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = Real(flow=True)

        @model
        class Resistor:
            """Simple resistor: v = R * i."""

            p = submodel(Pin)
            n = submodel(Pin)
            R = Real(1000.0, parameter=True)

            @equations
            def _(m):
                # Ohm's law
                m.p.v - m.n.v == m.R * m.p.i
                # Current conservation
                m.p.i + m.n.i == 0

        resistor = Resistor()
        flat = resistor.flatten()

        # Should have parameter R
        assert "R" in flat.param_names

        # Should have connector variables
        assert "p.v" in flat.algebraic_names or "p.v" in flat.state_names
        assert "p.i" in flat.algebraic_names or "p.i" in flat.state_names

    def test_series_connection(self) -> None:
        """Test series connection of two resistors."""
        from cyecca.dsl import Real, connect, connector, equations, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = Real(flow=True)

        @model
        class Resistor:
            p = submodel(Pin)
            n = submodel(Pin)
            R = Real(1000.0, parameter=True)

            @equations
            def _(m):
                m.p.v - m.n.v == m.R * m.p.i
                m.p.i + m.n.i == 0

        @model
        class SeriesResistors:
            r1 = submodel(Resistor)
            r2 = submodel(Resistor)

            @equations
            def _(m):
                # Connect r1.n to r2.p (series connection)
                connect(m.r1.n, m.r2.p)

        circuit = SeriesResistors()
        eqs = circuit.get_equations()

        # Should have connection equations from connect()
        # Plus equations from submodels (Ohm's law, current conservation)
        assert len(eqs) >= 2  # At least the connection equations

    def test_flow_repr(self) -> None:
        """Test that flow=True appears in var repr."""
        from cyecca.dsl import Real, var

        v = Real(flow=True)
        assert "flow=True" in repr(v)

    def test_connector_flatten(self) -> None:
        """Test that connector variables are properly flattened."""
        from cyecca.dsl import Real, connector, model, submodel, var

        @connector
        class Pin:
            v = Real()
            i = Real(flow=True)

        @model
        class Component:
            pin = submodel(Pin)

        comp = Component()
        flat = comp.flatten()

        # Connector variables should appear with dotted names
        all_vars = flat.state_names + flat.algebraic_names + flat.param_names + flat.input_names + flat.output_names
        assert "pin.v" in all_vars
        assert "pin.i" in all_vars
