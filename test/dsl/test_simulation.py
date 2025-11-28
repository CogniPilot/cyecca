"""
Tests for cyecca.dsl.simulation module.

Covers: Simulator abstract class
"""

import pytest


class TestSimulatorAbstract:
    """Test Simulator abstract class interface."""

    def test_simulator_is_abstract(self) -> None:
        """Verify Simulator is an abstract base class."""
        from abc import ABC

        from cyecca.dsl.simulation import Simulator

        # Check it's an ABC
        assert issubclass(Simulator, ABC)

    def test_cannot_instantiate_simulator(self) -> None:
        """Verify Simulator cannot be instantiated directly."""
        from cyecca.dsl.simulation import Simulator

        with pytest.raises(TypeError):
            Simulator()  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
