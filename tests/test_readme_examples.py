"""
README Examples Testing - DEPRECATED

This test is deprecated in favor of Sphinx doctests in source code.
Examples should now be added as docstrings using NumPy-style format.

To run doctests:
    pytest --doctest-modules cyecca/
    
Or in Sphinx docs:
    cd docs && make doctest
"""

import pytest
from pathlib import Path


@pytest.mark.skip(
    reason="Deprecated: Use doctests in source code (pytest --doctest-modules)"
)
def test_readme_examples_deprecated():
    """Old README testing approach is deprecated."""
    pass
