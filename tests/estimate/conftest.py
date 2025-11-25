'''
Pytest configuration for estimator tests.

These tests are for old estimator code from PhD research (5+ years old)
and require heavy dependencies (pyulog, matplotlib, graphviz).

To run these tests:
1. Install optional dependencies: pip install 'cyecca[estimator]'
2. Run with: pytest tests/estimate/
'''

import pytest


def pytest_configure(config):
    '''Configure pytest to ignore estimator tests.'''
    config.addinivalue_line(
        'markers',
        'estimator: marks tests as requiring estimator dependencies (deselect with '-m \'not estimator\'')',
    )

