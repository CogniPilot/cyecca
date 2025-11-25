# Documentation Guide for Cyecca

## Overview

Cyecca uses **Sphinx** with **NumPy-style docstrings** for documentation, following best practices from NumPy, SciPy, and JAX.

## Quick Reference

### Running Tests

```bash
# Test doctests in source code (recommended)
pytest --doctest-modules cyecca/

# Test specific module
pytest --doctest-modules cyecca/lie/group_se3.py -v

# Test regular unit tests
pytest tests/

# Build and test Sphinx docs
cd docs
make doctest
make html
```

### Adding Examples to Code

Add examples directly in docstrings using NumPy format:

```python
def my_function(x, y):
    """
    Short one-line description.
    
    Longer description if needed.
    
    Parameters
    ----------
    x : type
        Description of x
    y : type
        Description of y
        
    Returns
    -------
    type
        Description of return value
        
    Examples
    --------
    >>> import casadi as ca
    >>> from cyecca.dynamics import my_function
    >>> result = my_function(1, 2)
    >>> result
    3
    """
    return x + y
```

### Building Documentation

```bash
# Install dependencies (one time)
poetry install

# Build HTML docs
cd docs
make html
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux

# Clean build
make clean
make html
```

### Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── Makefile             # Build commands
├── api/                 # API reference (auto-generated)
│   ├── index.rst
│   ├── lie.rst
│   └── model.rst
├── user_guide/          # Tutorials and guides
│   └── index.rst
└── examples/            # Example gallery
    └── index.rst
```

## Doctest Best Practices

### What to Include

✅ **DO:**
- Simple, runnable examples
- Common use cases
- Edge cases that might confuse users
- Examples that demonstrate the API

❌ **DON'T:**
- Complex multi-step workflows (put in user_guide/)
- Examples requiring external data
- Slow computations (use `# doctest: +SKIP`)
- Platform-specific code

### Doctest Directives

```python
# Skip a test
>>> slow_computation()  # doctest: +SKIP

# Allow ellipsis for long output
>>> long_array
array([1, 2, ..., 100])

# Normalize whitespace
>>> messy_output()  # doctest: +NORMALIZE_WHITESPACE
```

### Global Setup

Common imports are configured in `docs/conf.py`:

```python
doctest_global_setup = """
import casadi as ca
import cyecca.lie as lie
from cyecca.dynamics import *
import matplotlib
matplotlib.use('Agg')
"""
```

## CI Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Test doctests
  run: poetry run pytest --doctest-modules cyecca/

- name: Build docs
  run: |
    cd docs
    poetry run make html
```

## References

- [NumPy Documentation Guide](https://numpydoc.readthedocs.io/)
- [Sphinx NumPy Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [JAX Documentation](https://jax.readthedocs.io/) (good example)
