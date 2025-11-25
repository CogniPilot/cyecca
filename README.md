# cyecca

[![Build](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml/badge.svg)](https://github.com/CogniPilot/cyecca/actions/workflows/ci.yml)

Estimation and Control with Computer Algebra systems

## Installation

### Core Installation (Lightweight Lie Group & Control Library)

For just the core Lie group and control functionality without heavy visualization dependencies:

```bash
git clone git@github.com:CogniPilot/cyecca.git
cd cyecca
poetry install
poetry shell
```

### Full Installation (With Old Estimator Code)

If you need the old attitude estimator code (5+ year old PhD research) with ULog replay and plotting:

```bash
git clone git@github.com:CogniPilot/cyecca.git
cd cyecca
poetry install --extras estimator
poetry shell
```

Or with system dependencies:
```bash
sudo apt install graphviz
poetry install --extras estimator
```

## Running Tests

The core Lie group and control tests run without any heavy dependencies:

```bash
poetry run pytest
```

To run the old estimator tests (requires `poetry install --extras estimator`):

```bash
poetry run pytest tests/estimate/
```
