# IR Standalone Test Plan

This document outlines the incremental steps required to decouple IR coverage from DSL tests.

## Coverage Baseline (2025-11-29)
- Command: `pytest --cov=cyecca.dsl --cov=cyecca.ir --cov-report=term-missing src/cyecca`
- DSL coverage gaps: `dsl/instance.py` (~75%), `context.py`, `decorators.py`, `equations.py` due to broad functional scope.
- IR coverage gaps: `ir/expr.py` (operator repr paths), `ir/equation.py` (dataclass helpers), `ir/causality.py` (implicit block handling), `ir/flat_model.py` (array bookkeeping), `ir/simulation.py` (Simulator/SimulationResult helpers).

## New IR Test Modules
1. `test_expr.py`
   - Cover arithmetic/boolean/comparison operators, `indexed_name`, and repr strings for `ExprKind` nodes that never appear in DSL tests (INDEX, ARRAY_LITERAL, PRE/EDGE/CHANGE, IF_THEN_ELSE, REINIT, INITIAL/TERMINAL/SAMPLE).
   - Verify `_to_expr_basic` conversions and `iter_indices` helpers if exposed.

2. `test_equation.py`
   - Exercise `IREquation`, `IRReinit`, `IRWhenClause`, `IRInitialEquation`, `IRAssignment` reprs and equality semantics directly.
   - Validate `IRWhenClause` prefixing via manual Expr trees.

3. `test_simulation.py`
   - Instantiate `SimulationResult` with synthetic data to ensure callable interface, slicing, interpolation, and metadata pathways; build a dummy `Simulator` subclass to exercise abstract methods.

4. `test_causality.py`
   - Construct minimal `IRModel` instances (without DSL) to feed into `analyze_causality` and `SortedSystem`, covering implicit vs explicit blocks, algebraic loops, and exception pathways currently hit only by DSL tests.

5. `test_flat_model.py`
   - Validate helper methods on `FlatModel` (e.g., `state_vector`, `parameter_defaults`) using IR-only fixtures.

6. `test_algorithms.py`
   - Cover algorithm/assignment flows by instantiating `IRModel` with manual `IRAssignment` lists, ensuring conversions do not rely on DSL decorators.

## DSL/IR Boundary Strategy
- Keep DSL tests focused on decorator plumbing, context managers, and syntactic sugar (do not rely on full backend compilation).
- Move behavior validations (state classification, causality order, expression formatting) into the IR test suite.
- Introduce lightweight fakes/mocks for `ModelMetadata` when DSL tests must assert IR output structures without invoking the real IR logic.

## Refactor Support
- Split `dsl/instance.py` into smaller modules:
  1. `dsl/symbol_table.py`: symbol creation, submodel flattening.
  2. `dsl/flatten.py`: equation/algorithm gathering, classification, and FlatModel construction.
  3. `dsl/model_instance.py`: user-facing `ModelInstance` delegating to the above utilities.
- This separation makes it easier to unit-test DSL syntactic sugar independent of IR internals.

## Execution Order
1. Implement `test_expr.py`, `test_equation.py`, `test_simulation.py` to immediately boost IR coverage.
2. Refactor `dsl/instance.py` into helper modules and update imports.
3. Adjust DSL tests to mock/stub the new helpers, verifying only the syntactic interfaces.
4. Expand IR causality/flat-model tests to cover the remaining uncovered lines.
