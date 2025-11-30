"""
Example: Mass-Spring-Damper system using the IR directly.

This demonstrates what Rumoca would generate when compiling a Modelica model.
"""

from cyecca.ir import Model, Variable, Equation, Expr, VariableType, der


def create_mass_spring_damper() -> Model:
    """
    Create a mass-spring-damper model programmatically.

    This is similar to the Modelica model:

    model MassSpringDamper
      Real x(start=1.0) "position";
      Real v(start=0.0) "velocity";
      input Real F "force";
      parameter Real m = 1.0 "mass";
      parameter Real c = 0.1 "damping";
      parameter Real k = 1.0 "spring constant";
    equation
      der(x) = v;
      der(v) = (F - c*v - k*x) / m;
    end MassSpringDamper;
    """
    model = Model(
        name="MassSpringDamper",
        description="Simple mass-spring-damper system",
        version="1.0",
    )

    # Define variables
    model.add_variable(
        Variable(
            name="x",
            var_type=VariableType.STATE,
            start=1.0,
            description="position",
            unit="m",
        )
    )

    model.add_variable(
        Variable(
            name="v",
            var_type=VariableType.STATE,
            start=0.0,
            description="velocity",
            unit="m/s",
        )
    )

    model.add_variable(
        Variable(
            name="F",
            var_type=VariableType.INPUT,
            description="external force",
            unit="N",
        )
    )

    model.add_variable(
        Variable(
            name="m",
            var_type=VariableType.PARAMETER,
            value=1.0,
            description="mass",
            unit="kg",
        )
    )

    model.add_variable(
        Variable(
            name="c",
            var_type=VariableType.PARAMETER,
            value=0.1,
            description="damping coefficient",
            unit="N.s/m",
        )
    )

    model.add_variable(
        Variable(
            name="k",
            var_type=VariableType.PARAMETER,
            value=1.0,
            description="spring constant",
            unit="N/m",
        )
    )

    # Define equations using expression builders
    x = Expr.var_ref("x")
    v = Expr.var_ref("v")
    F = Expr.var_ref("F")
    m = Expr.var_ref("m")
    c = Expr.var_ref("c")
    k = Expr.var_ref("k")

    # Equation 1: der(x) = v
    model.add_equation(Equation.simple(der(x), v))

    # Equation 2: der(v) = (F - c*v - k*x) / m
    damping_force = Expr.mul(c, v)
    spring_force = Expr.mul(k, x)
    net_force = Expr.sub(F, Expr.add(damping_force, spring_force))
    acceleration = Expr.div(net_force, m)

    model.add_equation(Equation.simple(der(v), acceleration))

    return model


if __name__ == "__main__":
    # Create the model
    model = create_mass_spring_damper()

    # Print model information
    print(model)
    print()

    # Validate the model
    errors = model.validate()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Model is valid!")

    print()
    print("Variables:")
    for var in model.variables:
        print(f"  {var.name:8s} ({var.var_type.name:14s}): {var.description}")

    print()
    print("Equations:")
    for i, eq in enumerate(model.equations, 1):
        print(f"  {i}. {eq}")
