"""
IPython magic commands for Cyecca.

Usage in a Jupyter notebook:
    # Load the extension
    %load_ext cyecca

    # Pre-declare the model to satisfy type checkers
    from cyecca.ir import Model
    ir_model: Model | None = None

    # Use the magic to compile Modelica code
    %%modelica_rumoca ir_model
    model MyModel
        Real x(start=1);
    equation
        der(x) = -x;
    end MyModel;

    # ir_model is now a cyecca.ir.Model
    # Create a backend from it
    from cyecca.backends.casadi import CasadiBackend
    backend = CasadiBackend(ir_model).compile()
    t, sol = backend.simulate(10.0)

Options:
    %%modelica_rumoca VAR_NAME [-m MODEL] [-v] [-l PATH] [--no-modelica-path]

    -m, --model MODEL       Specify which model to compile (useful with multiple classes)
    -v, --verbose           Display the compiled IR model
    -l, --library PATH      Add a library path (can be used multiple times)
    --no-modelica-path      Don't search MODELICAPATH environment variable

For help in Jupyter: %%modelica_rumoca?
"""

from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring


@magics_class
class CyeccaMagics(Magics):
    """IPython magics for Cyecca."""

    @magic_arguments()
    @argument("var_name", type=str, help="Variable name to store the compiled Model")
    @argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model name to compile (auto-detected if not provided)",
    )
    @argument("-v", "--verbose", action="store_true", help="Display the compiled IR model")
    @argument(
        "-l",
        "--library",
        action="append",
        default=None,
        help="Library path to include (can be used multiple times)",
    )
    @argument(
        "--no-modelica-path",
        action="store_true",
        help="Do not search MODELICAPATH environment variable",
    )
    @argument(
        "-t",
        "--threads",
        type=int,
        default=None,
        help="Number of threads for parallel parsing (default: num_cpus - 1)",
    )
    @cell_magic
    def modelica_rumoca(self, line, cell):
        """
        Compile Modelica code to a cyecca.ir.Model.

        Usage:
            %%modelica_rumoca VAR_NAME [-m MODEL] [-v] [-l PATH] [--no-modelica-path]

        Arguments:
            VAR_NAME            Variable name to store the compiled Model
            -m, --model         Model name to compile (auto-detected if omitted)
            -v, --verbose       Display the compiled IR model
            -l, --library       Library path (can be repeated, e.g., -l /path/to/MSL)
            --no-modelica-path  Don't search MODELICAPATH env var for libraries

        Examples:
            # Auto-detect model name (quiet by default):
            %%modelica_rumoca ir_model
            model MyModel
                Real x(start=1);
            equation
                der(x) = -x;
            end MyModel;

            # With MSL library:
            %%modelica_rumoca ir_model -l /path/to/MSL
            model MyModel
                import Modelica.Blocks.Continuous.PID;
                PID pid(k=2.0);
                Real x;
            equation
                der(x) = -x + pid.y;
                pid.u = x;
            end MyModel;

            # Explicit model name (useful with multiple classes):
            %%modelica_rumoca ir_model -m SecondModel
            model FirstModel
                Real a;
            end FirstModel;

            model SecondModel
                Real b;
            equation
                der(b) = -b;
            end SecondModel;

            # Verbose mode (display compiled IR):
            %%modelica_rumoca ir_model -v
            model MyModel
                Real x;
            equation
                der(x) = -x;
            end MyModel;

        After compilation, create a backend:
            from cyecca.backends.casadi import CasadiBackend
            backend = CasadiBackend(ir_model).compile()

        For help: %%modelica_rumoca?
        """
        args = parse_argstring(self.modelica_rumoca, line)
        var_name = args.var_name
        model_name = args.model
        verbose = args.verbose
        library_paths = args.library
        use_modelica_path = not args.no_modelica_path
        threads = args.threads

        # Compile the Modelica code to IR Model
        from cyecca.io import compile_modelica

        ir_model = compile_modelica(
            cell,
            model_name,
            library_paths=library_paths,
            use_modelica_path=use_modelica_path,
            threads=threads,
        )

        # Store in user namespace
        self.shell.user_ns[var_name] = ir_model

        if verbose:
            return ir_model


def load_ipython_extension(ipython):
    """Load the Cyecca magic extension."""
    ipython.register_magics(CyeccaMagics)


def unload_ipython_extension(ipython):
    """Unload the Cyecca magic extension."""
    pass
