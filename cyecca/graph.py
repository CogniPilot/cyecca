from IPython.display import Image, display
import os
from pathlib import Path
import tempfile
import casadi as ca
from casadi import print_operator as print_operator_orig
import casadi.tools.graph as mod_graph

try:
    import pydot
    PYDOT_AVAILABLE = True
except ImportError:
    PYDOT_AVAILABLE = False
    pydot = None


def draw_casadi(
    expr: ca.SX or ca.MX, filename: str = None, width: int = 1920, direction: str = "LR"
) -> pydot.Graph:
    """
    Draw a CasADi expression graph.
    
    Requires pydot to be installed: pip install pydot
    or install cyecca with visualization extras: pip install cyecca[visualization]
    """
    if not PYDOT_AVAILABLE:
        raise ImportError(
            "pydot is required for graph visualization. "
            "Install it with: pip install pydot "
            "or: pip install cyecca[visualization]"
        )
    
    curdir = Path(os.path.abspath(os.getcwd()))
    # dot draw creates a source.dot file, lets move to the tmp directory
    os.chdir(tempfile.gettempdir())
    graph = mod_graph.dotgraph(expr, direction=direction)  # pydot.Dot
    if filename is None:
        png = graph.create_png()
        os.chdir(curdir)
        return Image(png, width=width)
    else:
        os.chdir(curdir)
        return graph.write_png(filename)
