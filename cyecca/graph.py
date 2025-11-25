from IPython.display import Image, display
import os
from pathlib import Path
import tempfile
import casadi as ca
from casadi import print_operator as print_operator_orig
import casadi.tools.graph as mod_graph
import pydot


def draw_casadi(
    expr: ca.SX or ca.MX, filename: str = None, width: int = 1920, direction: str = "LR"
) -> pydot.Graph:
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
