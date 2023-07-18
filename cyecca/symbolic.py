import casadi as ca
import sympy


def taylor_series_near_zero(x, f, order=6, eps=1e-7, verbose=False):
    """
    Takes a sympy function and near zero approximates it by a taylor
    series. The resulting function is converted to a casadi function.

    @x: sympy independent variable
    @f: sympy function
    @eps: tolerance for using series
    @verbose: show functions
    @return: casadi.Function
    """
    symbols = {"x": ca.SX.sym("x")}
    f_series = f.series(x, 0, order).removeO()
    f_series, _ = sympy_to_casadi(f=f_series, symbols=symbols)
    if verbose:
        print("f_series: ", f_series, "\nf:", f)
    f, _ = sympy_to_casadi(f, symbols=symbols)
    f = ca.Function(
        "f", [symbols["x"]], [ca.if_else(ca.fabs(symbols["x"]) < eps, f_series, f)]
    )
    return f


def sympy_to_casadi(f, f_dict=None, symbols=None, cse=False, verbose=False):
    if symbols is None:
        symbols = {}
    return (
        _sympy_parser(f=f, f_dict=f_dict, symbols=symbols, cse=cse, verbose=verbose),
        symbols,
    )


def _sympy_parser(f, f_dict=None, symbols=None, depth=0, cse=False, verbose=False):
    if f_dict is None:
        f_dict = {}
    prs = lambda f: _sympy_parser(
        f=f, f_dict=f_dict, symbols=symbols, depth=depth + 1, cse=False, verbose=verbose
    )
    f_type = type(f)
    dict_keys = list(f_dict.keys())
    if verbose:
        print("-" * depth, f, "type", f_type)
    if cse:
        cse_defs, cse_exprs = sympy.cse(f)
        assert len(cse_exprs) == 1
        ca_cse_defs = {}
        for symbol, subexpr in reversed(cse_defs):
            ca_cse_defs[prs(symbol)] = prs(subexpr)
        f_ca = prs(cse_exprs[0])
        for k, v in ca_cse_defs.items():
            f_ca = ca.substitute(f_ca, k, v)
        for symbol, subexpr in reversed(cse_defs):
            if str(symbol) in symbols:
                symbols.pop(str(symbol))
        return f_ca
    if f_type == sympy.core.add.Add:
        s = 0
        for arg in f.args:
            s += prs(arg)
        return s
    elif f_type == sympy.core.mul.Mul:
        prod = 1
        for arg in f.args:
            prod *= prs(arg)
        return prod
    elif f_type == sympy.core.numbers.Integer:
        return int(f)
    elif f_type == sympy.core.power.Pow:
        base, power = f.args
        base_ca = prs(base)
        if type(power) == sympy.core.numbers.Half:
            return ca.sqrt(base_ca)
        else:
            return base_ca ** prs(power)
    elif f_type == sympy.core.symbol.Symbol:
        if str(f) not in symbols:
            symbols[str(f)] = ca.SX.sym(str(f))
        return symbols[str(f)]
    elif f_type == sympy.matrices.dense.MutableDenseMatrix:
        mat = ca.SX(f.shape[0], f.shape[1])
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                mat[i, j] = prs(f[i, j])
        return mat
    elif f_type == int:
        return f
    elif f_type == sympy.core.numbers.Rational:
        return prs(f.numerator) / prs(f.denominator)
    elif f_type == sympy.core.numbers.Float:  # Convert Float to int
        return int(f)
    elif f_type == sympy.core.numbers.One:
        return 1
    elif f_type == sympy.core.numbers.Zero:
        return 0
    elif f_type == sympy.core.numbers.NegativeOne:
        return -1
    elif f_type == sympy.core.numbers.Half:
        return 0.5
    elif str(f_type) == "sin":
        return ca.sin(prs(f.args[0]))
    elif str(f_type) == "cos":
        return ca.cos(prs(f.args[0]))
    elif str(f_type) in dict_keys:
        for i in range(len(dict_keys)):
            return f_dict[dict_keys[i]](prs(f.args[0]))
    else:
        print("unhandled type", type(f), f)
