import casadi as ca
import sympy

__all__ = ["taylor_series_near_zero", "sympy_to_casadi", "SERIES", "casadi_to_sympy"]


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


def casadi_to_sympy(expr, syms=None):
    if syms is None:
        syms = {}

    def binary(expr, f):
        a = casadi_to_sympy(expr.dep(0), syms)
        b = casadi_to_sympy(expr.dep(1), syms)
        return f(a, b)

    def unary(expr, f):
        a = casadi_to_sympy(expr.dep(0), syms)
        return f(a)

    # handle matrix expression
    if expr.numel() > 1:
        m, n = expr.shape
        M = sympy.zeros(m, n)
        for i in range(m):
            for j in range(n):
                M[i, j] = casadi_to_sympy(expr.elements()[i + m * j], syms)
        return M

    # handle scalar expression
    op = expr.op()

    if op == ca.OP_ASSIGN:
        raise NotImplementedError("op", op)
    elif op == ca.OP_ADD:
        return binary(expr, lambda a, b: a + b)
    elif op == ca.OP_SUB:
        return binary(expr, lambda a, b: a - b)
    elif op == ca.OP_MUL:
        return binary(expr, lambda a, b: a * b)
    elif op == ca.OP_DIV:
        return binary(expr, lambda a, b: a / b)
    elif op == ca.OP_NEG:
        return unary(expr, lambda a: -a)
    elif op == ca.OP_EXP:
        return unary(expr, lambda a: sympy.exp(a))
    elif op == ca.OP_LOG:
        return unary(expr, lambda a: sympy.log(a))
    elif op == ca.OP_POW:
        return binary(expr, lambda a, b: a**b)
    elif op == ca.OP_CONSTPOW:
        raise NotImplementedError("op", op)
    elif op == ca.OP_SQRT:
        return unary(expr, lambda a: sympy.sqrt(a))
    elif op == ca.OP_SQ:
        return unary(expr, lambda a: a**2)
    elif op == ca.OP_TWICE:
        return unary(expr, lambda a: 2 * a)
    elif op == ca.OP_SIN:
        return unary(expr, lambda a: sympy.sin(a))
    elif op == ca.OP_COS:
        return unary(expr, lambda a: sympy.cos(a))
    elif op == ca.OP_TAN:
        return unary(expr, lambda a: sympy.tan(a))
    elif op == ca.OP_ASIN:
        return unary(expr, lambda a: sympy.asin(a))
    elif op == ca.OP_ACOS:
        return unary(expr, lambda a: sympy.acos(a))
    elif op == ca.OP_ATAN:
        return unary(expr, lambda a: sympy.atan(a))
    elif op == ca.OP_LT:
        return binary(expr, lambda a, b: a < b)
    elif op == ca.OP_LE:
        return binary(expr, lambda a, b: a <= b)
    elif op == ca.OP_EQ:
        return binary(expr, lambda a, b: a == b)
    elif op == ca.OP_NE:
        return binary(expr, lambda a, b: a != b)
    elif op == ca.OP_NOT:
        return unary(expr, lambda a: sympy.Not(a))
    elif op == ca.OP_AND:
        return binary(expr, lambda a, b: sympy.And(a, b))
    elif op == ca.OP_OR:
        return binary(expr, lambda a, b: sympy.Or(a, b))
    elif op == ca.OP_FLOOR:
        return unary(expr, lambda a: sympy.floor(a))
    elif op == ca.OP_CEIL:
        return unary(expr, lambda a: sympy.ceil(a))
    elif op == ca.OP_FMOD:
        return binary(expr, lambda a, b: sympy.Mod(a, b))
    elif op == ca.OP_FABS:
        return unary(expr, lambda a: sympy.Abs(a))
    elif op == ca.OP_SIGN:
        return unary(expr, lambda a: sympy.sign(a))
    elif op == ca.OP_COPYSIGN:
        raise NotImplementedError("")
    elif op == ca.OP_IF_ELSE_ZERO:
        return binary(expr, lambda cond, val: sympy.Piecewise((val, cond), (0, True)))
    elif op == ca.OP_ERF:
        return unary(expr, lambda a: sympy.erf(a))
    elif op == ca.OP_FMIN:
        return binary(expr, lambda a, b: sympy.Piecewise((a, a < b), (b, True)))
    elif op == ca.OP_FMAX:
        return binary(expr, lambda a, b: sympy.Piecewise((a, a > b), (b, True)))
    elif op == ca.OP_SINH:
        return unary(expr, lambda a: sympy.sinh(a))
    elif op == ca.OP_COSH:
        return unary(expr, lambda a: sympy.cosh(a))
    elif op == ca.OP_TANH:
        return unary(expr, lambda a: sympy.tanh(a))
    elif op == ca.OP_ASINH:
        return unary(expr, lambda a: sympy.asinh(a))
    elif op == ca.OP_COSH:
        return unary(expr, lambda a: sympy.acosh(a))
    elif op == ca.OP_ATANH:
        return unary(expr, lambda a: sympy.atanh(a))
    elif op == ca.OP_ATAN2:
        return binary(expr, lambda a, b: sympy.atan2(a, b))
    elif op == ca.OP_CONST:
        f_num = float(expr)
        int_num = int(expr)
        if f_num - int_num == 0:
            return int_num
        else:
            return f_num
    elif op == ca.OP_INPUT:
        raise NotImplementedError("")
    elif op == ca.OP_OUTPUT:
        raise NotImplementedError("")
    elif op == ca.OP_PARAMETER:
        if expr not in syms:
            syms[expr] = sympy.symbols(str(expr))
        return syms[expr]
    elif op == ca.OP_CALL:
        raise NotImplementedError("")
    elif op == ca.OP_FIND:
        raise NotImplementedError("")
    elif op == ca.OP_LOW:
        raise NotImplementedError("")
    elif op == ca.OP_MAP:
        raise NotImplementedError("")
    elif op == ca.OP_MTIMES:
        raise NotImplementedError("")
    elif op == ca.OP_SOLVE:
        raise NotImplementedError("")
    elif op == ca.OP_TRANSPOSE:
        raise NotImplementedError("")
    elif op == ca.OP_DETERMINANT:
        raise NotImplementedError("")
    elif op == ca.OP_INVERSE:
        raise NotImplementedError("")
    elif op == ca.OP_INVERSE:
        raise NotImplementedError("")
    elif op == ca.OP_DOT:
        raise NotImplementedError("")
    elif op == ca.OP_BILIN:
        raise NotImplementedError("")
    elif op == ca.OP_RANK1:
        raise NotImplementedError("")
    elif op == ca.OP_HORZCAT:
        raise NotImplementedError("")
    elif op == ca.OP_VERTCAT:
        raise NotImplementedError("")
    elif op == ca.OP_DIAGCAT:
        raise NotImplementedError("")
    elif op == ca.OP_HORZSPLIT:
        raise NotImplementedError("")
    elif op == ca.OP_VERTSPLIT:
        raise NotImplementedError("")
    elif op == ca.OP_DIAGSPLIT:
        raise NotImplementedError("")
    elif op == ca.OP_RESHAPE:
        raise NotImplementedError("")
    elif op == ca.OP_SUBREF:
        raise NotImplementedError("")
    elif op == ca.OP_SUBASSIGN:
        raise NotImplementedError("")
    elif op == ca.OP_GETNONZEROS:
        raise NotImplementedError("")
    elif op == ca.OP_GETNONZEROS_PARAM:
        raise NotImplementedError("")
    elif op == ca.OP_ADDNONZEROS:
        raise NotImplementedError("")
    elif op == ca.OP_ADDNONZEROS_PARAM:
        raise NotImplementedError("")
    elif op == ca.OP_SETNONZEROS:
        raise NotImplementedError("")
    elif op == ca.OP_SETNONZEROS_PARAM:
        raise NotImplementedError("")
    elif op == ca.OP_PROJECT:
        raise NotImplementedError("")
    elif op == ca.OP_ASSERTION:
        raise NotImplementedError("")
    elif op == ca.OP_MONITOR:
        raise NotImplementedError("")
    elif op == ca.OP_NORM2:
        raise NotImplementedError("")
    elif op == ca.OP_NORM1:
        raise NotImplementedError("")
    elif op == ca.OP_NORMINF:
        raise NotImplementedError("")
    elif op == ca.OP_NORMF:
        raise NotImplementedError("")
    elif op == ca.OP_MMIN:
        raise NotImplementedError("")
    elif op == ca.OP_MMAX:
        raise NotImplementedError("")
    elif op == ca.OP_HORZREPSUM:
        raise NotImplementedError("")
    elif op == ca.OP_ERFINV:
        raise NotImplementedError("")
    elif op == ca.OP_PRINTME:
        raise NotImplementedError("")
    elif op == ca.OP_LIFT:
        raise NotImplementedError("")
    elif op == ca.OP_EINSTEIN:
        raise NotImplementedError("")
    elif op == ca.OP_BSPLINE:
        raise NotImplementedError("")
    elif op == ca.OP_CONVEXIFY:
        raise NotImplementedError("")
    elif op == ca.OP_SPARSITY_CAST:
        raise NotImplementedError("")
    elif op == ca.OP_LOG1P:
        raise NotImplementedError("")
    elif op == ca.OP_EXPM1:
        raise NotImplementedError("")
    elif op == ca.OP_HYPOT:
        raise NotImplementedError("")
    elif op == ca.OP_LOGSUMEXP:
        raise NotImplementedError("")
    elif op == ca.OP_REMAINDER:
        raise NotImplementedError("")
    else:
        raise NotImplementedError("op", op)


def derive_series():
    x = sympy.symbols("x")
    cos = sympy.cos
    sin = sympy.sin
    return {
        "sin(x)/x": taylor_series_near_zero(x, sin(x) / x),
        "(1 - cos(x))/x": taylor_series_near_zero(x, (1 - cos(x)) / x),
        "(1 - cos(x))/x^2": taylor_series_near_zero(x, (1 - cos(x)) / x**2),
        "(x - sin(x))/x^3": taylor_series_near_zero(x, (x - sin(x)) / x**3),
        "(1 - x*sin(x)/(2*(1 - cos(x))))/x^2": taylor_series_near_zero(
            x, (1 - x * sin(x) / (2 * (1 - cos(x)))) / x**2
        ),
        "(-x^2/2 - cos(x) + 1)/x^2": taylor_series_near_zero(
            x, (-(x**2) / 2 - cos(x) + 1) / x**2
        ),
        "(x^2/2 + cos(x) - 1)/x^4": taylor_series_near_zero(
            x, (x**2 / 2 + cos(x) - 1) / x**4
        ),
        "1/x^2": taylor_series_near_zero(x, 1 / x**2),
    }


SERIES = derive_series()
