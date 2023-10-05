import casadi as ca
import pathlib


def generate_code(eqs: dict, dest_dir: str, **kwargs):
    dest_dir = pathlib.Path(dest_dir)
    p = {
        "verbose": True,
        "mex": False,
        "cpp": False,
        "main": False,
        "with_header": True,
        "with_mem": False,
        "with_export": False,
        "with_import": False,
        "include_math": True,
        "avoid_stack": True,
    }
    for k, v in kwargs.items():
        assert k in p.keys()
        p[k] = v

    for name, eq in eqs.items():
        filename = "{:s}.c".format(name)
        gen = ca.CodeGenerator(filename, p)
        for f_name in eq:
            gen.add(eq[f_name])

        dest_dir = pathlib.Path(dest_dir)
        dest_dir.mkdir(exist_ok=True)

        gen.generate(str(dest_dir) + pathlib.os.sep)
