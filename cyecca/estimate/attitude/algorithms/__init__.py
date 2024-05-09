import casadi as ca
import os

# from . import mekf, mrp, quat, sim

import cyecca.estimate.attitude.algorithms.mrp as mrp
import cyecca.estimate.attitude.algorithms.sim as sim


def eqs(**kwargs):
    return {
        # "mekf": mekf.eqs(**kwargs),
        "mrp": mrp.eqs(**kwargs),
        # "quat": quat.eqs(**kwargs),
        "sim": sim.eqs(**kwargs),
    }


def generate_code(eqs, dest_dir, **kwargs):
    p = {"main": False, "mex": False, "with_header": True, "with_mem": True}
    for k, v in kwargs.items():
        assert k in p.keys()
        p[k] = v

    # generate code
    # Code Generation
    for name, eqs in eqs.items():
        filename = "casadi_{:s}.c".format(name)
        gen = ca.CodeGenerator(filename, p)
        for f_name in eqs:
            gen.add(eqs[f_name])

        os.makedirs(dest_dir, exist_ok=True)
        gen.generate(dest_dir + os.path.sep)
