import multiprocessing as mp

import numpy as np

from cyecca.sim import replay
from cyecca.sim import uros
from cyecca.estimate.attitude import algorithms
from cyecca.estimate.attitude.estimator import AttitudeEstimator
from cyecca.estimate.attitude.simulator import Simulator

default_params = {
    "t0": 0,
    "tf": 1,
    "n_monte_carlo": 1,
    "replay_log_file": None,
    "name": "default",
    "initialize": True,
    "estimators": [],
    "x0": [0, 0, 0, 0, 0, 0],
    "params": {},
}

eqs = algorithms.eqs()


def init_params(params):
    p = dict(default_params)
    for k, v in params.items():
        if k not in p.keys():
            raise KeyError(k)
        p[k] = v
    return p


def launch_sim(params):
    p = init_params(params)
    core = uros.Core()
    Simulator(core, eqs, p["x0"])
    for name in p["estimators"]:
        AttitudeEstimator(core, name, eqs[name], p["initialize"])
    logger = uros.Logger(core)
    core.init_params()
    for k, v in p["params"].items():
        core.set_param(k, v)
    core.run(until=p["tf"])
    print(p["name"], "done")
    return logger.get_log_as_array()


def launch_monte_carlo_sim(params):
    p = init_params(params)
    if p["n_monte_carlo"] == 1:
        d = dict(p)
        d.pop("n_monte_carlo")
        data = [launch_sim(d)]
    else:
        new_params = []
        for i in range(p["n_monte_carlo"]):
            d = dict(p)
            d.pop("n_monte_carlo")
            d["name"] = i
            new_params.append(d)
        with mp.Pool(mp.cpu_count()) as pool:
            data = np.array(pool.map(launch_sim, new_params))
    return data


def launch_replay(params):
    p = init_params(params)
    core = uros.Core()
    replay.ULogReplay(core, p["replay_log_file"])
    for name in p["estimators"]:
        AttitudeEstimator(core, name, eqs[name], p["initialize"])
    logger = uros.Logger(core)
    core.init_params()
    for k, v in p["params"].items():
        core.set_param(k, v)
    core.run(until=p["tf"])
    print(p["name"], "done")
    return logger.get_log_as_array()
