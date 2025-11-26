import cProfile
import unittest
from pathlib import Path
from pstats import Stats

import casadi as ca
from beartype import beartype
from beartype.typing import Union

EPS = 1e-9

import numpy as np


@beartype
def is_finite(e: ca.SX) -> bool:
    """Check if all elements in a CasADi expression are finite."""
    return bool(np.all(np.isfinite(ca.DM(e))))


@beartype
def SX_close(e1: Union[ca.SX, ca.DM], e2: Union[ca.SX, ca.DM]):
    """Check if two CasADi expressions are close within EPS tolerance."""
    close = ca.mmax(e1 - e2) < EPS
    if not close:
        print(ca.DM(e1), ca.DM(e2))
    return close


@beartype
class ProfiledTestCase(unittest.TestCase):
    """Base test case with profiling support."""

    def setUp(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self) -> None:
        p = Stats(self.pr)
        p.strip_dirs()
        p.sort_stats("cumtime")
        profile_dir = Path(".profile")
        profile_dir.mkdir(exist_ok=True)
        p.dump_stats(profile_dir / self.id())
