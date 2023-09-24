import unittest

from beartype.roar import BeartypeCallHintParamViolation

from pathlib import Path
import cProfile
from pstats import Stats

import casadi as ca

EPS = 1e-9

from cyecca.symbolic import casadi_to_sympy


def SX_close(e1: (ca.SX, ca.DM), e2: (ca.SX, ca.DM)):
    close = ca.norm_2(e1 - e2) < EPS
    if not close:
        print(ca.DM(e1), ca.DM(e2))
    return close


class ProfiledTestCase(unittest.TestCase):
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
