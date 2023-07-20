import unittest

from beartype.roar import BeartypeCallHintParamViolation

from pathlib import Path
import cProfile
from pstats import Stats

import casadi as ca

EPS = 1e-9


def SX_close(e1: (ca.SX, ca.DM), e2: (ca.SX, ca.DM)):
    return ca.norm_2(e1 - e2) < EPS


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
