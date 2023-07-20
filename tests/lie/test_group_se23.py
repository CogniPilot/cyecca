from ..common import *

from cyecca.lie.group_se23 import *


class Test_LieGroupSE23Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()
        self.v1 = ca.DM([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])

    def test_ctor(self):
        SE23Mrp.elem(param=self.v1)
