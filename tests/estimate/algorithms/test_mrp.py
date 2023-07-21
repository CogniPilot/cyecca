from tests.common import *

from cyecca.lie.group_so2 import *
from cyecca.estimate.attitude.algorithms.mrp import *


class Test_Mrp(ProfiledTestCase):
    def setUp(self):
        super().setUp()

    @unittest.skip
    def test_derive_eqs(self):
        eqs()
