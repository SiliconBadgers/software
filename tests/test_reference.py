import unittest
from reference import step

class MacReferenceTest(unittest.TestCase):
    def test_signed_extremes(self):
        self.assertEqual(step(0,-128,127),-16256)
        self.assertEqual(step(0,-128,-128),16384)
    def test_control_priority(self):
        self.assertEqual(step(19,3,7,enable=False),19)
        self.assertEqual(step(19,3,7,enable=True,clear=True),0)
    def test_wrap_boundaries(self):
        self.assertEqual(step(2147483647,1,1),-2147483648)
        self.assertEqual(step(-2147483648,-1,1),2147483647)
    def test_input_bounds(self):
        with self.assertRaises(ValueError): step(0,128,1)
