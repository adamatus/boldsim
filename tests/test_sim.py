import unittest
import numpy as np

from fmrisim import sim

class TestStimfunction(unittest.TestCase):
    """Unit tests for sim.stimfunction"""

    def setUp(self):
        self.total_time = 8
        self.onsets = [0,4]
        self.duration = 2
        self.acc = 1

    def test_with_acc_of_one(self):
        """Test stimfunction with accuracy=1"""
        s = sim.stimfunction(self.total_time, self.onsets,
                             self.duration, 1)
        self.assertTrue(np.all(s == [1, 1, 0, 0, 1, 1, 0, 0]))

    def test_with_acc_of_half(self):
        """Test stimfunction with accuracy=.5"""
        s = sim.stimfunction(self.total_time, self.onsets,
                             self.duration, .5)
        self.assertTrue(np.all(s == [1, 1, 1, 1, 0, 0, 0, 0,
                                     1, 1, 1, 1, 0, 0, 0, 0]))

    def test_with_acc_of_two(self):
        """Test stimfunction with accuracy=2"""
        s = sim.stimfunction(self.total_time, self.onsets,
                             self.duration, 2)
        self.assertTrue(np.all(s == [1, 0, 1, 0]))

    def test_onset_exceptions(self):
        """Test stimfunction onsets exception handling"""
        def f(x):
            s = sim.stimfunction(10, x,
                                 self.duration, 1)
        self.assertRaises(Exception, f, 12)

        # We shouldn't raise an error if onset is < total_time
        try:
            f(2)
        except:
            self.fail('Unexpected exception thrown')

        # We need to have matching length lists for durations and onsets
        def f2():
            s = sim.stimfunction(10,
                                 onsets=[1, 2, 3],
                                 durations=[1, 2], accuracy=1)
        self.assertRaises(Exception, f)
