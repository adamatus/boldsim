import unittest
import numpy as np

from fmrisim import sim

class TestStimfunction(unittest.TestCase):
    """Unit tests for sim.stimfunction"""

    def setUp(self):
        """Setup defaults for stimfunction tests"""
        self.total_time = 8
        self.onsets = [0,4]
        self.duration = 2
        self.acc = 1

    def test_output_is_correct_length(self):
        """Test stimfunction returns correct length output"""
        d = sim.stimfunction(100, self.onsets, self.duration,
                              accuracy=1)
        self.assertTrue(len(d) == 100)

        d = sim.stimfunction(100, self.onsets, self.duration,
                              accuracy=.1)
        self.assertTrue(len(d) == 100/.1)

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

class TestSpecifyDesign(unittest.TestCase):
    """Unit tests for sim.specifydesign"""

    def setUp(self):
        """Setup defaults for stimfunction tests"""
        self.total_time = 100
        self.onsets = [0,49]
        self.duration = 1
        self.acc = 1

    def test_output_is_correct_length(self):
        """Test specifydesign returns correct length output"""
        d = sim.specifydesign(100, self.onsets, self.duration,
                              accuracy=1, conv='gamma')
        self.assertTrue(len(d) == 100)

        d = sim.specifydesign(100, self.onsets, self.duration,
                              accuracy=.1, conv='gamma')
        self.assertTrue(len(d) == 100/.1)

    def test_with_no_arguments(self):
        """Test specifydesign with no args matches stimfunction"""
        s = sim.stimfunction()
        d = sim.specifydesign()
        self.assertTrue(np.all(s == d))

    def test_with_no_conv(self):
        """Test specifydesign with no convolution matches function"""
        s = sim.stimfunction()
        d = sim.specifydesign(conv='none')
        self.assertTrue(np.all(s == d))

    def test_with_gamma(self):
        """Test specifydesign with gamma convolution"""
        d = sim.specifydesign(self.total_time, self.onsets, self.duration,
                              accuracy=self.acc, conv='gamma')
        g = np.round(sim.gamma(np.arange(30)),decimals=5)
        self.assertTrue(np.all(g == np.round(d[0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[49:79],decimals=5)))

    def test_with_double_gamma(self):
        """Test specifydesign with double-gamma convolution"""
        d = sim.specifydesign(self.total_time, self.onsets, self.duration,
                              accuracy=self.acc, conv='double-gamma')
        g = np.round(sim.double_gamma(np.arange(30)),decimals=5)
        self.assertTrue(np.all(g == np.round(d[0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[49:79],decimals=5)))


