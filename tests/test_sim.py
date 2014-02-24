import unittest
import numpy as np

from boldsim import sim

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
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

        # We need to have matching length lists for durations and onsets
        with self.assertRaises(Exception):
            s = sim.stimfunction(10,
                                 onsets=[1, 2, 3],
                                 durations=[1, 2], accuracy=1)

class TestSpecifyDesign(unittest.TestCase):
    """Unit tests for sim.specifydesign"""

    def setUp(self):
        """Setup defaults for stimfunction tests"""
        self.total_time = 100
        self.onsets = [0,49]
        self.duration = 1
        self.acc = 1

    def test_non_matching_onsets_and_single_dur_is_ok(self):
        """Test specifydesign doesn't throw exception with non-matched onsets + single dur"""
        try:
            d = sim.specifydesign(onsets=[1, 2, 3, 4], durations=2)
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_non_matching_onsets_and_single_effect_size_is_ok(self):
        """Test specifydesign doesn't throw exception with non-matched onsets + single effect size"""
        try:
            d = sim.specifydesign(onsets=[1, 2, 3, 4], effect_sizes=2)
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_non_matching_onsets_and_single_dur_in_list_is_ok(self):
        """Test specifydesign doesn't throw exception with non-matched onsets + single dur in list"""
        try:
            d = sim.specifydesign(onsets=[1, 2, 3, 4], durations=[2])
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_non_matching_onsets_and_single_effect_size_in_list_is_ok(self):
        """Test specifydesign doesn't throw exception with non-matched onsets + single effect size in list"""
        try:
            d = sim.specifydesign(onsets=[1, 2, 3, 4], effect_sizes=[2])
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_matching_onsets_and_dur_list_is_ok(self):
        """Test specifydesign doesn't throw exception with matching onsets + dur lists"""
        try:
            d = sim.specifydesign(onsets=[1, 2, 3, 4], durations=[2, 3, 2, 3])
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_matching_onsets_and_effect_size_list_is_ok(self):
        """Test specifydesign doesn't throw exception with matching onsets + effect_size lists"""
        try:
            d = sim.specifydesign(onsets=[[1, 2],[3, 4]], effect_sizes=[2, 3])
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_non_matching_onsets_and_durs_throws_exception(self):
        """Test specifydesign throws exception with non-matched onsets/dur lists"""
        with self.assertRaises(Exception):
            d = sim.specifydesign(onsets=[1, 2, 3, 4], durations=[1, 2])

    def test_non_matching_onsets_and_effect_sizes_throws_exception(self):
        """Test specifydesign throws exception with non-matched onsets/effect_sizes lists"""
        with self.assertRaises(Exception):
            d = sim.specifydesign(onsets=[[1, 2],[3], [4]], effect_sizes=[1, 2])

    def test_nested_non_matching_onsets_and_durs_throws_exception(self):
        """Test specifydesign throws exception with nested non-matched onsets/dur lists"""
        with self.assertRaises(Exception):
            d = sim.specifydesign(onsets=[[1, 2],[1,2,3]], durations=[[1, 2],[1,2]])

    def test_nested_matching_onsets_and_durs_dont_throw_exception(self):
        """Test specifydesign doesn't throw exception with nested matching onsets/dur lists"""
        try:
            d = sim.specifydesign(onsets=[[1, 2],[1,2,3]], durations=[[1, 2],[1,2,3]])
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_nested_matching_onsets_and_single_durs_dont_throw_exception(self):
        """Test specifydesign doesn't throw exception with nested matching onsets + single dur in lists"""
        try:
            d = sim.specifydesign(onsets=[[1, 2],[1,2,3]], durations=[[1],[1,2,3]])
        except Exception as e:
            self.fail('Unexpected exception thrown: ' + e.message)

    def test_output_is_correct_length(self):
        """Test specifydesign returns correct length output"""
        d = sim.specifydesign(100, self.onsets, self.duration,
                              accuracy=1, conv='gamma')
        self.assertTrue(d.shape[1] == 100)

        d = sim.specifydesign(100, self.onsets, self.duration,
                              accuracy=.1, conv='gamma')
        self.assertTrue(d.shape[1] == 100/.1)

    def test_with_no_arguments(self):
        """Test specifydesign with no args matches stimfunction"""
        s = sim.stimfunction()
        d = sim.specifydesign()
        self.assertTrue(np.all(s == d))

    def test_single_with_no_conv(self):
        """Test specifydesign for 1 condition with no convolution matches function"""
        s = sim.stimfunction()
        d = sim.specifydesign(conv='none')
        self.assertTrue(np.all(s == d))

    def test_single_with_gamma(self):
        """Test specifydesign for 1 condition with gamma convolution"""
        d = sim.specifydesign(self.total_time, self.onsets, self.duration,
                              accuracy=self.acc, conv='gamma')
        g = sim.gamma(np.arange(30))
        g /= np.max(g)
        g = np.round(g,decimals=5)
        self.assertTrue(np.all(g == np.round(d[0,0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[0,49:79],decimals=5)))

    def test_single_with_double_gamma(self):
        """Test specifydesign for 1 condition with double-gamma convolution"""
        d = sim.specifydesign(self.total_time, self.onsets, self.duration,
                              accuracy=self.acc, conv='double-gamma')
        g = sim.double_gamma(np.arange(30))
        g /= np.max(g)
        g = np.round(g,decimals=5)
        self.assertTrue(np.all(g == np.round(d[0,0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[0,49:79],decimals=5)))

    def test_multiple_with_no_conv(self):
        """Test specifydesign for 2 conditions with no convolution matches function"""
        onsets = [[0,50],[25,75]]
        duration = 1
        acc = 1
        s1 = sim.stimfunction(onsets=onsets[0], durations=duration)
        s2 = sim.stimfunction(onsets=onsets[1], durations=duration)
        d = sim.specifydesign(onsets=onsets, durations=duration, conv='none')
        self.assertTrue(np.all(s1 == d[0,:]))
        self.assertTrue(np.all(s2 == d[1,:]))

    def test_multiple_with_no_conv_diff_effect_sizes(self):
        """Test specifydesign for 2 conditions with no convolution but diff effect sizes matches function"""
        onsets = [[0,50],[25,75]]
        duration = 1
        effect_sizes = [1,2]
        acc = 1
        s1 = sim.stimfunction(onsets=onsets[0], durations=duration)
        s2 = sim.stimfunction(onsets=onsets[1], durations=duration)
        d = sim.specifydesign(onsets=onsets, durations=duration, effect_sizes=effect_sizes, conv='none')
        self.assertTrue(np.all(s1*effect_sizes[0] == d[0,:]))
        self.assertTrue(np.all(s2*effect_sizes[1] == d[1,:]))

    def test_multiple_with_gamma(self):
        """Test specifydesign for 2 conditions with gamma convolution"""
        onsets = [[0,50],[25]]
        duration = 1
        d = sim.specifydesign(onsets=onsets, durations=duration, conv='gamma')
        g = sim.gamma(np.arange(30))
        g /= np.max(g)
        g = np.round(g,decimals=5)
        self.assertTrue(np.all(g == np.round(d[0,0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[0,50:80],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[1,25:55],decimals=5)))

    def test_multiple_with_gamma_diff_effect_sizes(self):
        """Test specifydesign for 2 conditions with diff effect sizes with gamma convolution"""
        onsets = [[0,50],[25]]
        effect_sizes = [1,2]
        duration = 1
        d = sim.specifydesign(onsets=onsets, durations=duration, effect_sizes=effect_sizes, conv='gamma')
        g = sim.gamma(np.arange(30))
        g /= np.max(g)
        g *= effect_sizes[0]
        g = np.round(g,decimals=5)
        g2 = sim.gamma(np.arange(30))
        g2 /= np.max(g2)
        g2 *= effect_sizes[1]
        g2 = np.round(g2,decimals=5)
        self.assertTrue(np.all(g == np.round(d[0,0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[0,50:80],decimals=5)))
        self.assertTrue(np.all(g2 == np.round(d[1,25:55],decimals=5)))
