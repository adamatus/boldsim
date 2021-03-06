import unittest
import warnings
import numpy as np

from boldsim import sim

class TestHelpers(unittest.TestCase):
    """Unit tests for sim helper functions"""

    def test_handle_nscan_returns_ints(self):
        """Test _handle_nscan returns int with int or float input"""
        self.assertEqual(10, sim._handle_nscan(10))
        self.assertEqual(12, sim._handle_nscan(12.0))

    def test_handle_nscan_throws_exceptions_on_bad_input(self):
        """Test _handle_nscan throws exception on bad input"""
        with self.assertRaises(Exception):
            sim._handle_nscan(10.5)
        with self.assertRaises(Exception):
            sim._handle_nscan([10])
        with self.assertRaises(Exception):
            sim._handle_nscan('10')

class TestHelperVerifyDesignParams(unittest.TestCase):
    """Unit tests for sim._verify_design_params"""

    def assert_all_lists(self,onsets,durations,effect_sizes):
        self.assertTrue(isinstance(onsets, list))
        self.assertTrue(isinstance(durations, list))
        self.assertTrue(isinstance(effect_sizes, list))

    def test_smoke_output_is_three_lists(self):
        """Test _verify_design_params output is reasonable [SMOKE]"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=1,
                                  durations=1,
                                  effect_sizes=1)
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(onsets == [[1]])
        self.assertTrue(durations == [[1]])
        self.assertTrue(effect_sizes == [[1]])


    def test_with_single_duration_as_num_is_ok(self):
        """Test _verify_design_params handles single dur as num"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[10, 12, 14],
                                  durations=1,
                                  effect_sizes=[1, 1, 1])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1]))
        self.assertTrue(effect_sizes == [[1, 1, 1]])

    def test_with_single_duration_as_list_is_ok(self):
        """Test _verify_design_params handles single dur as list"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[10, 12, 14],
                                  durations=[1],
                                  effect_sizes=[1, 1, 1])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1]))
        self.assertTrue(effect_sizes == [[1, 1, 1]])

    def test_with_single_duration_as_list_is_ok_in_multiple_cond(self):
        """Test _verify_design_params handles single dur as list with multiple conds"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[10, 10]],
                                  durations=[1],
                                  effect_sizes=[[1, 1, 1], [1, 1]])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1]))
        self.assertTrue(np.all(effect_sizes[0] == [1, 1, 1]))
        self.assertTrue(np.all(onsets[1] == [10, 10]))
        self.assertTrue(np.all(durations[1] == [1, 1]))
        self.assertTrue(np.all(effect_sizes[1] == [1, 1]))

    def test_with_single_duration_as_num_is_ok_in_multiple_cond(self):
        """Test _verify_design_params handles single dur as num with multiple conds"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[10, 10]],
                                  durations=1,
                                  effect_sizes=[[1, 1, 1], [1, 1]])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1]))
        self.assertTrue(np.all(onsets[1] == [10, 10]))
        self.assertTrue(np.all(durations[1] == [1, 1]))

    def test_with_complex_single_durations_as_num_is_ok_in_multiple_cond(self):
        """Test _verify_design_params handles complex single durs as list with multiple conds"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[10, 10]],
                                  durations=[1,[2]],
                                  effect_sizes=[[1, 1, 1], [1, 1]])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1]))
        self.assertTrue(np.all(onsets[1] == [10, 10]))
        self.assertTrue(np.all(durations[1] == [2, 2]))


    def test_handles_single_ndarry_onset(self):
        """Test _verify_design_params handles single ndarray onsets"""
        onsets=np.arange(1,200,40)
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=onsets,
                                  durations=[1],
                                  effect_sizes=[1])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [1,41,81,121,161]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1, 1, 1]))

    def test_handles_list_of_ndarry_onset(self):
        """Test _verify_design_params handles single ndarray onsets"""
        onsets=[np.arange(1,200,40),np.arange(15,200,40)]
        print onsets
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=onsets,
                                  durations=[1,2],
                                  effect_sizes=[1])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [1,41,81,121,161]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1, 1, 1]))
        self.assertTrue(np.all(onsets[1] == [15, 55, 95, 135, 175]))
        self.assertTrue(np.all(durations[1] == [2, 2, 2, 2, 2]))

    def test_with_complex_one_single_durations_as_num_is_ok_in_multiple_cond(self):
        """Test _verify_design_params handles complex one single durs as list with multiple conds"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[10, 10]],
                                  durations=[1,[2, 3]],
                                  effect_sizes=[[1, 1, 1], [1, 1]])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 1, 1]))
        self.assertTrue(np.all(onsets[1] == [10, 10]))
        self.assertTrue(np.all(durations[1] == [2, 3]))

    def test_with_complex_one_single_durations_as_num_throws_exception_in_multiple_cond(self):
        """Test _verify_design_params throws exception with complex one single durs with multiple conds"""
        with self.assertRaises(Exception):
            onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[10, 10]],
                                  durations=[1,[2, 3, 4]],
                                  effect_sizes=[[1, 1, 1], [1, 1]])

    def test_with_too_many_durations_lists_throws_exception_in_multiple_cond(self):
        """Test _verify_design_params throws exception with too many dur lists"""
        with self.assertRaises(Exception):
            onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[10, 10]],
                                  durations=[1,[2, 3, 4],[1, 2, 3]],
                                  effect_sizes=[[1, 1, 1], [1, 1]])

    def test_with_non_match_duration_single_list_throws_exception(self):
        """Test _verify_design_params throws exception on single nonmatch list"""
        with self.assertRaises(Exception):
            onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[10, 12, 14],
                                  durations=[1, 2],
                                  effect_sizes=[1, 1, 1])

    def test_with_non_match_effects_single_list_throws_exception(self):
        """Test _verify_design_params throws exception on single nonmatch list"""
        with self.assertRaises(Exception):
            onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[10, 12, 14],
                                  durations=[1, 2, 1],
                                  effect_sizes=[1, 1])

    def test_with_matching_lists_is_ok(self):
        """Test _verify_design_params handles matching lists"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[10, 12, 14],
                                  durations=[1, 2, 3],
                                  effect_sizes=[4, 5, 6])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 2, 3]))
        self.assertTrue(np.all(effect_sizes[0] == [4, 5, 6]))

    def test_with_matching_nested_lists_is_ok(self):
        """Test _verify_design_params handles matching nested lists"""
        onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[1,2]],
                                  durations=[[1, 2, 3], [1, 1]],
                                  effect_sizes=[[1, 1, 1],[1,1]])
        self.assert_all_lists(onsets,durations,effect_sizes)
        self.assertTrue(np.all(onsets[0] == [10, 12, 14]))
        self.assertTrue(np.all(durations[0] == [1, 2, 3]))
        self.assertTrue(np.all(onsets[1] == [1, 2]))
        self.assertTrue(np.all(durations[1] == [1, 1]))

    def test_with_non_match_duration_nested_list_throws_exception(self):
        """Test _verify_design_params throws exception on bad dur list"""
        with self.assertRaises(Exception):
            onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[1,2]],
                                  durations=[[1, 2, 3], [1, 1, 3]],
                                  effect_sizes=[[1, 1, 1],[1,1]])

    def test_with_non_match_effect_nested_list_throws_exception(self):
        """Test _verify_design_params throws exception on bad effect list"""
        with self.assertRaises(Exception):
            onsets, durations, effect_sizes = sim._verify_design_params(
                                  onsets=[[10, 12, 14],[1,2]],
                                  durations=[[1, 2, 3], [1, 1]],
                                  effect_sizes=[[1, 1, 1],[1,1, 3]])

class TestStimfunction(unittest.TestCase):
    """Unit tests for sim.stimfunction"""

    def setUp(self):
        """Setup defaults for stimfunction tests"""
        self.total_time = 8
        self.onsets = [0,4]
        self.duration = 2
        self.acc = 1

    def test_total_time_is_number(self):
        """Test stimfunction only takes number for total_time"""
        d = sim.stimfunction(total_time=100)
        d = sim.stimfunction(total_time=100.5, accuracy=0.5)
        with self.assertRaises(Exception):
            d = sim.stimfunction(total_time=[100.5])

    def test_throws_exception_on_multiple_conditions(self):
        """Test stimfunction throws exception on multiple conditions"""
        with self.assertRaises(Exception):
            d = sim.stimfunction(total_time=100, onsets=[[1,2],[3,4]])

    def test_output_is_correct_length(self):
        """Test stimfunction returns correct length output"""
        d = sim.stimfunction(100, self.onsets, self.duration,
                              accuracy=1)
        self.assertTrue(len(d) == 100)

        d = sim.stimfunction(100, self.onsets, self.duration,
                              accuracy=.1)
        self.assertTrue(len(d) == 100/.1)

    def test_with_different_effect_sizes(self):
        """Test stimfunction with different_effect_sizes"""
        s = sim.stimfunction(8, [0, 4],
                             2, [2, 3], 1)
        self.assertTrue(np.all(s == [2, 2, 0, 0, 3, 3, 0, 0]))

    def test_with_acc_of_one(self):
        """Test stimfunction with accuracy=1"""
        s = sim.stimfunction(self.total_time, self.onsets,
                             self.duration, 1)
        self.assertTrue(np.all(s == [1, 1, 0, 0, 1, 1, 0, 0]))

    def test_with_acc_of_half(self):
        """Test stimfunction with accuracy=.5"""
        s = sim.stimfunction(self.total_time, self.onsets,
                             self.duration, accuracy=.5)
        self.assertTrue(np.all(s == [1, 1, 1, 1, 0, 0, 0, 0,
                                     1, 1, 1, 1, 0, 0, 0, 0]))

    def test_with_acc_of_two(self):
        """Test stimfunction with accuracy=2"""
        s = sim.stimfunction(self.total_time, self.onsets,
                             self.duration, accuracy=2)
        self.assertTrue(np.all(s == [1, 0, 1, 0]))

    def test_truncation_warning(self):
        """Test stimfunction warns user if total_time is not divisible by accuracy"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = sim.stimfunction(total_time=100.5, accuracy=1.0)
            d = sim.stimfunction(total_time=100.5, accuracy=0.5)
            self.assertTrue(len(w) == 1)

    def test_durations_past_total_time_warning(self):
        """Test stimfunction warns user if onsets/durations go past total_time"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = sim.stimfunction(total_time=10, onsets=[8,15], durations=1,
                                 accuracy=1.0) # Warn
            d = sim.stimfunction(total_time=10, onsets=[9], durations=2,
                                 accuracy=1.0) # Warn
            d = sim.stimfunction(total_time=10, onsets=[8], durations=1,
                                 accuracy=1.0) # Don't Warn
            self.assertTrue(len(w) == 2)

class TestSpecifyDesign(unittest.TestCase):
    """Unit tests for sim.specifydesign"""

    def setUp(self):
        """Setup defaults for stimfunction tests"""
        self.total_time = 100
        self.onsets = [0,49]
        self.duration = 1
        self.acc = 1

    def test_non_numeric_total_time_throws_exception(self):
        """Test specifydesign throws exception with non-numeric total_time"""
        with self.assertRaises(Exception):
            d = sim.specifydesign(total_time='bad',onsets=[1, 2], durations=[1, 2])

    def test_non_evenly_divisible_sampling_throws_exception(self):
        """Test specifydesign throws exception with total_time/TR problem"""
        with self.assertRaises(Exception):
            d = sim.specifydesign(total_time=10.5,onsets=[1, 2], durations=[1, 2],TR=2)

    def test_output_is_correct_length(self):
        """Test specifydesign returns correct length output"""
        total_time = 100
        TR = 2
        d = sim.specifydesign(total_time=total_time, onsets=self.onsets,
                              durations=self.duration, TR=TR, accuracy=1,
                              conv='gamma')
        self.assertTrue(d.shape[1] == total_time/TR)

        d = sim.specifydesign(total_time=total_time, onsets=self.onsets,
                              durations=self.duration, TR=TR, accuracy=.1,
                              conv='gamma')
        self.assertTrue(d.shape[1] == total_time/TR)

    def test_with_no_arguments(self):
        """Test specifydesign with no args matches stimfunction"""
        s = sim.stimfunction()
        d = sim.specifydesign()
        self.assertTrue(np.all(s[::2] == d))

    def test_single_with_gamma(self):
        """Test specifydesign for 1 condition with gamma convolution"""
        d = sim.specifydesign(self.total_time, self.onsets, self.duration,
                              accuracy=1, TR=1, conv='gamma')
        g = sim.gamma(np.arange(30))
        g = np.round(g,decimals=5)
        self.assertTrue(np.all(g == np.round(d[0,0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[0,49:79],decimals=5)))

    def test_single_with_double_gamma(self):
        """Test specifydesign for 1 condition with double-gamma convolution"""
        d = sim.specifydesign(self.total_time, self.onsets, self.duration,
                              accuracy=1, TR=1, conv='double-gamma')
        g = sim.double_gamma(np.arange(30))
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
        self.assertTrue(np.all(s1[::2] == d[0,:]))
        self.assertTrue(np.all(s2[::2] == d[1,:]))

    def test_multiple_with_no_conv_diff_effect_sizes(self):
        """Test specifydesign for 2 conditions with no convolution but diff effect sizes matches function"""
        onsets = [[0,50],[25,75]]
        duration = 1
        effect_sizes = [1,2]
        acc = 1
        s1 = sim.stimfunction(onsets=onsets[0], durations=duration, accuracy=1)
        s2 = sim.stimfunction(onsets=onsets[1], durations=duration, accuracy=1)
        d = sim.specifydesign(onsets=onsets, durations=duration, effect_sizes=effect_sizes, conv='none')
        self.assertTrue(np.all(s1[::2]*effect_sizes[0] == d[0,:]))
        self.assertTrue(np.all(s2[::2]*effect_sizes[1] == d[1,:]))

    def test_multiple_with_gamma(self):
        """Test specifydesign for 2 conditions with gamma convolution"""
        onsets = [[0,50],[25]]
        duration = 1
        d = sim.specifydesign(onsets=onsets, durations=duration, accuracy=1, TR=1, conv='gamma')
        g = sim.gamma(np.arange(30))
        g = np.round(g,decimals=5)
        self.assertTrue(np.all(g == np.round(d[0,0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[0,50:80],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[1,25:55],decimals=5)))

class TestSpecifyRegion(unittest.TestCase):
    """Unit tests for sim.specifyregion"""

    def test_default_output_is_sane(self):
        """Test specifyregion default output is sane [SMOKE]"""
        region = sim.specifyregion()
        self.assertTrue(isinstance(region, np.ndarray))
        self.assertTrue(np.all(region.shape == (9,9)))

    def test_dim_is_2d_or_3d(self):
        """Test specifyregion dim is 2d or 3d"""
        sim.specifyregion(dim=[32,32], coord=[1,1])
        sim.specifyregion(dim=[64,64,32], coord=[1,1,1])
        with self.assertRaises(Exception):
            sim.specifyregion(dim=[32], coord=[1])
        with self.assertRaises(Exception):
            sim.specifyregion(dim=[32,32,32,32], coord=[1,1,1,1])

    def test_mismatching_coords_and_dims_throws_exception(self):
        """Test specifyregion throws exception on mismatching dims and coords"""
        sim.specifyregion(dim=[64,64,32], coord=[[1,1,1],[2,2,2]])
        with self.assertRaises(Exception):
            sim.specifyregion(dim=[64,64,32], coord=[[1,1,1,1],[2,2,2,2]])

    def test_manual_2d_returns_expected(self):
        """Test specifyregion returns expected for 2d manual input"""
        region = sim.specifyregion(dim=[3,3], coord=[[1,1],[2,0]],radius=0, form='manual')
        self.assertTrue(np.all(region == np.asarray([[0,0,0],[0,1,0],[1,0,0]])))

    def test_manual_3d_returns_expected(self):
        """Test specifyregion returns expected for 3d manual input"""
        region = sim.specifyregion(dim=[2,2,2], coord=[[0,1,1],[1,0,0]],radius=0, form='manual')
        expected = np.asarray([[[0,0],[0,1]],
                               [[1,0],[0,0]]])
        self.assertTrue(np.all(region == expected))

    def test_cube_2d_returns_expected(self):
        """Test specifyregion returns expected for 2d cube input"""
        region = sim.specifyregion(dim=[7,7], coord=[[1,1],[5,5]],radius=[1,2], form='cube')
        expected = np.asarray([[1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1],
                              ])
        self.assertTrue(np.all(region == expected))

    def test_sphere_2d_returns_expected(self):
        """Test specifyregion returns expected for 2d cube input"""
        region = sim.specifyregion(dim=[7,7], coord=[[1,1],[5,5]],radius=[1,2], form='sphere')
        expected = np.asarray([[0, 1, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 0, 0, 1, 1, 1],
                              ])
        self.assertTrue(np.all(region == expected))

    def test_mixed_2d_returns_expected(self):
        """Test specifyregion returns expected for 2d mixed input"""
        region = sim.specifyregion(dim=[7,7], coord=[[1,1],[5,5],[6,0]],radius=[1,2,0],
                                   form=['cube','sphere','manual'])
        expected = np.asarray([[1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1],
                               [1, 0, 0, 0, 1, 1, 1],
                              ])
        self.assertTrue(np.all(region == expected))

    def test_faded_2d_output_is_sane(self):
        """Test specifyregion faded 2d output is sane [SMOKE]"""
        region = sim.specifyregion(dim=[7,7], coord=[[1,1],[5,5]],radius=[1,2], fading=.5, form='cube')
        self.assertTrue(isinstance(region, np.ndarray))
        self.assertTrue(np.all(region.shape == (7,7)))

    def test_faded_3d_output_is_sane(self):
        """Test specifyregion faded 3d output is sane [SMOKE]"""
        region = sim.specifyregion()
        region = sim.specifyregion(dim=[7,7,7], coord=[[1,1,1],[5,5,5]],radius=[1,2], fading=.5, form='cube')
        self.assertTrue(isinstance(region, np.ndarray))
        self.assertTrue(np.all(region.shape == (7,7,7)))


class TestSystemNoise(unittest.TestCase):
    """Unit tests for sim.systemnoise"""

    def test_default_returns_gaussian_noise(self):
        """Test systemnoise default arguments"""
        noise = sim.system_noise()
        noise_mean = np.mean(noise)
        noise_sd = np.std(noise)
        self.assertTrue(-.5 < noise_mean < .5)
        self.assertTrue(.8 < noise_sd < 1.2)

    def test_bad_dist_throws_exception(self):
        """Test systemnoise throws exception with bad dist"""
        with self.assertRaises(Exception):
            sim.system_noise(noise_dist='butterworth')

    def test_return_correct_length_output(self):
        """Test systemnoise returns correct length output"""
        self.assertTrue(len(sim.system_noise(nscan=200,noise_dist='gaussian',sigma=1)[0]) == 200)
        self.assertTrue(len(sim.system_noise(nscan=113,noise_dist='gaussian',sigma=1)[0]) == 113)
        self.assertTrue(len(sim.system_noise(nscan=200,noise_dist='rayleigh',sigma=1)[0]) == 200)
        self.assertTrue(len(sim.system_noise(nscan=113,noise_dist='rayleigh',sigma=1)[0]) == 113)
        self.assertTrue(len(sim.system_noise(nscan=200,noise_dist='gaussian',sigma=10)[0]) == 200)
        self.assertTrue(len(sim.system_noise(nscan=113,noise_dist='gaussian',sigma=10)[0]) == 113)
        self.assertTrue(len(sim.system_noise(nscan=200,noise_dist='rayleigh',sigma=10)[0]) == 200)
        self.assertTrue(len(sim.system_noise(nscan=113,noise_dist='rayleigh',sigma=10)[0]) == 113)

    def test_return_correct_dim_output(self):
        """Test systemnoise returns correct dimension output"""
        self.assertTrue(sim.system_noise(nscan=100).shape == (1,100))
        self.assertTrue(sim.system_noise(nscan=200).shape == (1,200))

    def test_return_correct_dim_handles_numeric(self):
        """Test systemnoise returns correct dimension with single numeric input"""
        self.assertTrue(sim.system_noise(nscan=100,dim=2).shape == (2,100))
        self.assertTrue(sim.system_noise(nscan=200,dim=3).shape == (3,200))

    def test_return_correct_dim_handles_list(self):
        """Test systemnoise returns correct dimensions with list input"""
        self.assertTrue(sim.system_noise(nscan=100,dim=[2]).shape == (2,100))
        self.assertTrue(sim.system_noise(nscan=200,dim=[2,2]).shape == (2,2,200))

    def test_return_correct_dim_handles_typle(self):
        """Test systemnoise returns correct dimensions with tuple input"""
        self.assertTrue(sim.system_noise(nscan=100,dim=(2)).shape == (2,100))
        self.assertTrue(sim.system_noise(nscan=200,dim=(2,2)).shape == (2,2,200))

    def test_throws_exception_on_bad_dim(self):
        """Test systemnoise throws error on bad dim parameter"""
        with self.assertRaises(Exception):
            sim.system_noise(dim='bad')

    def test_returns_good_gaussian_noise(self):
        """Test systemnoise returns good Gaussian noise"""
        for sigma in [1, 10]:
            noise = sim.system_noise(nscan=1000,noise_dist='gaussian',sigma=sigma)
            noise_mean = np.mean(noise[0])
            noise_sd = np.std(noise[0])
            self.assertTrue((-.2*sigma) < noise_mean < (.20*sigma))
            self.assertTrue((sigma*.8) < noise_sd < (sigma*1.2))

    def test_returns_good_rayleigh_noise(self):
        """Test systemnoise returns good Rayleigh noise"""
        for sigma in [1, 10]:
            noise = sim.system_noise(nscan=1000,noise_dist='rayleigh',sigma=sigma)

            correct_mean = sigma * np.sqrt(np.pi/2)
            correct_var = (4-np.pi)/2 * sigma**2

            noise_mean = np.mean(noise[0])
            noise_var = np.var(noise[0])

            self.assertTrue((correct_mean*.8) < noise_mean < (correct_mean*1.2))
            self.assertTrue((correct_var*.8) < noise_var < (correct_var*1.2))

class TestLowFreqNoise(unittest.TestCase):
    """Unit tests for sim.lowfreqnoise"""

    def test_default_returns_expected(self):
        """Test lowfreqnoise default arguments"""
        noise = sim.lowfreqdrift()
        self.assertTrue(noise.shape == (1,200))

    def test_too_short_and_too_fast_throws_exception(self):
        """Test lowfreqnoise with bad lenght/freq combos"""
        with self.assertRaises(Exception):
            noise = sim.lowfreqdrift(nscan=10)
        with self.assertRaises(Exception):
            noise = sim.lowfreqdrift(nscan=200, freq=1000)

    def test_dim_as_number(self):
        """Test lowfreqnoise handles numeric dim"""
        noise = sim.lowfreqdrift(nscan=100,dim=10)
        self.assertTrue(noise.shape == (10,100))

    def test_dim_as_tuple(self):
        """Test lowfreqnoise handles tuple dim"""
        noise = sim.lowfreqdrift(nscan=100,dim=(64, 64))
        self.assertTrue(noise.shape == (64,64,100))

    def test_dim_as_list(self):
        """Test lowfreqnoise handles list dim"""
        noise = sim.lowfreqdrift(nscan=100,dim=[32, 32])
        self.assertTrue(noise.shape == (32,32,100))

    def test_bad_dim_throws_exception(self):
        """Test lowfreqnoise with bad dim throws exception"""
        with self.assertRaises(Exception):
            noise = sim.lowfreqdrift(dim='bad')

    def test_reshape_is_good(self):
        """Test lowfreqnoise reshape returns expected timeseries"""
        noise2 = sim.lowfreqdrift(dim=(2,2),random_phase=0, random_weights=0)
        self.assertTrue(noise2.shape == (2,2,200))
        self.assertTrue(np.all(noise2[0,1,:] == noise2[1,1,:]))

class TestPhysNoise(unittest.TestCase):
    """Unit tests for sim.physnoise"""

    def test_default_returns_expected(self):
        """Test physnoise default arguments"""
        noise = sim.physnoise()
        self.assertTrue(noise.shape == (1,200))

    def test_reshape_is_good(self):
        """Test physnoise reshape returns expected timeseries"""
        noise1 = sim.physnoise()
        noise2 = sim.physnoise(dim=(2,2))
        self.assertTrue(np.all(noise1[0,:] == noise2[0,0,:]))
        self.assertTrue(np.all(noise1[0,:] == noise2[1,1,:]))

class TestTaskNoise(unittest.TestCase):
    """Unit tests for sim.tasknoise"""

    def test_default_returns_expected(self):
        """Test tasknoise default arguments"""
        d = sim.specifydesign()
        noise = sim.tasknoise(design=d)
        self.assertTrue(noise.shape == (1,50))

    def test_handles_simple_list_design(self):
        """Test tasknoise handles simple list design"""
        noise = sim.tasknoise(design=[0, 0, 0, 0, 1, 1, 1, 1])
        self.assertTrue(noise.shape == (1,8))

    def test_noise_is_only_during_task(self):
        """Test tasknoise is only present during task"""
        design = np.concatenate((np.zeros(1000),np.ones(1000)))
        noise = sim.tasknoise(design=design, sigma=1)
        self.assertAlmostEqual(0.0, np.std(noise[0,design==0]))
        good_noise_sd = np.std(noise[0,design==1])
        self.assertTrue(.8 < good_noise_sd < 1.2 )

class TestTemporalNoise(unittest.TestCase):
    """Unit tests for sim.temporalnoise"""

    def test_default_returns_expected(self):
        """Test temporalnoise default arguments [SMOKE]"""
        noise = sim.temporalnoise()
        self.assertTrue(noise.shape == (1,200))

    def test_handles_ar_coeff_list(self):
        """Test temporalnoise with ar_coef arguments [SMOKE]"""
        noise = sim.temporalnoise(ar_coef=[.2, .3, .2])
        self.assertTrue(noise.shape == (1,200))

    def test_handles_reshape(self):
        """Test temporalnoise provides sane output shape [SMOKE]"""
        noise = sim.temporalnoise(nscan=150, ar_coef=[.2, .3, .2], dim=[3, 3])
        self.assertTrue(noise.shape == (3,3,150))

class TestSpatialNoise(unittest.TestCase):
    """Unit tests for sim.spatialnoise"""

    def test_throws_exception_on_vector(self):
        """Test spatialnoise throws exception on vector"""
        with self.assertRaises(Exception):
            sim.spatialnoise(dim=[10])

    def test_throws_exception_on_high_dim(self):
        """Test spatialnoise throws exception on higher dim"""
        with self.assertRaises(Exception):
            sim.spatialnoise(dim=[10,10,10,10])

    def test_throws_exception_on_unknown_method(self):
        """Test spatialnoise throws exception on unknownmethod"""
        with self.assertRaises(Exception):
            sim.spatialnoise(method='magic',dim=[10,10])

    def test_corr_3d_produces_resonable_output_dim(self):
        """Test spatialnoise corr 3d produces correct size output [SMOKE]"""
        noise = sim.spatialnoise(nscan=20, dim=[10,12,15])
        self.assertEqual(noise.shape, (10, 12, 15, 20))

    def test_corr_2d_produces_resonable_output_dim(self):
        """Test spatialnoise corr 2d produces correct size output [SMOKE]"""
        noise = sim.spatialnoise(nscan=20, dim=[10,12])
        self.assertEqual(noise.shape, (10, 12, 20))

    def test_gaussRF_2d_produces_reasonable_output_dim(self):
        """Test spatialnoise gaussRF 2d produces correct size output [SMOKE]"""
        noise = sim.spatialnoise(nscan=20, dim=[10,12], method='gaussRF')
        self.assertEqual(noise.shape, (10, 12, 20))

    def test_gaussRF_3d_produces_reasonable_output_dim(self):
        """Test spatialnoise gaussRF 3d produces correct size output [SMOKE]"""
        noise = sim.spatialnoise(nscan=20, dim=[10,12,15], method='gaussRF')
        self.assertEqual(noise.shape, (10, 12, 15, 20))

class TestSimPrepTemporal(unittest.TestCase):
    """Unit tests for sim.simprepTemporal"""

    def test_smoke_output_is_dict(self):
        """Test simprepTemporal output is dict [SMOKE]"""
        design = sim.simprepTemporal()
        self.assertTrue(isinstance(design,dict))

    def test_throws_exception_on_bad_total_time(self):
        """Test simprepTemporal throws exception on bad total_time"""
        with self.assertRaises(Exception):
            design = sim.simprepTemporal(total_time='bad')

    def test_throws_exception_on_bad_total_time_by_TR(self):
        """Test simprepTemporal throws exception on bad total_time/TR"""
        with self.assertRaises(Exception):
            design = sim.simprepTemporal(total_time=100.5, TR=2)

class TestSimPrepSpatial(unittest.TestCase):
    """Unit tests for sim.simprepTemporal"""

    def test_smoke_output_is_list_of_dists(self):
        """Test simprepSpatial output is list of dicts [SMOKE]"""
        regions = sim.simprepSpatial()
        self.assertTrue(isinstance(regions,dict))

    def test_throws_exception_on_bad_fade(self):
        """Test simprepSpatial throws exception on bad fade value"""
        sim.simprepSpatial(fading=.5)
        sim.simprepSpatial(fading=0)
        sim.simprepSpatial(fading=1.0)
        with self.assertRaises(Exception):
            sim.simprepSpatial(fading=-1.0)
        with self.assertRaises(Exception):
            sim.simprepSpatial(fading=1.5)

    def test_regions_should_be_int(self):
        """Test simprepSpatial regions arg should be int"""
        regions = sim.simprepSpatial(regions=1)
        regions = sim.simprepSpatial(regions=1.0)
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=1.5)
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=[10])

    def test_bad_form_throws_exception(self):
        """Test simprepSpatial throws exception on unknown form type"""
        regions = sim.simprepSpatial(form='cube')
        regions = sim.simprepSpatial(form='sphere')
        regions = sim.simprepSpatial(form='manual')
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(form='bad')

    def test_coords_should_be_list(self):
        """Test simprepSpatial coords should be list"""
        regions = sim.simprepSpatial(coord=[[1,2,3]])
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(coord=1)

    def test_coords_throws_exception_if_dims_differ(self):
        """Test simprepSpatial throws exception if dims differ"""
        regions = sim.simprepSpatial(regions=2, coord=[[1,2,3],[4,5,6]])
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=2, coord=[[1,2,3],[4,5]])

    def test_throws_exception_if_regions_and_coords_differ(self):
        """Test simprepSpatial throws exception if regions and coords differ"""
        regions = sim.simprepSpatial(regions=2, coord=[[1,2,3],[4,5,6]])
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=1, coord=[[1,2],[4,5]])

    def test_throws_exception_on_args_of_wrong_type(self):
        """Test simprepSpatial throws exception if args are of wrong type"""
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=2, coord=[[1,2,3],1])
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=1, radius = ['2'])
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=1, form = ['2'])
        with self.assertRaises(Exception):
            regions = sim.simprepSpatial(regions=1, fading = ['2'])

    def test_returns_expected_output_on_full_input(self):
        """Test simprepSpatial returns expected output on full input"""
        regions = sim.simprepSpatial(regions=2, coord=[[1,2,3],[4,5,6]],
                                     radius=[3, 4], form=['cube','sphere'],
                                     fading=[0, 1])
        self.assertEqual(regions['coord'][0],[1,2,3])
        self.assertEqual(regions['coord'][1],[4,5,6])
        self.assertEqual(regions['radius'][0],3)
        self.assertEqual(regions['radius'][1],4)
        self.assertEqual(regions['form'][0],'cube')
        self.assertEqual(regions['form'][1],'sphere')
        self.assertEqual(regions['fading'][0],0)
        self.assertEqual(regions['fading'][1],1)

    def test_returns_expected_output_on_some_single_input(self):
        """Test simprepSpatial returns expected output on some single input"""
        regions = sim.simprepSpatial(regions=2, coord=[1,2,3],
                                     radius=4, form=['cube','sphere'],
                                     fading=[0])
        self.assertEqual(regions['coord'][0],[1,2,3])
        self.assertEqual(regions['coord'][1],[1,2,3])
        self.assertEqual(regions['radius'][0],4)
        self.assertEqual(regions['radius'][1],4)
        self.assertEqual(regions['form'][0],'cube')
        self.assertEqual(regions['form'][1],'sphere')
        self.assertEqual(regions['fading'][0],0)
        self.assertEqual(regions['fading'][1],0)

class TestSimTSfMRI(unittest.TestCase):
    """Unit tests for sim.simTSfmri"""

    def setUp(self):
        self.design = sim.simprepTemporal(total_time=200,
                                        onsets=[[1,41, 81, 121, 161],
                                                [15, 55, 95, 135, 175]],
                                        durations=[[20],[7]],
                                        effect_sizes=[3,10], conv='double-gamma')

    def test_no_design_throws_exception(self):
        """Test simTSfmri without design throws exception"""
        with self.assertRaises(Exception):
            #pylint: disable=no-value-for-parameter
            sim.simTSfmri()
            #pylint: enable=no-value-for-parameter

    def test_smoke_output_is_ndarray(self):
        """Test simTSfmri output is ndarray [SMOKE]"""
        design = sim.simTSfmri(design=self.design)
        self.assertTrue(isinstance(design,np.ndarray))

    def test_smoke_all_noise_types_work(self):
        """Test simTSfmri all noise types work [SMOKE]"""
        design = sim.simTSfmri(design=self.design, noise='none')
        self.assertTrue(isinstance(design,np.ndarray))
        design = sim.simTSfmri(design=self.design, noise='white')
        self.assertTrue(isinstance(design,np.ndarray))
        design = sim.simTSfmri(design=self.design, noise='temporal')
        self.assertTrue(isinstance(design,np.ndarray))
        design = sim.simTSfmri(design=self.design, noise='low-freq')
        self.assertTrue(isinstance(design,np.ndarray))
        design = sim.simTSfmri(design=self.design, noise='phys')
        self.assertTrue(isinstance(design,np.ndarray))
        design = sim.simTSfmri(design=self.design, noise='task-related')
        self.assertTrue(isinstance(design,np.ndarray))
        design = sim.simTSfmri(design=self.design, noise='mixture')
        self.assertTrue(isinstance(design,np.ndarray))

    def test_mixture_weight_handling(self):
        """Test simTSfmri handles weight vector appropriately"""
        with self.assertRaises(Exception):
            sim.simTSfmri(design=self.design, noise='mixture', weights=[.1, .1])
        with self.assertRaises(Exception):
            sim.simTSfmri(design=self.design, noise='mixture', weights=[1, 1, 1, 1, 1])

    def test_bad_noise_param_throws_exception(self):
        """Test simTSfmri bad noise type throws exception"""
        with self.assertRaises(Exception):
            sim.simTSfmri(design=self.design, noise='bad')

    def test_output_with_no_noise_one_cond_is_same_as_specifydesign(self):
        """Test simTSfmri output with no noise, one condition is same as specifydesign"""
        total_time=20
        onsets=[[1]]
        durations=[[2]]
        effect_sizes=[3]
        conv='double-gamma'
        design = sim.simprepTemporal(total_time=total_time,
                                     onsets=onsets,
                                     durations=durations,
                                     effect_sizes=effect_sizes, conv=conv, TR=2, accuracy=1)
        simts = sim.simTSfmri(design=design, noise='none', base=0)
        designts = sim.specifydesign(total_time=total_time,
                                     onsets=onsets,
                                     durations=durations,
                                     effect_sizes=effect_sizes, conv=conv, TR=2, accuracy=1)
        self.assertTrue(np.all(simts == designts.squeeze()))

class TestSimVOLfMRI(unittest.TestCase):
    """Unit tests for sim.simVOLfmri"""

    def setUp(self):
        self.design = sim.simprepTemporal(total_time=200,
                                        onsets=[[1,41, 81, 121, 161],
                                                [15, 55, 95, 135, 175]],
                                        durations=[[20],[7]], TR=2,
                                        effect_sizes=[3,10], conv='double-gamma')
        self.image = sim.simprepSpatial(regions=3, coord=[[1,1],[5,5],[6,0]],radius=[1,2,0],
                                   form=['cube','sphere','manual'], fading=[.5, 0, 0])

    def test_smoke_output_is_ndarray(self):
        """Test simVOLfmri output is ndarray [SMOKE]"""
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7,7])
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        self.assertTrue(sim_ds.shape == (7, 7, 100))

    def test_dim_handling(self):
        """Test simVOLfmri dim handling throws exceptions on bad input"""
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image])
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[1,2,3,4])
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=2)

    def test_design_and_image_verification_missing_image(self):
        """Test simVOLfmri throws exception with missing image"""
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design], dim=[7,7])

    def test_design_and_image_verification_missing_design(self):
        """Test simVOLfmri throws exception with missing design"""
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(images=[self.image], dim=[7,7])

    def test_design_and_image_verification_mismatching_design_image_lens(self):
        """Test simVOLfmri throws exception with mismatching design and image lenghts"""
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design, self.design],
                                    images=[self.image, self.image, self.image], dim=[7,7])

    def test_design_and_image_verification_handles_dicts(self):
        """Test simVOLfmri handles design and image and single dicts [SMOKE]"""
        sim_ds = sim.simVOLfmri(designs=self.design, images=self.image, dim=[7,7])

    def test_design_and_image_verification_handles_single_design(self):
        """Test simVOLfmri handles single design and multiple images [SMOKE]"""
        sim_ds = sim.simVOLfmri(designs=[self.design],
                                images=[self.image, self.image, self.image], dim=[7,7])

    def test_design_and_image_verification_handles_single_image(self):
        """Test simVOLfmri handles multiple designs and single image [SMOKE]"""
        sim_ds = sim.simVOLfmri(designs=[self.design, self.design, self.design],
                                images=[self.image], dim=[7,7])

    def test_design_and_image_verification_throws_exception_on_mismatched_trs(self):
        """Test simVOLfmri throws exception on mismatched TRs"""
        design2 = sim.simprepTemporal(total_time=200,
                                        onsets=[[1,41, 81, 121, 161],
                                                [15, 55, 95, 135, 175]],
                                        durations=[[20],[7]], TR=1,
                                        effect_sizes=[3,10], conv='double-gamma')
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design, self.design, design2],
                                    images=[self.image, self.image, self.image], dim=[7,7])

    def test_design_and_image_verification_warns_on_mismatched_times(self):
        """Test simVOLfmri warns on mismatched total_times"""
        design2 = sim.simprepTemporal(total_time=230,
                                        onsets=[[1,41, 81, 121, 161],
                                                [15, 55, 95, 135, 175]],
                                        durations=[[20],[7]], TR=2,
                                        effect_sizes=[3,10], conv='double-gamma')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim_ds = sim.simVOLfmri(designs=[self.design, self.design, design2],
                                    images=[self.image, self.image, self.image], dim=[7,7])
            print w
            self.assertTrue(len(w) == 1)

    def test_design_and_image_verification_returns_longest_on_mismatched_times(self):
        """Test simVOLfmri returns longest on mismatched total_times"""
        design2 = sim.simprepTemporal(total_time=230,
                                        onsets=[[1,41, 81, 121, 161],
                                                [15, 55, 95, 135, 175]],
                                        durations=[[20],[7]], TR=2,
                                        effect_sizes=[3,10], conv='double-gamma')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim_ds = sim.simVOLfmri(designs=[self.design, self.design, design2],
                                    images=[self.image, self.image, self.image], dim=[7,7])
            self.assertTrue(sim_ds.shape == (7,7,115))

    def test_smoke_noise_output(self):
        """Test simVOLfmri returns noise with no design and image [SMOKE]"""
        sim_ds = sim.simVOLfmri(dim=[7,7], noise='white', nscan=100, TR=2)
        self.assertTrue(sim_ds.shape == (7,7,100))

    def test_mixture_weight_handling(self):
        """Test simVOLfmri throws exceptions on bad weights"""
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7,7],
                                    noise='mixture', weights=[1, 1, 1])
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7,7],
                                    noise='mixture', weights=[1, 1, 1, 1, 1, 1])

    def test_bad_noise_type(self):
        """Test simVOLfmri throws exceptions on bad noise type"""
        with self.assertRaises(Exception):
            sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7,7],
                                    noise='bad', weights=[1, 1, 1])


    def test_smoke_all_noise_types_work(self):
        """Test simVOLfmri all noise types work [SMOKE]"""
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='none')
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='white')
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='temporal')
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='low-freq')
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='phys')
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='task-related')
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='spatial')
        self.assertTrue(isinstance(sim_ds, np.ndarray))
        sim_ds = sim.simVOLfmri(designs=[self.design], images=[self.image], dim=[7, 7],noise='mixture')
        self.assertTrue(isinstance(sim_ds, np.ndarray))

