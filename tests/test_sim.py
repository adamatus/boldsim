import unittest
import warnings
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

    def test_total_time_is_number(self):
        """Test stimfunction only takes number for total_time"""
        d = sim.stimfunction(total_time=100)
        d = sim.stimfunction(total_time=100.5, accuracy=0.5)
        with self.assertRaises(Exception):
            d = sim.stimfunction(total_time=[100.5])

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
        """Test stimfunction throws exception with non-matching onsets and durs"""
        # We need to have matching length lists for durations and onsets
        s = sim.stimfunction(10,
                             onsets=[1, 2, 3],
                             durations=[1, 2, 1], accuracy=1)
        with self.assertRaises(Exception):
            s = sim.stimfunction(10,
                                 onsets=[1, 2, 3],
                                 durations=[1, 2], accuracy=1)

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
                                 accuracy=1.0) # Don't warn
            self.assertTrue(len(w) == 2)

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
        d = sim.specifydesign(onsets=[1, 2, 3, 4], durations=2)

    def test_non_matching_onsets_and_single_effect_size_is_ok(self):
        """Test specifydesign doesn't throw exception with non-matched onsets + single effect size"""
        d = sim.specifydesign(onsets=[1, 2, 3, 4], effect_sizes=2)

    def test_non_matching_onsets_and_single_dur_in_list_is_ok(self):
        """Test specifydesign doesn't throw exception with non-matched onsets + single dur in list"""
        d = sim.specifydesign(onsets=[1, 2, 3, 4], durations=[2])

    def test_non_matching_onsets_and_single_effect_size_in_list_is_ok(self):
        """Test specifydesign doesn't throw exception with non-matched onsets + single effect size in list"""
        d = sim.specifydesign(onsets=[1, 2, 3, 4], effect_sizes=[2])

    def test_matching_onsets_and_dur_list_is_ok(self):
        """Test specifydesign doesn't throw exception with matching onsets + dur lists"""
        d = sim.specifydesign(onsets=[1, 2, 3, 4], durations=[2, 3, 2, 3])

    def test_matching_onsets_and_effect_size_list_is_ok(self):
        """Test specifydesign doesn't throw exception with matching onsets + effect_size lists"""
        d = sim.specifydesign(onsets=[[1, 2],[3, 4]], effect_sizes=[2, 3])

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

    def test_nested_non_matching_onsets_and_durs_len_throws_exception(self):
        """Test specifydesign throws exception with nested non-matched onsets/dur len lists"""
        with self.assertRaises(Exception):
            d = sim.specifydesign(onsets=[[1, 2],[1, 2, 3],[1,2,3]], durations=[[1, 2],[1,2]])

    def test_nested_non_matching_onsets_and_durs_nonnest_throws_exception(self):
        """Test specifydesign throws exception with nested non-matched onsets/single dur lists"""
        with self.assertRaises(Exception):
            d = sim.specifydesign(onsets=[[1, 2],[1, 2, 3]], durations=[1, 2])

    def test_nested_matching_onsets_and_durs_dont_throw_exception(self):
        """Test specifydesign doesn't throw exception with nested matching onsets/dur lists"""
        d = sim.specifydesign(onsets=[[1, 2],[1,2,3]], durations=[[1, 2],[1,2,3]])

    def test_nested_matching_onsets_and_single_durs_dont_throw_exception(self):
        """Test specifydesign doesn't throw exception with nested matching onsets + single dur in lists"""
        d = sim.specifydesign(onsets=[[1, 2],[1,2,3]], durations=[[1],[1,2,3]])

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
        g /= np.max(g)
        g = np.round(g,decimals=5)
        self.assertTrue(np.all(g == np.round(d[0,0:30],decimals=5)))
        self.assertTrue(np.all(g == np.round(d[0,49:79],decimals=5)))

    def test_single_with_double_gamma(self):
        """Test specifydesign for 1 condition with double-gamma convolution"""
        d = sim.specifydesign(self.total_time, self.onsets, self.duration,
                              accuracy=1, TR=1, conv='double-gamma')
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
        d = sim.specifydesign(onsets=onsets, durations=duration, effect_sizes=effect_sizes,
                              accuracy=1, TR=1, conv='gamma')
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
        noise1 = sim.lowfreqdrift()
        noise2 = sim.lowfreqdrift(dim=(2,2))
        self.assertTrue(np.all(noise1[0,:] == noise2[0,0,:]))
        self.assertTrue(np.all(noise1[0,:] == noise2[1,1,:]))

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
        """Test temporalnoise default arguments"""
        noise = sim.temporalnoise()
        self.assertTrue(noise.shape == (1,200))

    def test_handles_ar_coeff_list(self):
        """Test temporalnoise default arguments"""
        noise = sim.temporalnoise(ar_coef=[.2, .3, .2])
        self.assertTrue(noise.shape == (1,200))

    def test_handles_reshape(self):
        """Test temporalnoise default arguments"""
        noise = sim.temporalnoise(nscan=150, ar_coef=[.2, .3, .2], dim=[3, 3])
        self.assertTrue(noise.shape == (3,3,150))

