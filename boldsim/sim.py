""" Sim module: for simulating fMRI data """
import numpy as np
from numbers import Number
from scipy.stats import rayleigh, norm

def _to_ndarray(arr):
    """
    Convert an object or list to an ndarray
    """
    if not isinstance(arr, np.ndarray):
        if isinstance(arr, list):
            arr = np.asarray(arr)
        else:
            arr = np.asarray([arr])
    return arr

def _handle_dim(dim):
    """
    Handle any of a variety of differnent ways to specify spatial dimensions
    (i.e., None (default), single number, tuple, list)

    Return a copy of the list so we can modify as necessary
    """
    if dim is None:
        mydim = [1]
    elif isinstance(dim, Number):
        mydim = [dim]
    elif isinstance(dim, tuple):
        mydim = list(dim)
    elif isinstance(dim, list):
        mydim = dim[:]
    else:
        raise Exception('Invalid dim provided to lowfreqdrift: {}'.format(dim))
    return mydim

def stimfunction(total_time=100, onsets=range(0, 99, 20),
                 durations=10, accuracy=1):
    """
    Generate a timeseries for a given set of onsets and durations

    Args:
        total_time (int): Total time of design (in seconds)
        onsets (list/ndarray) : Onset times of events (in seconds)
        durations (int/list/ndarray): Duration time/s or events (in seconds)
        accuracy (float): Microtime resolution in seconds

    Returns:
        A ndarray with the stimulus timeseries

    Raises:
        Exception
    """
    if type(total_time) is not int:
        raise Exception("total_time should be an integer")

    # Make sure onsets is an ndarray
    onsets = _to_ndarray(onsets)

    # Make sure durations is an ndarray the same length as onsets
    durations = _to_ndarray(durations)
    if len(durations) == 1:
        durations = durations.repeat(len(onsets))
    if len(durations) != len(onsets):
        raise Exception("durations and onsets need to be same length")

    if np.max(onsets) >= total_time:
        raise Exception("Mismatch between onsets and totaltime")

    resampled_onsets = onsets/accuracy
    resampled_durs = durations/accuracy

    output_series = np.zeros(total_time/accuracy)

    for onset, dur in zip(resampled_onsets, resampled_durs):
        output_series[onset:(onset+dur)] = 1

    return output_series

def specifydesign(total_time=100, onsets=range(0, 99, 20),
                 durations=10, effect_sizes=1, TR=2, accuracy=1,
                 conv='none'):
    """
    Generate a model hemodynamic response for given onsets and durations

    Args:
        total_time (int): Total time of design (in seconds)
        onsets (list/ndarray) : Onset times of events (in seconds)
        durations (int/list/ndarray): Duration time/s of events (in seconds)
        effect_sizes (int/list/ndarray): Effect sizes for conditions
        TR (int/float): Time of sampling
        accuracy (float): Microtime resolution in seconds
        conv (string): Convolution method, one of: "none", "gamma",
                       "double-gamma"

    Returns:
        A ndarray with the stimulus timeseries

    Raises:
        Exception
    """

    # Check to see how if we got a list of onset lists, or just one list
    if isinstance(onsets, list) and any(isinstance(x, list) for x in onsets):
        nconds = len(onsets)
        onsets = [_to_ndarray(x) for x in onsets]
    else:
        nconds = 1
        onsets = [onsets]

    # Check to make sure durations make sense with onsets
    if isinstance(durations, Number):
        # We only got a single number, so assume it is the dur for everything
        durations = [[durations] for x in range(nconds)]
    elif len(durations) == 1:
        durations = [durations for x in range(nconds)]
        # Currently relying on stimfunction to make sure durations matches
    else:
        if any(isinstance(x, list) for x in durations):
            # This is multiple lists, check that each matches or is single item
            if not len(durations) == nconds:
                raise Exception("Num of onset lists and dur lists should match")
        else:
            # This is a single list, make sure it matches the number of onsets
            if nconds > 1:
                raise Exception("Num of onset lists and dur lists should match")
            if not len(durations) == len(onsets[0]):
                raise Exception("Num of onset times and dur times should match")

    # Check to make sure effect sizes make sense with onsets
    if isinstance(effect_sizes, Number):
        # We only got a single number, so assume it is the dur for everything
        effect_sizes = [[effect_sizes] for x in range(nconds)]
    elif len(effect_sizes) == 1:
        effect_sizes = [effect_sizes for x in range(nconds)]
    else:
        if not len(effect_sizes) == len(onsets):
            raise Exception("Num of onset lists and effect sizes should match")

    design_out = np.zeros((nconds, total_time/accuracy))
    for cond, (onset, dur) in enumerate(zip(onsets, durations)):
        stim_timeseries = stimfunction(total_time, onset, dur, accuracy)

        if conv == 'none':
            design_out[cond, :] = stim_timeseries * effect_sizes[cond]

        if conv in ['gamma', 'double-gamma']:
            x = np.arange(0, total_time, accuracy)
            hrf = gamma if conv == 'gamma' else double_gamma
            out = np.convolve(stim_timeseries, hrf(x),
                              mode='full')[0:(len(stim_timeseries))]
            out /= np.max(out, axis=0)
            out *= effect_sizes[cond]
            design_out[cond, :] = out

    return design_out

def gamma(x, fwhm=4):
    """
    Generate a gamma-shaped HRF
    """
    th = 0.242*fwhm

    return 1/(th*6) * (x/th)**3 * np.exp(-x/th)

def double_gamma(x, a1=6., a2=12., b1=.9, b2=.9, c=0.35):
    """
    Generate a double-gamma-shaped HRF
    """
    d1 = a1 * b1
    d2 = a2 * b2

    #pylint: disable=bad-whitespace
    return (    (x / d1)**a1 * np.exp((d1 - x) / b1)) - \
           (c * (x / d2)**a2 * np.exp((d2 - x) / b2))
    #pylint: enable=bad-whitespace

def system_noise(nscan=200, noise_dist='gaussian', sigma=1, dim=None):
    """
    Generate system noise

    Args:
        nscan (int): Total time of design (in scans)
        noise_dist (string): Noise distribution, one of: "gaussian", "rayleigh"
        sigma (float): Sigma of noise distribution
        dim (list): XYZ Dimensions of output

    Returns:
        A ndarray [dim, nscan] with the noise timeseries

    Raises:
        Exception
    """
    # Handle the dim parameter
    mydim = _handle_dim(dim)
    mydim.append(nscan)

    if noise_dist == 'gaussian':
        return norm.rvs(scale=sigma, size=mydim)
    elif noise_dist == 'rayleigh':
        return rayleigh.rvs(scale=sigma, size=mydim)
    else:
        raise Exception('Unknown noise distribution provided: {}'.format(
                                                                  noise_dist))

def lowfreqdrift(nscan=200, freq=128.0, TR=2, dim=None):
    """
    Generate low frequency drift noise

    Args:
        nscan (int): Total time of design (in scans)
        freq (float): Low frequency drift
        TR (int): Repetition time in seconds
        dim (list/tuple): Spatial dimensions of output, default = (1,)

    Returns:
        A ndarray [spatial dim, nscan] with the noise timeseries

    Raises:
        Exception
    """
    # Handle the dim parameter
    mydim = _handle_dim(dim)

    num_basis_funcs = np.floor(2 * (nscan * TR)/freq + 1)
    if num_basis_funcs < 3:
        raise Exception('Number of basis functions is too low. \
                         Longer scanner time or lower freq needed')

    def spm_drift(nscans, nbasis):
        """
        Generate a basis set of cosine functions for low frequency
        drift noise generation
        """
        timepoint = np.arange(nscans)
        cosine_set = np.zeros((nscans, nbasis))

        cosine_set[:, 0] = 1.0/np.sqrt(200)
        for basis in np.arange(1, num_basis_funcs):
            cosine_set[:, basis] = np.sqrt(2.0/nscans) * 10 * \
                                   np.cos(np.pi * (2.0 * timepoint+1) * \
                                          (basis)/(2.0*nscans))

        return cosine_set

    drift_base = spm_drift(nscan, num_basis_funcs)
    drift_image = np.ones(mydim)
    drift_out = np.outer(drift_image, np.sum(drift_base, axis=1))

    mydim.append(nscan)

    return drift_out.reshape(mydim)
