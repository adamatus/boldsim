""" Sim module: for simulating fMRI data """
import numpy as np

def _to_ndarray(arr):
    if type(arr) is not np.ndarray:
        if type(arr) is list:
            arr = np.asarray(arr)
        else:
            arr = np.asarray([arr])
    return arr


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
        durations (int/list/ndarray): Duration time/s or events (in seconds)
        effect_sizes (int/list/ndarray): Effect sizes for conditions
        TR (int): Time of sampling
        accuracy (float): Microtime resolution in seconds
        conv (string): Convolution method, on of: "none", "gamma"

    Returns:
        A ndarray with the stimulus timeseries

    Raises:
        Exception
    """
    stim_timeseries = stimfunction(total_time, onsets, durations, accuracy)

    if conv == 'none':
        return stim_timeseries

    if conv in ['gamma','double-gamma']:
        x = np.arange(0, total_time, accuracy)
        hrf = gamma if conv == 'gamma' else double_gamma
        out = np.convolve(stim_timeseries, hrf(x), mode='full')[0:(len(stim_timeseries))]
        return out

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

    return (    (x / d1)**a1 * np.exp((d1 - x) / b1)) - \
           (c * (x / d2)**a2 * np.exp((d2 - x) / b2))
