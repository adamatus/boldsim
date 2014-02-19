""" Sim module: for simulating fMRI data """
import numpy as np

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
    if type(onsets) is not np.ndarray:
        if type(onsets) is list:
            onsets = np.asarray(onsets)
        else:
            onsets = np.asarray([onsets])

    # Make sure durations is an ndarray the same length as onsets
    if type(durations) is not np.ndarray:
        if type(durations) is list:
            durations = np.asarray(durations)
            if len(durations) != len(onsets):
                raise Exception("durations and onsets need to be same length")
        else:
            # Repeat single item for each onset
            durations = np.asarray([durations]*len(onsets))

    if np.max(onsets) >= total_time:
        raise Exception("Mismatch between onsets and totaltime")

    resampled_onsets = onsets/accuracy
    resampled_durs = durations/accuracy

    output_series = np.zeros(total_time/accuracy)

    for onset, dur in zip(resampled_onsets, resampled_durs):
        output_series[onset:(onset+dur)] = 1

    return output_series

