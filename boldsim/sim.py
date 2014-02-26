""" Sim module: for simulating fMRI data """
import numpy as np
from warnings import warn
from numbers import Number
from scipy.stats import rayleigh, norm
import statsmodels.tsa.arima_process

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
        raise Exception('Invalid dim provided: {}'.format(dim))
    return mydim

def stimfunction(total_time=100, onsets=range(0, 99, 20),
                 durations=10, accuracy=1):
    """
    Generate a timeseries for a given set of onsets and durations

    Args:
        total_time (int/float): Total time of design (in seconds)
        onsets (list/ndarray) : Onset times of events (in seconds)
        durations (int/list/ndarray): Duration time/s or events (in seconds)
        accuracy (float): Microtime resolution in seconds

    Returns:
        A ndarray with the stimulus timeseries

    Raises:
        Exception
    """
    if not isinstance(total_time, Number):
        raise Exception("Argument total_time should be a number")

    accuracy = float(accuracy)

    # Make sure onsets is an ndarray
    onsets = _to_ndarray(onsets)

    # Make sure durations is an ndarray the same length as onsets
    durations = _to_ndarray(durations)
    if len(durations) == 1:
        durations = durations.repeat(len(onsets))
    if len(durations) != len(onsets):
        raise Exception("Stim durations and onsets lists " +
                        "need to be same length")

    if np.max(onsets+durations) >= total_time:
        warn('Onsets/durations go past total_time. ' +
             'Some events will be truncated/missing')

    resampled_onsets = onsets/accuracy
    resampled_durs = durations/accuracy

    output_len = total_time/accuracy
    if not output_len.is_integer():
        warn('total_time is not evenly divisibly by accuracy, ' +
             'output will be slightly truncated and may not ' +
             'exactly match total_time')

    output_series = np.zeros(int(output_len))

    for onset, dur in zip(resampled_onsets, resampled_durs):
        output_series[onset:(onset+dur)] = 1

    return output_series

def _verify_design_params(onsets, durations, effect_sizes):
    """
    Make sure the onsets, durations, and effect_sizes are sane
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

    return (onsets, durations, effect_sizes)

def specifydesign(total_time=100, onsets=range(0, 99, 20),
                 durations=10, effect_sizes=1, TR=2, accuracy=1,
                 conv='none'):
    """
    Generate a model hemodynamic response for given onsets and durations

    Args:
        total_time (int/float): Total time of design (in seconds)
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
    if not isinstance(total_time, Number):
        raise Exception("Argument total_time should be a number")

    TR = float(TR)
    accuracy = float(accuracy)

    output_len = total_time/TR
    if not output_len.is_integer():
        raise Exception("total_time is not divisible by TR")

    onsets, durations, effect_sizes = _verify_design_params(onsets,
                                                            durations,
                                                            effect_sizes)

    design_out = np.zeros((len(onsets), total_time/TR))
    sample_idx = np.round(np.arange(0, total_time/accuracy, TR/accuracy))
    sample_idx = np.asarray(sample_idx, dtype=np.int)

    for cond, (onset, dur) in enumerate(zip(onsets, durations)):
        stim_timeseries = stimfunction(total_time, onset, dur, accuracy)

        if conv == 'none':
            design_out[cond, :] = stim_timeseries[sample_idx] * \
                                  effect_sizes[cond]

        if conv in ['gamma', 'double-gamma']:
            x = np.arange(0, total_time, accuracy)
            hrf = gamma if conv == 'gamma' else double_gamma
            out = np.convolve(stim_timeseries, hrf(x),
                              mode='full')[0:(len(stim_timeseries))]
            out /= np.max(out, axis=0)
            out *= effect_sizes[cond]
            design_out[cond, :] = out[sample_idx]

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
        TR (int/float): Repetition time in seconds
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

def physnoise(nscan=200, sigma=1, freq_heart=1.17, freq_respiration=0.2, \
              TR=2, dim=None):
    """
    Generate physiological (cardiac and repiratory) noise

    Args:
        nscan (int): Total time of design (in scans)
        freq_heart (float): Heart rate
        freq_respiration (float): Repiration rate
        TR (int/float): Repetition time in seconds
        dim (list/tuple): Spatial dimensions of output, default = (1,)

    Returns:
        A ndarray [spatial dim, nscan] with the noise timeseries

    Raises:
        Exception
    """

    # Handle the dim parameter
    mydim = _handle_dim(dim)

    heart_beat = 2 * np.pi * freq_heart * TR
    repiration = 2 * np.pi * freq_respiration * TR
    timepoints = np.arange(nscan)

    hr_drift = np.sin(heart_beat * timepoints) + \
               np.cos(repiration * timepoints)
    hr_sigma = np.std(hr_drift)
    hr_weight = sigma/hr_sigma

    noise_image = np.ones(mydim)
    noise_out = np.outer(noise_image, hr_drift * hr_weight)
    mydim.append(nscan)

    return noise_out.reshape(mydim)

def tasknoise(design, sigma=1, noise_dist='gaussian', dim=None):
    """
    Generate task-related noise

    Args:
        design (ndarray [spatial dims, nscan]): Output from specify design
        noise_dist (string): Noise distribution, one of: "gaussian", "rayleigh"
        sigma (float): Sigma of noise distribution
        dim (list/tuple): Spatial dimensions of output, default = (1,)

    Returns:
        A ndarray [spatial dim, nscan] with the noise timeseries

    Raises:
        Exception
    """

    # Handle a single list by making it into a matrix
    design = _to_ndarray(design)
    if len(design.shape) == 1:
        design = design.reshape((1, design.shape[0]))

    noise = system_noise(nscan=design.shape[-1], sigma=sigma,
                         noise_dist=noise_dist, dim=dim)

    return noise * np.apply_along_axis(sum, axis=0, arr=design)

def temporalnoise(nscan=200, sigma=1, ar_coef=0.2, dim=None):
    """
    Generate temporally correlated noise

    Args:
        nscan (int): Total time of design (in scans)
        sigma (float): Sigma of noise distribution
        ar_coef (float/list): Autocorrelation coefficients. Length of list
                         determines order of autoregressive model,
                         default = [.2]
        dim (list/tuple): Spatial dimensions of output, default = (1,)

    Returns:
        A ndarray [spatial dim, nscan] with the noise timeseries

    Raises:
        Exception
    """

    # Handle the dim parameter
    mydim = _handle_dim(dim)

    # Convert to ndarray and pop 1 on the front
    ar_coef = np.concatenate(([1], _to_ndarray(ar_coef)))

    # Get one big random series, then chop it up
    samples = nscan * np.prod(mydim)
    tnoise = statsmodels.tsa.arima_process.arma_generate_sample(ar=ar_coef,
                                                                ma=[1, 0],
                                                                nsample=samples,
                                                                sigma=sigma,
                                                                burnin=500)
    mydim.append(nscan)
    return tnoise.reshape(mydim)

def spatialnoise(nscan=200, method='corr', noise_dist='gaussian', sigma=1, \
                 rho=0.75, FWHM=4, gamma_shape=6, gamma_rate=1, dim=None):
    """
    Generate spatially correlated noise

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

    if len(mydim) == 1:
        raise Exception('Spatially noise is not defined for vectors')
    elif len(mydim) > 3:
        raise Exception('Image space with more than 3 dimensions not supported')

    if method == 'corr':
        noise = np.zeros(mydim + [nscan])
        for scan in range(nscan):
            start = system_noise(nscan=1, sigma=sigma,
                    noise_dist=noise_dist, dim=mydim).squeeze()

            noise_scan = np.zeros(mydim)

            if len(mydim) == 2:
                noise_scan[0, 0] = start[0, 0]

                # Fill in first column with correlated noise
                for i in range(1, mydim[0]):
                    noise_scan[i, 0] = rho * noise_scan[i-1, 0] + \
                                      np.sqrt(1-rho**2) * start[i, 0]

                # Fill in remaining columns with correlated noise
                for j in range(1, mydim[1]):
                    noise_scan[:, j] = rho * noise_scan[:, j-1] + \
                                      np.sqrt(1-rho**2) * start[:, j]

                # Add correlation across rows
                for i in range(1, mydim[0]):
                    noise_scan[i, 1:] = rho * noise_scan[i-1, 1:] + \
                                        np.sqrt(1-rho**2) * noise_scan[i, 1:]

                noise[:, :, scan] = noise_scan

            else: # 3 dim spatial noise
                noise_scan[0, 0, 0] = start[0, 0, 0]

                # Fill in first column with correlated noise
                for i in range(1, mydim[0]):
                    noise_scan[i, 0, 0] = rho * noise_scan[i-1, 0, 0] + \
                                          np.sqrt(1-rho**2) * start[i, 0, 0]

                # Fill in remaining columns with correlated noise
                for j in range(1, mydim[1]):
                    noise_scan[:, j, 0] = rho * noise_scan[:, j-1, 0] + \
                                          np.sqrt(1-rho**2) * start[:, j, 0]

                # Fill in remaining 3rd dim with correlated noise
                for k in range(1, mydim[2]):
                    noise_scan[:, :, k] = rho * noise_scan[:, :, k-1] + \
                                          np.sqrt(1-rho**2) * start[:, :, k]

                # Add correlation across rows
                for i in range(1, mydim[0]):
                    noise_scan[i, 1:, 1:] = rho * noise_scan[i-1, 1:, 1:] + \
                                            np.sqrt(1-rho**2) * \
                                            noise_scan[i, 1:, 1:]

                noise[:, :, :, scan] = noise_scan

    else:
        raise Exception('Unrecognized method {}'.format(method))

    return noise
