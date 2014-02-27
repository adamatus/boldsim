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
                 durations=10, effect_sizes=1, accuracy=1):
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

    onsets, durations, effect_sizes = _verify_design_params(onsets,
                                                            durations,
                                                            effect_sizes)

    if len(onsets) > 1:
        raise Exception('stimfunction only works for a single condition')

    # Warn if events go past end of time
    if np.max(onsets[0]+durations[0]) >= total_time:
        warn('Onsets/durations go past total_time. ' +
             'Some events will be truncated/missing')

    resampled_onsets = onsets[0]/accuracy
    resampled_durs = _to_ndarray(durations[0])/accuracy

    output_len = total_time/accuracy
    if not output_len.is_integer():
        warn('total_time is not evenly divisibly by accuracy, ' +
             'output will be slightly truncated and may not ' +
             'exactly match total_time')

    output_series = np.zeros(int(output_len))

    for onset, dur, effect in zip(resampled_onsets.astype(int),
                                  resampled_durs.astype(int),
                                  effect_sizes[0]):
        output_series[onset:(onset+dur)] = effect

    return output_series

def _slice_only_if_list(arg):
    """
    Return a copied slice if given a list, otherwise return item
    """
    if isinstance(arg, list):
        return arg[:]
    else:
        return arg


def _match_to_onsets(onsets, arr_to_fix):
    """
    Return properly validated and formatted array to match onsets
    """

    arr = _slice_only_if_list(arr_to_fix)

    nconds = len(onsets)
    if isinstance(arr, Number):
        # We only got a single number, so assume it is the dur for everything
        arr = [[arr] * len(onsets[x]) for x in range(nconds)]
    elif len(arr) == 1:
        # We got a list with a single number,
        # so assume it is the dur for everything
        arr = [arr * len(onsets[x]) for x in range(nconds)]
    else:
        # We got a possibly complex list, deal with it accordingly

        # Check if we are only dealing with a single list
        if nconds == 1:
            if len(arr) == len(onsets[0]):
                arr = [arr]
            else:
                raise Exception("Num of durs does not match num of onsets")

        if not len(arr) == nconds:
            raise Exception("Num of dur lists should match num of onsets")

        for cond, durs in enumerate(arr):
            if isinstance(durs, list):
                # If it's just 1 item, replicate it to match number on onsets
                if len(durs) == 1:
                    arr[cond] = durs * len(onsets[cond])

                # If it's more than 1 item, make sure it
                # matches number of onsets
                elif not len(durs) == len(onsets[cond]):
                    raise Exception("Num of durs doesn't match num of \
                                     onsets for cond {}".format(cond))
            else:
                arr[cond] = [durs] * len(onsets[cond])
    return arr

def _verify_design_params(onsets, durations, effect_sizes):
    """
    Return properly formatted copies of onsets, durations and effect_sizes
    """
    onsets_cp = _slice_only_if_list(onsets)
    durations_cp = _slice_only_if_list(durations)
    effect_sizes_cp = _slice_only_if_list(effect_sizes)

    # Check to see how if we got a list of onset lists, or just one list
    if isinstance(onsets_cp, list) and any(isinstance(x, list) for x in onsets_cp):
        # See if we have a list of lists, or just a list
        onsets_cp = [_to_ndarray(x) for x in onsets_cp]
    else:
        # We got a single duration, make into a list
        onsets_cp = [_to_ndarray(onsets_cp)]

    durations_cp = _match_to_onsets(onsets_cp, durations_cp)
    effect_sizes_cp = _match_to_onsets(onsets_cp, effect_sizes_cp)

    return (onsets_cp, durations_cp, effect_sizes_cp)

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

    for cond, (onset, dur, effect) in enumerate(zip(onsets,
                                                    durations,
                                                    effect_sizes)):
        stim_timeseries = stimfunction(total_time, onset, dur, effect, accuracy)

        if conv == 'none':
            design_out[cond, :] = stim_timeseries[sample_idx]

        if conv in ['gamma', 'double-gamma']:
            x = np.arange(0, total_time, accuracy)
            hrf = gamma if conv == 'gamma' else double_gamma
            out = np.convolve(stim_timeseries, hrf(x),
                              mode='full')[0:(len(stim_timeseries))]
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


    elif method == 'gaussRF':
        #FIXME Finish implementing this

        # Compute 3d covariance matrix of field
        s = np.diag([FWHM**2]*3)/(8*np.log(2))


        # We always do RFs in 3d, so if we got a 2-d space,
        # add a single 3rd dim
        dim_rf = mydim[:]
        if len(dim_rf) == 2:
            dim_rf.append(1)

        voxdim = len(dim_rf) # FIXME Won't this always be 3?

        # Make the size of the kernel odd
        if np.mod(FWHM, 2) == 0:
            FWHM += 1

        noise = np.zeros(dim_rf + [nscan])

        def sim_3d_grf(voxdim, sigma, ksize, dims):
            """Simulate 3d Gaussian Random Field """
            grf_noise = np.zeros(dims)
            return grf_noise

        for scan in range(nscan):
            noise[:, :, :, scan] = sim_3d_grf(voxdim, s, FWHM, dim_rf)

        if len(mydim) == 2:
            noise = noise.squeeze()

    else:
        raise Exception('Unrecognized method {}'.format(method))

    return noise

def simprepTemporal(total_time=100, onsets=range(0, 99, 20),
                    durations=10, effect_sizes=1, TR=2, accuracy=1,
                    conv='none'):
    """"
    Verify and package simulation parameters

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
    out = dict()
    out['onsets'] = onsets
    out['durations'] = durations
    out['effect_sizes'] = effect_sizes
    out['total_time'] = total_time
    out['TR'] = TR
    out['accuracy'] = accuracy
    out['hrf'] = conv

    return out
