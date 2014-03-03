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
                 durations=10, effect_sizes=1, accuracy=1, verify_params=True):
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

    if verify_params:
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
    if isinstance(onsets_cp, list) and \
       any(isinstance(x, list) for x in onsets_cp):
        # See if we have a list of lists, or just a list
        onsets_cp = [_to_ndarray(x) for x in onsets_cp]
    else:
        # We got a single duration, make into a list
        onsets_cp = [_to_ndarray(onsets_cp)]

    durations_cp = _match_to_onsets(onsets_cp, durations_cp)
    effect_sizes_cp = _match_to_onsets(onsets_cp, effect_sizes_cp)

    return (onsets_cp, durations_cp, effect_sizes_cp)

def _verify_spatial_params(regions, coord, radius, form, fading):
    """
    Return properly formatted copies of coord, radius, form, fading
    """
    coord_cp = _slice_only_if_list(coord)
    radius_cp = _slice_only_if_list(radius)
    form_cp = _slice_only_if_list(form)
    fading_cp = _slice_only_if_list(fading)

    if not isinstance(coord_cp, list):
        raise Exception("Argument coord should be a list of lists")

    # If we got single list (rather than a nested list), go ahead and nest
    if np.all([isinstance(item, Number) for item in coord_cp]):
        coord_cp = [coord_cp]

    coord_cp = _match_to_regions(regions, coord_cp, list)
    radius_cp = _match_to_regions(regions, radius_cp, Number)
    form_cp = _match_to_regions(regions, form_cp, str)
    fading_cp = _match_to_regions(regions, fading_cp, Number)

    # Verify that fading values are within 0 and 1
    if not np.all([(fade >= 0) & (fade <= 1) for fade in fading_cp]):
        raise Exception("Fading values must be between 0 and 1, inclusive")

    if not np.all([form_item in ['cube', 'sphere', 'manual'] \
                    for form_item in form_cp]):
        raise Exception("Unknown spatial method specified: {}".format(form_cp))

    if not np.all([len(item) == len(coord_cp[0]) for item in coord_cp]):
        raise Exception("All coordinates must be of the same dimensions")

    return (coord_cp, radius_cp, form_cp, fading_cp)

def specifydesign(total_time=100, onsets=range(0, 99, 20),
                 durations=10, effect_sizes=1, TR=2, accuracy=1,
                 conv='none', verify_params=True):
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

    if verify_params:
        onsets, durations, effect_sizes = _verify_design_params(onsets,
                                                            durations,
                                                            effect_sizes)

    design_out = np.zeros((len(onsets), total_time/TR))
    sample_idx = np.round(np.arange(0, total_time/accuracy, TR/accuracy))
    sample_idx = np.asarray(sample_idx, dtype=np.int)

    for cond, (onset, dur, effect) in enumerate(zip(onsets,
                                                    durations,
                                                    effect_sizes)):
        stim_timeseries = stimfunction(total_time, [onset], [dur], [effect],
                                       accuracy, verify_params=False)

        if conv == 'none':
            design_out[cond, :] = stim_timeseries[sample_idx]

        if conv in ['gamma', 'double-gamma']:
            x = np.arange(0, total_time, accuracy)
            hrf = gamma if conv == 'gamma' else double_gamma
            out = np.convolve(stim_timeseries, hrf(x),
                              mode='full')[0:(len(stim_timeseries))]
            design_out[cond, :] = out[sample_idx]

    return design_out

def specifyregion(coord=None, radius=1, form='cube', fading=0, dim=None):
    """
    Generate an image with activation regions for specified dimensions

    Args:
        coords (list/list of lists): List of coordinates specify center of
            regions
        radius (int/float/list): If form is 'cube' or 'sphere', the distance
            between the center and edge. If form is 'manual', the number of
            voxels in the region
        form (string/list): Shape of regions: 'cube', 'sphere', 'manual'
        fading (float): Value between 0 and 1 specifing decay rate of signal
            from the center outward. 0 is none, 1 is most rapid
        dim (list): XYZ Dimensions of output

    Returns:
        A ndarray with the activation image

    Raises:
        Exception
    """

    if dim is None:
        dim = [9, 9]
    mydim = _handle_dim(dim)

    if not len(mydim) in [2, 3]:
        raise Exception('specifyregion only supports 2d or 3d regions')

    if coord is None:
        coord = [[4, 4]]
    regions = len(coord)

    coord, radius, form, fading = _verify_spatial_params(regions, coord, \
                                                    radius, form, fading)

    if not np.all([len(item) == len(dim) for item in coord]):
        raise Exception("Coordinates don't match image dimensions")

    out = np.zeros(mydim)
    for xyz, rad, shape, fade in zip(coord, radius, form, fading):
        if len(mydim) == 2:
            if shape == 'manual':
                out[xyz[0], xyz[1]] = 1
            else:
                irange = np.arange(xyz[0]-rad, xyz[0]+1+rad)
                for i in irange[(irange > -1) & (irange < mydim[0])]:
                    jrange = np.arange(xyz[1]-rad, xyz[1]+1+rad)
                    for j in jrange[(jrange > -1) & (jrange < mydim[1])]:
                        if (shape == 'cube') or ((shape == 'sphere') and \
                                (((i-xyz[0])**2 + (j-xyz[1])**2) <= (rad)**2)):
                            if fade == 0:
                                out[i, j] = 1
                            else:
                                out[i, j] = (2 * np.exp(-((i-xyz[0])**2 + \
                                                          (j-xyz[1])**2) * \
                                                          fade) + 2)/4
        else: # 3d version
            if shape == 'manual':
                out[xyz[0], xyz[1], xyz[2]] = 1
            else:
                irange = np.arange(xyz[0]-rad, xyz[0]+1+rad)
                for i in irange[(irange > -1) & (irange < mydim[0])]:
                    jrange = np.arange(xyz[1]-rad, xyz[1]+1+rad)
                    for j in jrange[(jrange > -1) & (jrange < mydim[1])]:
                        krange = np.arange(xyz[2]-rad, xyz[2]+1+rad)
                        for k in krange[(krange > -1) & (krange < mydim[2])]:
                            if (shape == 'cube') or ((shape == 'sphere') and \
                                    (((i-xyz[0])**2 + \
                                      (j-xyz[1])**2 + \
                                      (k-xyz[2])**2) <= (rad)**2)):
                                if fade == 0:
                                    out[i, j, k] = 1
                                else:
                                    out[i, j, k] = (3*np.exp(-((i-xyz[0])**2+ \
                                                               (j-xyz[1])**2+ \
                                                               (k-xyz[2])**2) \
                                                               *fade)+3)/6

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
    Verify and package simulation temporal parameters

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
        A dict with verified temporal parameters

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

def _match_to_regions(nreg, item_list, expected_type):
    """
    Return properly validated and formatted list to match number of regions
    """
    if not isinstance(item_list, list):
        item_list = [item_list]

    if not len(item_list) == nreg:
        if len(item_list) == 1:
            # Replicate single item list so it matches number of regions
            item_list *= nreg
        else:
            raise Exception("Argument coord should be a list of lists")

    if not np.all([isinstance(item, expected_type) for item in item_list]):
        raise Exception("Argument of unexpected type included in list")

    return item_list

def simprepSpatial(regions=1, coord=None, radius=1, form='cube', fading=0):
    """"
    Verify and package simulation spatial parameters

    Args:
        regions (int): Number of activated regions
        coords (list/list of lists): List of coordinates specify center of
            regions
        radius (int/float/list): If form is 'cube' or 'sphere', the distance
            between the center and edge. If form is 'manual', the number of
            voxels in the region
        form (string/list): Shape of regions: 'cube', 'sphere', 'manual'
        fading (float): Value between 0 and 1 specifing decay rate of signal
            from the center outward. 0 is none, 1 is most rapid

    Returns:
        A list of dicts with verified spatial parameters

    Raises:
        Exception
    """
    if coord is None:
        coord = [0, 0]

    if (not isinstance(regions, Number)) or (regions != int(regions)):
        raise Exception("Argument regions should be an integer")
    regions = int(regions)

    coord, radius, form, fading = _verify_spatial_params(regions, coord, \
                                                    radius, form, fading)

    regions_out = []
    for region in range(regions):
        out = dict()
        out['name'] = 'Region {}'.format(region)
        out['coord'] = coord[region]
        out['radius'] = radius[region]
        out['form'] = form[region]
        out['fading'] = fading[region]
        regions_out.append(out)

    return regions_out

def simTSfmri(design=None, base=10, SNR=2, noise='mixture', noise_dist='gaussian',
              weights=None, ar_coef=0.2, freq_low=128, freq_heart=1.17,
              freq_resp=0.2):
    """
    Generate a simulated fMRI timeseries
    """

    if design is None:
        raise Exception('Must provide design from simprepTemporal to simTSfmri')

    d_mat = specifydesign(total_time=design['total_time'],
                          onsets=design['onsets'],
                          durations=design['durations'],
                          effect_sizes=design['effect_sizes'], TR=design['TR'],
                          accuracy=design['accuracy'], conv=design['hrf'],
                          verify_params=False)
    bold = base + np.apply_along_axis(sum, axis=0, arr=d_mat)
    sigma = np.mean(bold)/SNR
    TR = design['TR']
    nscan = design['total_time']/TR

    # Setup weights based on the specific type of noise desired
    if noise == 'none':
        return bold
    elif noise == 'white':
        weights = [1, 0, 0, 0, 0]
    elif noise == 'temporal':
        weights = [0, 1, 0, 0, 0]
    elif noise == 'low-freq':
        weights = [0, 0, 1, 0, 0]
    elif noise == 'phys':
        weights = [0, 0, 0, 1, 0]
    elif noise == 'task-related':
        weights = [0, 0, 0, 0, 1]
    elif noise == 'mixture':
        if weights is None:
            weights = [0.3, 0.3, 0.01, 0.09, 0.3]
        if len(weights) != 5:
            raise Exception('Weights vector should have 5 elements')
        if np.sum(weights) != 1:
            raise Exception('Weights vector should sum to 1')
    else:
        raise Exception('Unknown noise setting: {}'.format(noise))
    weights = np.asarray(weights)

    # Setup output matrix to hold all possible noises,
    # but we'll only generate ones that are to be used
    noise_ts = np.zeros((5, nscan))

    if weights[0] != 0:
        noise_ts[0, :] = system_noise(nscan=nscan, sigma=sigma,
                                      noise_dist=noise_dist, dim=1).squeeze()
    if weights[1] != 0:
        noise_ts[1, :] = temporalnoise(nscan=nscan, sigma=sigma,
                                       ar_coef=ar_coef, dim=1)
    if weights[2] != 0:
        noise_ts[2, :] = lowfreqdrift(nscan=nscan, freq=freq_low,
                                            TR=TR, dim=1)
    if weights[3] != 0:
        noise_ts[3, :] = physnoise(nscan=nscan, sigma=sigma,
                                   freq_heart=freq_heart,
                                   freq_respiration=freq_resp, TR=TR, dim=1)
    if weights[4] != 0:
        noise_ts[4, :] = tasknoise(design=d_mat, sigma=sigma,
                                   noise_dist=noise_dist, dim=1)

    # Apply weights to noise types, sum and scale
    noise_ts = noise_ts * weights.reshape((5, 1))
    noise_ts = np.apply_along_axis(sum, 0, noise_ts)
    noise_ts /= np.sqrt(np.sum(np.power(weights, 2)))

    return bold + noise_ts - np.mean(noise_ts)
