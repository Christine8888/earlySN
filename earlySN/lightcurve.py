import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math
from astropy.cosmology import Planck18 as cosmo 
import scipy.optimize as opt 
from scipy import stats 

def weighted_average(data):
    """ Compute weighted average of SN lightcurve data points.
    
    Parameters:
    -----------
    data: [N x 3] array of SN lightcurve data, where N is the number of data points; 0: MJD, 1: flux, 2: flux error"""

    # Set MJD range
    jdmin = np.min(data[:, 0]) - 0.2
    jdmax = np.max(data[:, 0]) + 0.2
    
    # Bin data by date
    hist, bins = np.histogram(data, bins = np.arange(jdmin, jdmax, 1))
    w_av = np.zeros((bins.shape[0], 3))
    bin_indices = np.digitize(data[:, 0], bins = bins)
    
    # Calculate weighted average by bin
    for i in np.unique(bin_indices):
        j = i - 1
        points = data[np.where(bin_indices == i)[0]]
        inv_variances = 1 / points[:,2] ** 2
        
        w_av[j, 0] = np.mean(points[:,0])
        w_av[j, 1] = np.average(points[:,1], weights = inv_variances)
        w_av[j, 2] = np.sqrt(1 / (np.sum(inv_variances)))
    
    return w_av[~np.all(w_av == 0, axis=1)]


def pl_rise_model(x, t_rise, amp, power, offset):
    """ Model for a single power-law lightcurve rise.
        
    Parameters:
    ----------- 
    x: array of input positions (floats, representing times of observation)
    t_rise: float, time of first explosion light; before t_rise, model returns offset + 0
    amp: float, scaling factor on light curve
    offset: float, offset factor on light curve
    ----------- 
    """
    vals = np.zeros(x.shape[0]) + offset
    vals[x >= t_rise] += amp *  (x[x >= t_rise] - t_rise) ** power
    
    return vals


def pl_model(params, data, bands):
    """ Calculates error between observed data and power-law model predictions.
    
    Parameters:
    ----------- 
    params: array of dictionaries containing parameters (by band) for power-law model
    data: dictionary of time [:, 0] and flux [:, 1] data, by band
    bands: array of bands
    """
    
    # Initialize all parameters
    t_rise = params[0]
    amps = {'r': params[1], 'g': params[2]}
    power = {'r': params[3], 'g': params[4]}
    offset = {'r': params[5], 'g': params[6]}
    
    # Count total number of data points
    total = 0
    for band in bands:
        total += data[band].shape[0]
    
    # Compute error by data point
    band_err = np.zeros(total)
    position = 0
    for band in bands:
        band_data = data[band]
        diff = (band_data[:,1] - pl_rise_model(band_data[:,0], t_rise, amps[band], power[band], offset[band]))
        diff /= band_data[:,2] # scale by uncertainty
        
        # move through uncertainty array
        nband = data[band].shape[0]
        band_err[position:position + nband] = diff
        position += nband
    
    return np.array(band_err)

def gauss_rise_model(x, t_rise, pl_amp, power, mu, sigma, gauss_amp, offset):
    """ Model for a single power-law + Gaussian component lightcurve rise.
        
    Parameters:
    ----------- 
    x: array of input positions (floats, representing dates of observation)
    t_rise: float (day), time of first power-law light
    pl_amp: float, scaling factor on power-law component
    mu: float, center (day) of Gaussian component
    sigma: float, standard deviation (days) of Gaussian component
    gauss_amp: float, scaling factor on Gaussian component
    offset: float, overall offset factor on light curve
    """
    vals = np.zeros(x.shape[0]) + offset
    vals[x>= t_rise] += pl_amp * (x[x >= t_rise] - t_rise) ** power
    vals += gauss_amp * stats.norm.pdf(x, loc = mu, scale = sigma)
    
    return vals

def gauss_model(params, data, bands):
    """ Calculates error between observed data and Gaussian + power-law model predictions.
    
    Parameters:
    ----------- 
    params: array of dictionaries containing parameters (by band) for power-law model
    data: dictionary of time [:, 0] and flux [:, 1] data, by band
    bands: array of bands
    """
    t_rise = params[0]
    amps = {'r': params[1], 'g': params[2]}
    power = {'r': params[3], 'g': params[4]}
    gdelta = params[5] # model time offset between power-law first light and Gaussian peak
    mu = gdelta + t_rise
    sigma = params[6]
    gauss_amps = {'r': params[7], 'g': params[8]}
    offset = {'r': params[9], 'g': params[10]}
     
    # Count total number of data points
    total = 0
    for band in bands:
        total += data[band].shape[0]
    
    # Compute error by data point
    band_err = np.zeros(total)
    position = 0
    for i in bands:
        band_dat = data[i]
        y = gauss_rise_model(band_dat[:,0], t_rise, amps[i], power[i], mu, sigma, gauss_amps[i], offset[i])
        diff = (band_dat[:,1] - y) / band_dat[:,2] # scale by flux uncertainty
        
        nband = data[i].shape[0]
        band_err[position:position + nband] = diff
        position += nband
    
    return np.array(band_err)

def identify_excess(diff, error, sigma = 3):
    """ Count outlying points above a given sigma threshold
    
    Parameters:
    ----------- 
    diff: length-N array of residual differences
    error: length-N array of uncertainties
    sigma (default = 3): threshold above which a point is considered an outlier
    """

    resid = diff / error
    outliers = resid[resid >= sigma]
    return outliers.shape[0]

def compute_bump_errs(result, bands, verbose = False):
    """ Utility function to compute excess flux value & uncertainty from Gaussian component.
        
    Parameters:
    ----------- 
    result: object output from scipy.minimize (created by self.fit_model)
    bands: list (string) of band names
    verbose: boolean; if verbose, will print out values
    ----------- 
    """

    n = len(bands)

    # calculate variances
    J = result.jac
    cov = np.linalg.inv(J.T.dot(J))
    var = np.sqrt(np.diagonal(cov))
    
    # compute flux at maximum of excess
    bumps = np.array([flux_from_amp(result.x[1 + n * 2], result.x[2 + n * 2], result.x[3 + 2 * n + i]) for i in range(len(bands))])
    bump_errs = np.array([result.x[3 + 2 * n + i] / var[3 + 2 * n + i] for i in range(len(bands))]) # compute significance (sigma)
    
    # compute flux at day 10
    tens = np.array([gauss_rise_model(np.array([result.x[0] + 10.0]), result.x[0], result.x[i + 1], result.x[1 + n + i], 
                                      result.x[1 + n * 2], result.x[2 + n * 2], result.x[3 + 2 * n + i], 
                                      result.x[3 + 3 * n + i])[0] for i in range(len(bands))])
    
    if verbose: print(bumps, tens, bumps/tens)
    
    return var, bumps, tens, bump_errs 

def tier(type, cuts, deltas, outliers, result, early_data, bands, best_cut, verbose):
    """ Determine whether or not the light-curve meets criteria for different tiers of early excess.
        
    Parameters:
    ----------- 
    type: string ("gold" or "bronze"), determines stringency of criteria
    cuts: length-N list, cut range values
    deltas: length-N list, BIC differences for each value in the cut range
    outliers: (N x 2) array, number of outlying points (by band) for each cut value
    result: output object from scipy.minimize (created by self.fit_model)
    early_data: lightcurve data, cut to specific dates
    bands: list (string) of band names
    verbose: boolean; if verbose, will print out values
    ----------- 
    """

    returns = {}
    if type == "gold":
        criteria = {"min_bump_significance": 3, "min_bump_ten": 0.02, "max_bump": 1, "min_2sigma": 2, "min_preexp": 2, "max_extreme": 0, "max_chi2": 3}
    elif type == "bronze":
        criteria = {"min_bump_significance": 1, "min_2sigma": 1, "min_preexp": 1, "max_extreme": 1, "max_chi2": 6, "min_bump_ten": 0.01, "max_bump": 1}

    cut_index = np.argmax(cuts == best_cut)
    outlier_range = outliers[cut_index : cut_index + 2, :]
    n = len(bands)

    # Check that enough samples have BIC significance
    if type == "gold":
        if deltas[deltas < -5].shape[0] < 2: 
            if verbose: print('No BIC significance')
            return False
        
         # Compute number of outliers
        if outlier_range.shape[0] <= 0:
            if verbose: print('Not enough outliers')
            return False
        
    elif type == "bronze":
        bic2 = deltas[deltas < 0].shape[0] >= 2
        out1 = outlier_range[outlier_range >= 1].shape[0] > 0
        out2 = outlier_range[outlier_range >= 2].shape[0] > 0
        if not ((bic2 and out1) or out2):
            if verbose: print('No real bump')
            return False
    
    # Make sure "bump" metrics are ment
    var, bumps, tens, bump_errs = compute_bump_errs(result, bands)
    if np.max(bump_errs) < criteria['min_bump_significance']:
        if verbose: print('Bump is insignificant')
        return False    
    if np.max(bumps/tens) < criteria['min_bump_ten']:
        if verbose: print('Bump is too small')
        return False
    if np.max(bumps/tens) > criteria['max_bump']:
        if verbose: print('Bad bump fit')
        return False

    # Calculate excess "bump" size w.r.t. 10-day flux
    dA = np.array([var[3 + 2 * n + i] / result.x[3 + 2 * n + i] for i in range(len(bands))])
    dsigma = var[2 + n * 2] / result.x[2 + n * 2]
    dbump = np.sqrt(dA ** 2 + 4 * (dsigma ** 2)) * (bumps / tens)
    returns['dbump'] = dbump
    
    # Compute data quality metrics
    binned = {}
    total = 0
    for band in bands:
        if type == "gold":
            early_data[band] = early_data[band][early_data[band][:, 0] > result.x[0] - 5] # data from <5 days pre-explosion to > cut days pre-peak
        
        band_data = early_data[band]
        
        # Reject samples with < 2 data points in a band
        binned[band] = weighted_average(band_data)
        if band_data.shape[0] < 2:
            return False
        
        # Reject samples without enough data within 2-sigma of Gaussian peak
        minus2 = result.x[0] + result.x[1 + n * 2] - 2 * result.x[2 + n * 2]
        plus2 = result.x[0] + result.x[1 + n * 2] + 2 * result.x[2 + n * 2]
        in_gauss = np.logical_and(binned[band][:, 0] >= minus2, binned[band][:, 0] <= plus2)

        if binned[band][in_gauss].shape[0] < criteria['min_2sigma']: 
            if verbose: print('No data within 2sigma of excess')
            return False
        
        # Reject samples without enough data before explosions
        pre_peak = binned[band][:,0] < result.x[0]
        if binned[band][pre_peak].shape[0] < criteria['min_preexp']:
            if verbose: print('Not enough pre explosion data.')
            return False
        
        total += binned[band].shape[0]
    
    # Reject samples with poor reduced chi2, extreme outliers
    chi = gauss_model(result.x, binned, bands)
    if chi[chi > 10].shape[0] > criteria['max_extreme']:
        if verbose: print('Extreme outlier in fit')
        return False
    
    r_chi2 = np.sum(chi ** 2)/(total - 1)
    if verbose: print(r_chi2)
    if r_chi2 > criteria["max_chi2"]:
        if verbose: print('Chi2 too high')
        return False
    
    # Passes all cuts
    return True, returns

def flux_from_amp(mu, sigma, amp):
    """ Compute the amplitude of the Gaussian peak in flux terms, given Gaussian parameters. """

    return stats.norm.pdf(mu, loc = mu, scale = sigma) * amp

def nd(cuts, deltas, outliers, result, early_data, bands, best_cut, verbose):
    """ Determine whether or not the light-curve meets criteria for gold-tier control/non-detection of excess.
        
    Parameters:
    ----------- 
    cuts: length-N list, cut range values
    deltas: length-N list, BIC differences for each value in the cut range
    outliers: (N x 2) array, number of outlying points (by band) for each cut value
    result: output object from scipy.minimize (created by self.fit_model)
    early_data: lightcurve data, cut to specific dates
    bands: list (string) of band names
    verbose: boolean; if verbose, will print out values
    ----------- 
    """

    # Bump size criterion
    var, bumps, tens, bump_errs = compute_bump_errs(result, bands)
    if np.max(bumps/tens) > 0.02:
        if verbose: print('Bump too large')
        return False
    if np.max(bumps/tens) > 1:
        if verbose: print('Bad bump fit')
        return False
    if np.max(bump_errs) > 2: #changed on 8.4
        if verbose: print('Bump is significant')
        return False
    

    # Check BIC criterion
    if deltas[deltas<5].shape[0] > 1:
        if verbose: print('BIC prefers gauss')
        return False
    
    # Check outliers criterion only for best fit
    cut_index = np.argmax(cuts == best_cut)
    outlier_range = outliers[cut_index : cut_index + 1, :]
    if outlier_range[outlier_range > 0].shape[0] > 1:
        if verbose: print('Too many outliers')
        return False

    # Data quality/availability criteria
    binned = {}
    total = 0
    avg_errors = np.zeros(len(bands))
    
    for i in bands:
        band_dat = early_data[i]
        if band_dat.shape[0] < 2:
            return False
        
        binned[i] = weighted_average(band_dat)
        in_early = np.logical_and(binned[i][:,0] > result.x[0], binned[i][:,0] < result.x[0] + 5)
        avg_errors[bands.index(i)] = np.mean(binned[i][in_early][:,2])
        if binned[i][in_early].shape[0] < 2:
            if verbose: print('No data in early region')
            return False
        
        pre_peak = binned[i][:,0] < result.x[0]
        if binned[i][pre_peak].shape[0] < 2:
            if verbose: print('Not enough pre explosion data.')
            return False        

        total += binned[i].shape[0]
    
    # Chi2/outleir checks
    chi = gauss_model(result.x, binned, bands)
    if chi[chi > 10].shape[0] >= 1: #changed on 8.4
        if verbose: print('Extreme outlier')
        return False
    
    r_chi2 = np.sum(chi ** 2)/(total - 1)
    if r_chi2 > 3:
        if verbose: print('Chi2 too large')
        return False
    
    return True

class Lightcurve(object):
    def __init__(self, sn_name, data, params, bands, source = 'yao', save_fig = None, verbose = False):
        """Initialize a lightcurve object.
        
        Parameters:
        ----------- 
        sn_name: string, supernova name
        data: DataFrame containing observation entries for target supernova
        params: DataFrame containing parameters for target supernova
        bands: list (string) of bands
        source: string, dataset name
        save_fig: string; if not None, folder to save final figures to
        verbose: boolean; if True, will print out helpful messages
        """

        self.sn_name = sn_name
        self.params = params
        self.save_fig = save_fig
        self.verbose = verbose
        self.data = data
        self.bands = bands
        self.pl_bounds = None
        self.source = source
        
        self.data_by_band = {}
        self.guess_t0 = self.params['t0'] - 30
        
        self.format_data()
    
    def format_data(self):
        """ Format data from DataFrame into list, for fitting and processing. Data will be stored in self.data_by_band as a [N x 3] array, 
        where N is the number of data points between peak and peak - 45 days."""
        
        for band in self.bands:
            self.data_by_band[band] = self.data[self.data['band'] =='ztf{}'.format(band)]

            # Format of list -- 0: MJD, 1: flux, 2: flux error
            self.data_by_band[band] = np.array((self.data_by_band[band]['jd'], self.data_by_band[band]['flux'], self.data_by_band[band]['flux_err'])).T
            
            # Make initial cuts
            peak_jd = self.params['t0']
            early_range = np.logical_and(self.data_by_band[band][:,0] < peak_jd, self.data_by_band[band][:,0] > peak_jd - 45)
            self.data_by_band[band] = self.data_by_band[band][early_range]

    def fit_model(self, cut, model):
        """ Fit model to lightcurve data, restricted to before [cut] days pre-peak.
        
        Parameters:
        ----------- 
        cut: integer, determines how many days pre-peak we exclude from the fit; useful to capture just the power-law rise
        model: string ("gauss" or "powerlaw") determining which model will be fit
        """

        guess_amp = {}
        
        # Only include data until up to [cut] days before peak light
        early_data = self.data_by_band.copy()
        
        for band in self.bands:
            pre_cut = early_data[band][:,0] < self.params['t0'] - cut
            early_data[band] = early_data[band][pre_cut]
            
            not_negative = early_data[band][:, 1]/early_data[band][:, 2] > -2
            early_data[band] = early_data[band][not_negative]

            if early_data[band].shape[0] >= 2: # if there are at least 2 data points in the band
                # Guess quadratic amplitude
                guess_amp[band] = (max(early_data[band][:,1]) - min(early_data[band][:,1])) # max flux difference
                guess_amp[band] /= (max(early_data[band][:,0]) - min(early_data[band][:,0])) ** 2 # max time difference
                guess_amp[band] = min(guess_amp[band], 500) # set max cutoff
            else:
                guess_amp[band] = 50

        # Avoid fitting light curves with insuffiient data
        total_size = np.sum([early_data[band].shape[0] for band in self.bands])
        if total_size <= 10:
            print('Not enough data points.')
            return None, None
        
        args = (early_data, self.bands)
        n = len(self.bands)
        
        # Allow flexibility in power-law slope choice
        pl_min = 1
        pl_max = 3
        if self.pl_bounds is not None:
            pl_min, pl_max = self.pl_bounds
        pl_guess = (pl_min + pl_max) / 2

        # Optimize parameters by model
        if model == 'powerlaw':
            x0 = [self.guess_t0] + [guess_amp[band] for band in self.bands] + [pl_guess] * n + [0] * n
            lbound = [self.guess_t0 - 10] + [0] * n + [pl_min] * n + [-100] * n
            ubound = [self.guess_t0 + 20] + [1000] * n + [pl_max] * n + [100] * n
            
            result = opt.least_squares(pl_model, x0 = x0, bounds = (lbound, ubound), args = args)
        
        elif model == 'gauss':
            # t0, PL amplitude, PL slope, mu + sigma, Gaussian amplitude, offset
            x0 = [self.guess_t0] + [guess_amp[band] for band in self.bands] + [pl_guess] * n + [2, 1] + [50] * n + [0] * n
            lbound = [self.guess_t0 - 10] + [0] * n + [pl_min] * n + [0, 0.5] + [0] * n + [-100] * n # sigma lower bound?
            ubound = [self.guess_t0 + 20] + [1000] * n + [pl_max] * n + [5, 4] + [500] * n + [100] * n # sigma lower bound?
            
            result = opt.least_squares(gauss_model, x0 = x0,  bounds = (lbound, ubound), args = args)
        
        else:
            print('Oops, not implemented yet!')
            return None, None
        
        return result, early_data
    
    def bic(self, models, cut):
        """ For each model, compute and compare the BICs at a given data cut.
        
        Parameters
        ----------- 
        models: list (string) of models, i.e. ["powerlaw", "gauss"]
        cut: int value, representing # days before maximum light at which to cut fitting
        """

        bics = np.zeros(len(models))
        bands = self.bands
        for i in range(len(models)):
            result, early_data = self.fit_model(cut = cut, model = models[i])
            n = len(bands)

            # Fit failed
            if result is None:
                return np.zeros(len(models)), [0, 0]   
            
            flux_errs = np.array([])
            total = 0
            
            for band in bands:
                total += early_data[band].shape[0]
                flux_errs = np.append(flux_errs, early_data[band][:, 2])
            
            # Determine value of k
            if models[i] == 'powerlaw':
                k = 1 + n * 3
                chi = pl_model(result.x, early_data, bands)
            elif models[i] == 'gauss':
                k = 3 + n * 4
                chi = gauss_model(result.x, early_data, bands)
            
            # Compute BIC
            loglike = -chi ** 2/2
            loglike += np.log(1 / np.sqrt(2 * math.pi * (flux_errs ** 2)))
            loglike = 2 * np.sum(loglike)
            
            bics[i] =  k * np.log(total) - loglike

            early_data, outliers, t_range, binned, (rise_y, binned_y) = self.analyze(model = models[i], early_data = early_data, fit_params = result.x, cut = cut)
        
        return bics, [outliers[bands[0]], outliers[bands[1]]]


    def bic_range(self, models, bands, plot = True):
        """ Compute and compare BIC between models over a range of cut values.
        
        Parameters
        ----------- 
        models: list (string) of models, i.e. ["powerlaw", "gauss"]
        bands: list (string) of band names
        plot: if True, will plot BIC difference vs. cut
        """

        cuts = np.arange(8, 14)
        if self.sn_name == 'ZTF18abssuxz':
            cuts = np.arange(1,10)
        
        # Compute outliers and BIC by band
        bics = np.zeros((len(cuts), len(models)))
        outliers = np.zeros((len(cuts), len(bands)))
        
        for i in range(len(cuts)):
            CUT = cuts[i]
            bics[i], outliers[i] = self.bic(models, CUT)
        
        deltas = bics[:,1] - bics[:,0]
        
        if plot:
            plt.plot(cuts, deltas, label='{} - {}'.format(models[1], models[0]))
            plt.xlabel('CUT')
            plt.ylabel('$\Delta$ BIC')
            plt.title('{}'.format(self.sn_name))
            plt.legend()
            plt.savefig('{}_bic.pdf'.format(self.sn_name))
            
        return cuts, deltas, outliers

    def analyze(self, model, early_data, fit_params, cut):
        """ Analyze fit and count outliers.
        
        Parameters
        ----------- 
        model: string ("gauss" or "powerlaw") representing the model that was fit for
        early_data: [N x 3] array representing SN lightcurve data cut within specific range
        fit_params: array of parameter values, likely from result.x
        """

        binned = {}
        binned_y = {}
        rise_y = {}
        outliers = {}
        t_range = {}
        n = len(self.bands)
        for band in self.bands: outliers[band] = 0

        for band in self.bands:
            i = self.bands.index(band)
            early_data[band][:,0] -= fit_params[0] # plot in terms of JD after initial rise
            early_data[band] = early_data[band][early_data[band][:,0] >= -10] # take only data close to initial rise
            
            if early_data[band].shape[0] >= 2:
                # Bin data by date and set date range
                binned[band] = weighted_average(early_data[band])
                t_range[band] = np.linspace(np.min(early_data[band][:,0]), np.max(early_data[band][:,0]))
                
                if model == 'powerlaw':
                    # [t0] + [n x amps] + [n x slopes] + [n x offsets]
                    t_rise = 0
                    amp = fit_params[1 + i]
                    slope = fit_params[1 + n + i]
                    offset = fit_params[1 + n * 2 + i]

                    rise_y[band] = [pl_rise_model(t_range[band], t_rise, amp, slope, offset)] # model prediction for flux given original data
                    binned_y[band] = [pl_rise_model(binned[band][:,0], t_rise, amp, slope, offset)] # model prediction for flux given binned data
                    resid_diff = binned[band][:,1] - binned_y[band][0] # difference between binned data and model prediction
                    outliers[band] = identify_excess(resid_diff, binned[band][:,2])

                elif model == 'gauss':
                    # [t0] + [n x amps] + [n x slopes] + [mu, sigma] + [n x amps] + [n x offsets]
                    t_rise = 0
                    amp = fit_params[1 + i]
                    slope = fit_params[1 + n + i]
                    mu = fit_params[1 + n * 2] # - result.x[0]
                    sigma = fit_params[2 + n * 2]
                    gauss_amp = fit_params[3 + 2 * n + i]
                    offset = fit_params[3 + 3 * n + i]

                    rise_y[band] = [gauss_rise_model(t_range[band], t_rise, amp, slope, mu, sigma, 0, offset),
                            gauss_rise_model(t_range[band], t_rise, 0, slope, mu, sigma, gauss_amp, offset),
                            gauss_rise_model(t_range[band], t_rise, amp, slope, mu, sigma, gauss_amp, offset)]

                    binned_y[band] = [gauss_rise_model(binned[band][:,0], t_rise, amp, slope, mu, sigma, gauss_amp, offset),
                                gauss_rise_model(binned[band][:,0], t_rise, amp, slope, mu, sigma, 0, offset)]
                    resid_diff = binned_y[band][0] - binned_y[band][1]
                    outliers[band] = identify_excess(resid_diff, binned[band][:,2])
                
                else:
                    print('Oops, not yet implemented.')
            
        if self.verbose: print(cut, outliers) # TO-DO: TURN OFF?  
        
        return early_data, outliers, t_range, binned, (rise_y, binned_y)
    
    def plot(self, early_data, t_range, binned, y):
        """ Plot lightcurve and model fit
        
        Parameters
        ----------- 
        early_data: [N x 3] array of SN lightcurve data cut within specific range
        t_range: range of (binned) times to plot against
        binned: weighted average of early_data, binned in time
        y: tuple containing (rise_y, binned_y) -- model prediction for t_range and at each time bin, respectively
        """

        rise_y, binned_y = y
        fig, ax = plt.subplots(len(self.bands), 2, figsize=(4 * len(self.bands), 8))

        
        for band in self.bands:
            i = self.bands.index(band)
            ax[i,0].errorbar(early_data[band][:,0], early_data[band][:,1], yerr = early_data[band][:,2], 
                                    fmt='o',color = band, markersize = 3) 
                    
            styles = ['--', '-.', '-']
            labels = ['PL', 'Gaussian', 'Total']

            for j in range(len(rise_y[band])):
                
                ax[i,0].plot(t_range[band], rise_y[band][j], label='{} fit'.format(labels[j]),color=band, ls=styles[j])

            ax[i,0].set_xlabel('JD since First Light')
            ax[i,0].set_ylabel('Flux')
            ax[i,0].legend()

            markers = ['o', 's']
            labels = ['all', 'just PL']
            alphas = [1, 0.3]

            for j in range(len(binned_y[band])): # plot binned points
                ax[i,1].errorbar(binned[band][:,0], (binned[band][:,1] - binned_y[band][j])/binned[band][:,2], 
                            fmt = 'o', alpha = alphas[j], marker = markers[j], color = band, label = '{}'.format(labels[j]), markersize = 6)


            ax[i,1].axhline(0, ls='--',c='k')

            ax[i,1].set_xlabel('JD since First Light')
            ax[i,1].set_ylabel('$\Delta$')
            fig.tight_layout()
            
            if self.save_fig:
                plt.savefig(self.save_fig + '{}_rise.pdf'.format(self.sn_name))

    def excess_search(self, pl_bounds = None, not_bronze = [], default_cut = 10, best_cut = None, verbose = False):
        """ Search for an early excess in the light curve.
        
        Parameters
        ----------- 
        pl_bounds: if not None, list [lower_bound, upper_bound] of floats representing PL slope bounds to enforce
        not_bronze: optional; list of manually vetted SN that should not be classified as bronze
        """

        self.pl_bounds = pl_bounds
        pl_params = None
        gauss_params = None

        result, early_data = self.fit_model(cut = default_cut, model = "gauss")

        if result is None:
            if self.verbose: print('Fit failed.')
            return None, None, None, None
            
        cuts, deltas, outliers = self.bic_range(['powerlaw', 'gauss'], self.bands, plot = False)
        if self.verbose: print(list(zip(cuts, deltas)))
        reasonable = deltas > -250

        if np.sum(reasonable) < 2:
            if self.verbose: print('Unreasonable.')
            return None, None, None, None      
        
        # Compute parameters for power law with default (early) cut
        result, early_data = self.fit_model(cut = 10, model = 'powerlaw')
        if result is not None:
            try:
                J = result.jac
                cov = np.linalg.inv(J.T.dot(J))
                var = np.sqrt(np.diagonal(cov))
            except:
                var = np.zeros(len(result.x))
            
            pl_params = list(result.x) + list(var)

        # Compute Gauss+PL parameters for best fit
        if best_cut == None:
            best_cut = cuts[reasonable][np.argsort(deltas[reasonable])[0]]
        
        if self.verbose: print('Best Cut: ', best_cut)   
        result, early_data = self.fit_model(cut = best_cut, model = 'gauss')
        
        # check that Jacobian is invertible
        ok_jac = True
        try:
            J = result.jac
            cov = np.linalg.inv(J.T.dot(J))
            var = np.sqrt(np.diagonal(cov))
        except:
            ok_jac = False
            var = np.zeros(len(result.x))
        
        sn_tier = "none"

        # Run tests to classify SN excess
        if result is not None and ok_jac:
            if self.verbose: print('Starting tests')
            
            gauss_params = list(result.x) + list(var)
            
            if tier("gold", cuts, deltas, outliers, result, early_data, self.bands, best_cut, self.verbose):
                early_data, outliers, t_range, binned, y = self.analyze("gauss", early_data, result.x, best_cut)
                self.plot(early_data, t_range, binned, y)
                sn_tier = "gold"
                if verbose: print("gold")
            
            elif nd(cuts, deltas, outliers, result, early_data, self.bands, best_cut, self.verbose):
                sn_tier = "gold_nd"
                if verbose: print("gold_nd")
            
            elif self.sn_name not in not_bronze and tier("bronze", cuts, deltas, outliers, result, early_data, self.bands, best_cut, self.verbose):
                sn_tier = "bronze"
                if verbose: print("bronze")
        
            return pl_params, gauss_params, sn_tier, best_cut
        
        return None, None, None, None