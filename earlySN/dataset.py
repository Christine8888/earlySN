import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math
import sncosmo
import os
import ztfdr
from ztfdr import release
from astropy.cosmology import Planck18 as cosmo 
from astropy.table import Table, unique
import dustmaps.sfd
#dustmaps.sfd.fetch()
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
import scipy.optimize as opt 
from scipy import stats 
from earlySN import lightcurve
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

# OPEN TO-DO LIST
# Add in more functionality for different bands + custom dataset formats
# Add ability to change fit parameters & cut

def avg_and_error(data, errs):
    """
    Calculate weighted average and standard error.

    Parameters:
    -----------
    data: array-like, data to average
    errs: array-like, uncertainties on data
    """

    err_weights = 1 / errs ** 2
    avg = np.average(data, weights = err_weights)
    err = np.sqrt(1 / np.sum(err_weights))

    return avg, err

def mass_statistics(sn_names, masses):
    std = np.std(masses.loc[sn_names].dropna())[0]
    avg = np.average(masses.loc[sn_names].dropna())
    err = std * np.sqrt(1/masses.loc[sn_names].dropna().shape[0])
    return avg, err 

def flux_from_amp(row, mu = 'mu', sigma = 'sigma', amp = 'amp'):
    """ Calculate the peak flux of a Gaussian from mu, sigma, and amplitude.
    
    Parameters:
    ----------
    row: pandas DataFrame row containing lightcurve parameters
    mu: string, column name for mu
    sigma: string, column name for sigma
    amp: string, column name for amplitude
    """

    return stats.norm.pdf(row[mu], loc = row[mu], scale = row[sigma]) * row[amp]

def fit_mu(params, z, x_0, x_1, c, masses):
    """ Calculate mu based on Tripp equation and SALT3 parameters.
    
    Parameters:
    ----------
    params: length-6 float array (alpha, beta, gamma, M(0 < z <= 0.033), M(0.033 < z <= 0.067), M(0.067 < z <= 0.1))
    z: float array of redshifts, length N = # SN
    x_0: size-N array of floats containing x_0 values
    x_1: size-N array of floats containing x_1 values
    c: size-N array of floats containing c values
    masses: size-N array of floats containing host galaxy masses
    """
    
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    bigM = params[3:]

    m_shift = np.zeros(x_0.shape[0])
    if masses is not None:
        m_shift[masses > 10] = 1 # offset at 10 dex
    
    mu_base = -2.5 * np.log10(x_0) + 10.635 + (alpha * x_1) - (beta * c) + gamma*m_shift
    
    # granular, z-dependent M_B
    mu_base[(z > 0) & (z <= 0.033)] -= bigM[0]
    mu_base[(z > 0.033) & (z <= 0.067)] -= bigM[1]
    mu_base[(z > 0.067) & (z <= 0.1)] -= bigM[2]
    
    return mu_base

def cost_mu(params, z, x_0, x_1, c, errors, cov, masses):
    """ Cost function for mu, to be minimized.
    
    Parameters:
    ----------
    params: length-6 float array (alpha, beta, gamma, M(0 < z <= 0.033), M(0.033 < z <= 0.067), M(0.067 < z <= 0.1))
    z: float array of redshifts, length N = # SN
    x_0: size-N array of floats containing x_0 values
    x_1: size-N array of floats containing x_1 values
    c: size-N array of floats containing c values
    cov: size-N array of 4x4 covariance matrices
    masses: size-N array of floats containing host galaxy masses
    """
    
    expected = np.array(cosmo.distmod(z))
    fit = np.array(fit_mu(params[:6], z, x_0, x_1, c, masses))
    err = fit_sigma_mu2(params[:6], z, x_0, errors, cov, dint = params[6])
    n = z.shape[0]
    return np.sum(-(fit - expected) ** 2 / (2 * (err ** 2)) - np.log(np.sqrt(2 * math.pi * err ** 2))) * -1

def fit_sigma_mu2(params, z, x_0, errors, cov, pec = 250.0, dint = 0.1):
    """ Calculate uncertainty on mu based on covariance, peculiar velocities, and internal dispersion.
    
    Parameters:
    ----------
    params: length-6 float array (alpha, beta, gamma, M(0 < z <= 0.033), M(0.033 < z <= 0.067), M(0.067 < z <= 0.1))
    z: float array of redshifts, length N = # SN
    x_0: size-N array of floats containing x_0 values
    errors: Nx4 array of uncertainties on parameters t0, x0, x1, c 
    cov: size-N array of 4x4 covariance matrices
    pec: float (default = 250.0), average size (km/s) of peculiar velocity dispersion
    dint: float (deafult = 0.1), average scatter in intrinsic magnitude within supernova population
    """

    dmuz = pec/3e5
    dlens = 0.055 * z
    dmb = -2.5 * 0.434 * errors[:,1] / x_0 # mb = 2.5log10(x0) + constant
    dx1 = params[0] * errors[:,2]
    dc = params[1] * errors[:,3]
    dcx1 = 2 * params[0] * params[1] * cov[:,2,3]
    dmbx1 = 2 * params[0] * cov[:,1,2] * (-2.5 / (x_0 * np.log(10.0)))
    dmbc = 2 * params[1] * cov[:,1,3] * (-2.5 / (x_0 * np.log(10.0)))
    err = np.sqrt(dint ** 2 + dmuz ** 2 + dlens ** 2 + dmb ** 2 + dx1 ** 2 + dc ** 2 + dcx1 + dmbx1 + dmbc)
    
    return err

def table_errs(mu, sigma):
    """Print out LaTeX-formatted values with rounded uncertainty.
    
    Parameters:
    ----------
    mu: float, mean value
    sigma: float, standard deviation"""

    n_decimal = int(np.floor(np.log10(np.abs(sigma))) * -1)
    n_dig = int(np.floor(np.log10(np.abs(mu))) * -1)
    diff = int(n_decimal - n_dig) 
    
    if n_decimal > 3:
        return '${} \pm {} \\times 10^{{ {} }}$'.format(np.round(mu * 10**(n_dig), decimals=diff), np.round(sigma * 10**(n_dig), decimals = diff), -1 * n_dig)
    
    else:
        return '${} \pm {}$'.format(np.round(mu, n_decimal), np.round(sigma, n_decimal))

class Dataset(object):
    def __init__(self, base_path, bands, default = None, path_to_data = None, path_to_params = None, path_to_masses = None, path_to_pecs = None, index_col = None, name = ""):
        """ Constructor for the Dataset object, with option to use default built-in datasets or upload data from files. 
        To create a Dataset object, user must either use a default option or provide paths to all necessary files.
        
        Parameters
        ----------
        base_path: str, path to this package
        bands: List(str), list of bandpasses in dataset
        default: str, "yao" (for Yao et al. 2019) or "dhawan" (for Dhawan et al. 2022) or "burke" (for Burke et al. 2022b)
        path_to_data: str, system path to lightcurve data
        path_to_params: str, system path to lightcurve params
        path_to_masses: str, system path to host galaxy masses
        path_to_pecs: str, system path to peculiar velocities
        index_col: str or int, column to use as DataFrame index
        name: str, ID for Dataset """

        self.name = ""
        self.base_path = base_path
        self.bands = bands

        # Set dataset name
        if name != None:
            self.name = name
        elif default != None:
            self.name = default
        
        self.data = None
        self.params = None
        self.masses = None 
        self.pecs = None

        if default != None: # build dataset from default sample
            self.name = default
            
            if default == 'yao' or default == "burke":
                self.data = pd.read_csv(base_path + '/data/yao_data.csv', index_col = 'SN')
                self.params = pd.read_csv(base_path + '/data/yao_params.csv', index_col = 'SN')
                self.masses = pd.read_csv(base_path + '/data/yao_masses.csv', index_col = 0).rename(columns = {'0': 'mass'})
                self.pecs = pd.read_csv(base_path + '/data/yao_pvs.txt', delimiter = ' ', index_col = 0).rename(columns = {'0': 'v'})
            
            elif default == 'dhawan':
                self.data = pd.read_csv(base_path + '/data/dhawan_data.csv',)
                self.params = pd.read_csv(base_path + '/data/dhawan_params.csv')
                self.masses = pd.read_csv(base_path + '/data/dhawan_masses.csv', index_col = 0).rename(columns={'0': 'mass'})
                self.pecs = pd.read_csv(base_path + '/data/dhawan_pvs.txt', delimiter = ' ', index_col = 0).rename(columns = {'0': 'v'})
            
            elif default == "burke":
                # Load in default gold/bronze tier lists from Burke et al. 2022b paper, + Yao et al. 2019 gold non-detections
                print("Setting excess list for Burke et al. 2022b")
                self.gold = ['ZTF18aaxsioa', 'ZTF18abcflnz', 'ZTF18abssuxz', 
                        'ZTF18abxxssh', 'ZTF18aavrwhu', 'ZTF18abfhryc',]
                self.bronze = ['ZTF18aawjywv', 'ZTF18aaqcozd', 'ZTF18abdfazk',
                        'ZTF18abimsyv', 'ZTF18aazsabq']
                self.excess = self.gold + self.bronze

                self.gold_nd = ['ZTF18aavrzxp', 'ZTF18aazblzy', 'ZTF18abcysdx', 'ZTF18abetehf', 'ZTF18abxygvv']
                
                for i in self.sn_names:
                    if i not in self.gold and i not in self.bronze:
                        self.nd += [i]

        if path_to_data != None and path_to_params != None and index_col != None: # build dataset from path
            self.data = pd.read_csv(path_to_data, index_col = index_col)
            self.params = pd.read_csv(path_to_params, index_col = index_col)
            
            if path_to_masses != None:
                self.masses = pd.read_csv(path_to_masses, index_col = index_col)
            else:
                self.masses = None
                print('No PVs found!')
            
            if path_to_pecs != None:
                self.pecs = pd.read_csv(path_to_pecs, index_col = index_col)
            else:
                self.pecs = None
                print('No PVs found!')
        
        self.hubble = None
        self.sn_names = pd.Series(self.data.index.unique()) # will hold all the valid SN
        
        self.N = pd.DataFrame(index = self.sn_names, columns=['N'])
        
    def fit_salt(self, save_fig = None, save_path = None, verbose = False):
        """ Fit SALT3 parameters (z, t0, x0, x1, c) to lightcurves in the Dataset using SNCosmo. Also performs dust extinction modeling. 
        
        Parameters:
        -----------
        save_fig: str, optional (default = None), path to directory to save SALT3 figures. 
        save_path: str, optional (default = None), path to directory to save SALT3 parameters. If None, parameters will not be saved outside the instance
        verbose: bool (default = True), if True, print progress and parameters"""

        self.sn_names = self.sn_names  # CHANGE THIS BACK TO-DO
        self.hubble = pd.DataFrame(index = self.sn_names, columns = ['z', 't0', 'dt0', 'x0', 'dx0', 'x1', 'dx1', 'c', 'dc', 'fmax', 'cov'])
        
        
        for i in range(len(self.sn_names)):
            sn = self.sn_names[i]

            if verbose:
                print('Fitting SN {}/{}'.format(i,len(self.sn_names)))
                print(sn)
            
            self.hubble.loc[sn, 'z'] = self.params.loc[sn, 'z']

            # Milky Way dust modeling for extinction
            dust = sncosmo.F99Dust()
            model = sncosmo.Model(source='salt3', effects=[dust], effect_names=['mw'], effect_frames=['obs'])
            sc = SkyCoord(self.params.loc[sn]['ra'], self.params.loc[sn]['dec'], frame='icrs', unit='deg')
            ebb = SFDQuery()(sc)*0.86
            model.set(mwebv = ebb)
            model.set(z = self.hubble.loc[sn, 'z'])

            # SALT3 Fitting
            tdata = Table.from_pandas(self.data.loc[sn]) # needed for SNCosmo to be happy
            t0_cand = tdata['jd'][np.argsort(tdata['flux'])[-5:]] # pick t0 based off lowest-flux data points
            max_cand = t0_cand + 10
            min_cand = t0_cand - 10
            t0_bound = (min(min_cand), max(max_cand)) # set range

            result, fitted_model = sncosmo.fit_lc(
                tdata, model,
                [ 't0', 'x0', 'x1','c'],
                bounds = {'t0':t0_bound, 'x0':(10 *(-12), 1), 'x1':(-10, 10), 'c': (-1, 1)}, warn = False)
            if verbose: print(fitted_model.parameters)

            # Calculate and record maximum flux by bandpass
            zp_sn = np.mean(self.data.loc[sn]['zp'])
            max_fluxes = {}
            for band in self.bands:
                max_fluxes[band] = float(fitted_model.bandflux('ztf{}'.format(band),result['parameters'][np.array(result['param_names']) == 't0'],zp_sn,'ab'))
            self.hubble.loc[sn, 'fmax'] = [max_fluxes]
            
            # Make lightcurve figure
            if save_fig is not None:
                fig = sncosmo.plot_lc(tdata, model = fitted_model, errors = result.errors)
                plt.savefig(save_fig + "{}_{}_SALT3.pdf".format(self.name, sn), format='pdf', bbox_inches='tight')
                plt.close()
            
            self.hubble['cov'] = self.hubble['cov'].astype(object)
            self.hubble.loc[sn, 'cov'] = [result.covariance]
            self.hubble.loc[sn, ('t0', 'x0', 'x1', 'c')] = fitted_model.parameters[1:5] # do not use SALT3 z
            self.hubble.loc[sn, ('dt0', 'dx0', 'dx1', 'dc')] = result.errors['t0'], result.errors['x0'], result.errors['x1'], result.errors['c']
            

        if verbose: self.salt_stats()
        if save_path is not None:
            self.hubble.to_csv(save_path + '{}_salt3_params.csv'.format(self.name))

        return self.hubble
        
    def salt_stats(self, return_vals = False):
        """ Return statistics on calculated SALT3 parameters. Useful for sanity checks or science
        
        Parameters:
        -----------
        return_vals: bool, optional. If True, return the 3 x 2 numpy array of the mean & standard deviations of each parameter."""
        
        if len(self.hubble) <= 1:
            print('SALT3 fits not yet complete')
        
        params = ['x0', 'x1', 'c']
        for param in params:
            print('{}: {:0.3f} $\pm$ {:0.3f}'.format(param, self.hubble[param].mean(), self.hubble[param].std()))
        
        if return_vals:
            return np.array([[self.hubble[param].mean(), self.hubble[param].std()] for param in params])

    def spectral_filter(self, spectra = None, ok_sn = ['SN Ia'], verbose = False):
        """ Load and apply a spectral type filter to the dataset's supernova. Usually should keep only normal SN Ia, rejecting peculiar spectral types.
        
        Parameters:
        -----------
        spectra: pandas Series or str, optional. If Series, index should match SN names. If str, spectra should indicate the column in self.params that contains the spectra.
        ok_sn: List(str) of spectral class labels that are OK. Any other SN will be rejected
        verbose: bool, optional. If True, print out SN names that are rejected."""

        if verbose: print("Applying spectral filter.")
        # Load in spectra
        if self.name == 'yao' or self.name == "burke":
            ok_sn = ['normal   ', '91T-like ', '99aa-like']
            self.spectra = self.params['Subtype']
        
        elif self.name == 'dhawan':
            ok_sn = ['SN Ia', 'SN Ia?','SN Ia-norm', 'SN Ia 91T-like', 'SN Ia 91T', 'SN-91T', 'SNIa-99aa']
            spectra = pd.read_csv(self.base_path + '/data/dhawan_data.csv', index_col=0)
            self.spectra = spectra[4]
        
        elif type(spectra) == str: # input is the name of a column in params
            self.spectra = self.params[spectra]
        
        elif spectra is None:
            print("No spectra included!")
            return
        
        # Remove all supernovae not satisfying spectral class requirements
        iterate = self.sn_names.index
        drop_count = 0
        for i in iterate:
            sn = self.sn_names[i]
            if self.spectra.loc[sn] not in ok_sn:
                self.sn_names.drop(i, inplace=True)
                drop_count += 1
        
        if verbose: print("Dropped {} SN".format(drop_count))
        self.hubble = self.hubble.loc[self.sn_names]
        
    
    def pv_correction(self):
        """ Correct velocities and redshifts based on input peculiar velocities
        
        Parameters:
        -----------
        pecs: Pandas Series or str. If Series, index should match SN names. If str, pecs should indicate the column in self.params containing the peculiar velocities."""
        pecs = self.pecs
        if pecs is not None:
            if type(pecs) == str:
                pecs = self.params[pecs]
            
            for sn in self.sn_names:
                pv = pecs.loc[sn]['vpec']
                z_cmb = pecs.loc[sn]['zcmb']
                self.hubble.loc[sn, 'z'] = (1.0 + z_cmb) / (1.0 + (pv / 3e8)) - 1.0

    def reset_all_cuts(self):
        """ Reset all cuts on spectral type, redshift, etc. and restore SN index to that of the original dataset."""

        self.sn_names = pd.Series(self.data.index.unique())

    def param_cuts(self, z_max = 0.1, min_points = 3, dx1_max = 1.0, dt0_max = 1.0, dc_max = 0.3, x1_max = 3.0, c_max = 0.3, verbose = False):
        """ Remove SN from consideration (by removal from index self.sn_names). Default numbers are based off Dhawan et al. 2022 (https://arxiv.org/pdf/2110.07256)
        
        Parameters:
        -----------
        z_max: float (default = 0.1), maximum redshift of sample
        min_points: int (default = 3), minimum number of points within +/- 10 days of peak light
        dx1_max: float (default = 1.0), maximum error on x_1
        dt0_max: float (default = 1.0), maximum error on t_0
        dc_max: float (default = 0.3), maximum error on c
        x1_max: float (default = 3.0), maximum absolute value of x_1
        c_max: float (default = 0.3), maximum absolute value of c
        """

        iterate = self.sn_names.index # work backwards to avoid indexing issues
        for i in iterate:
            sn = self.sn_names.loc[i]
            sn_data = self.data.loc[sn]

            if self.hubble.loc[sn, 'z'] > z_max:
                self.sn_names.drop(i, inplace=True)
                print(sn, 'z')
                continue
            
            if len(sn_data[np.abs(sn_data['jd'] - self.hubble.loc[sn, 't0']) < 10]) < min_points:
                self.sn_names.drop(i, inplace=True)
                print(sn, 'jd')
                continue
            
            if self.hubble.loc[sn, 'dx1'] > dx1_max:
                self.sn_names.drop(i, inplace=True)
                print(sn, 'dx1')
                continue

            if self.hubble.loc[sn, 'dt0'] > dt0_max:
                self.sn_names.drop(i, inplace=True)
                print(sn, 'dt0')
                continue

            if self.hubble.loc[sn, 'dc'] > dc_max:
                self.sn_names.drop(i, inplace=True)
                print(sn, 'dc')
                continue

            if self.hubble.loc[sn, 'x1'] > x1_max or self.hubble.loc[sn, 'x1'] < -1 * x1_max:
                self.sn_names.drop(i, inplace=True)
                print(sn, 'x1')
                continue

            if self.hubble.loc[sn, 'c'] > c_max or self.hubble.loc[sn, 'c'] < -1 * c_max:
                self.sn_names.drop(i, inplace=True)
                print(sn, 'c')
                continue
            
        self.hubble = self.hubble.loc[self.sn_names]
        
    def fit_Tripp(self, verbose = False, mass_step = True):
        """ Fit Tripp equation with optional host-galaxy mass step, get distance modulus (mu) by redshift (z), and compare to Planck18 cosmology.
        
        Parameters:
        -----------
        verbose: bool (default = False), option to print out progress and fit results
        mass_step: bool (default = True), option to model host-galaxy mass step; if False, default range for parameter gamma is set to [-0.01, 0.01].
        """
        
        # Set up parameters for vectorized calculations
        z = self.hubble.loc[self.sn_names, 'z'].to_numpy(dtype=float)
        x_0 = self.hubble.loc[self.sn_names, 'x0'].to_numpy(dtype=float)
        x_1 = self.hubble.loc[self.sn_names, 'x1'].to_numpy(dtype=float)
        c = self.hubble.loc[self.sn_names, 'c'].to_numpy(dtype=float)
        e = self.hubble.loc[self.sn_names, ['dt0', 'dx0', 'dx1', 'dc']].to_numpy(dtype=float)
        cv = np.array([cov[0] for cov in self.hubble["cov"]]).astype(float)
        masses = self.masses.loc[self.sn_names, 'mass'].to_numpy(dtype=float)

        # Optimize initial fit
        if verbose: print('Fitting Hubble residuals')

        x0_guess = [0.14, 3.0, 0.0, 19.36, 19.36, 19.36, 0.1]

        # Include optional mass step; if no mass step, restrict bounds tightly
        if mass_step:
            gamma_bounds = ((0.0, 0.4), (2, 4), (-1, 1), (-20, -18), (-20, -18), (-20, -18), (0.0, 0.3))
            result = opt.minimize(cost_mu, x0 = x0_guess, args=(z, x_0, x_1, c, e, cv, masses))
        else: 
            gamma_bounds = ((0.0, 0.4), (2, 4), (-0.01, 0.01), (-20, -18), (-20, -18), (-20, -18), (0.0, 0.3))
            result = opt.minimize(cost_mu, x0 = x0_guess, bounds = gamma_bounds, args=(z, x_0, x_1, c, e, cv, masses))

        # Print output
        errs = np.sqrt(np.diagonal(result.hess_inv))

        param_names = ["$\alpha$", "$\beta$", "$\gamma$", "M_0", "M_1", "M_2", "$\sigma_i$"]
        if verbose: 
            result_string = ""
            for i in range(len(result.x)): # TO-DO: add in parameter names
                result_string += param_names[i]
                result_string += "= "
                result_string += str(result.x[i])
                result_string += " $\pm$ "
                result_string += str(errs[i])
                result_string += " $\pm$, "
            print('Tripp parameters: {}'.format(result_string))

        mu = fit_mu(result.x, z, x_0, x_1, c, masses)
        mu_errs = fit_sigma_mu2(result.x, z, x_0, e, cv)

        return result, mu, mu_errs

    def fit_hubble(self, save_path = None, verbose = False, mass_step = True, outlier_cut = 4.0):
        """ Fit for Hubble residuals, including sample cuts, Tripp modeling, and outlier rejection.
        
        Parameters:
        -----------
        save_path: str, optional (default = None), path to directory to save Hubble parameters and figures. If None, parameters will not be saved outside the instance
        verbose: bool (default = True); if True, print progress and parameters
        mass_step: bool (default = True), option to model host-galaxy mass step; if False, default range for parameter gamma is set to [-0.01, 0.01]
        outlier_cut: float (default = 5.0), minimum sigma difference from Planck18 distance modulus to label supernova as outlier
        """

        result, mu, mu_errs = self.fit_Tripp(verbose = False, mass_step = mass_step)

        # Calculate and report outliers
        z = self.hubble.loc[self.sn_names, 'z'].to_numpy(dtype=float)
        mu_resids = np.array(mu) - np.array(cosmo.distmod(z))
        mu_resids /= mu_errs
        
        
        outlier_cut = np.abs(mu_resids) > outlier_cut
        if verbose: print('Outliers: ', self.sn_names[list(self.sn_names[outlier_cut].index)])
        self.sn_names = self.sn_names[~outlier_cut]
        self.hubble = self.hubble.loc[self.sn_names]
        if verbose: print("Repeating Tripp fit without outliers")

        # Repeat analysis without outliers
        result, mu, mu_errs = self.fit_Tripp(verbose = verbose, mass_step = mass_step)
        
        z = self.hubble.loc[self.sn_names, 'z'].to_numpy(dtype=float)
        if save_path is not None:
            plt.errorbar(z, mu, yerr = mu_errs, fmt='o', markersize=3, label='ZTF18')#, yerr=mu_errs)
            plt.plot(np.sort(z), cosmo.distmod(np.sort(z)), label='Planck18')
            plt.xlabel('z')
            plt.ylabel('$\mu$')
            plt.legend()
            plt.savefig(save_path + '/{}_hubble_diagram.pdf'.format(self.name))

        # Save distance moduli (mu) and uncertainties (mu_errs)
        self.hubble.loc[self.sn_names, 'mu'] = mu
        self.hubble.loc[self.sn_names, 'dmu'] = mu_errs

        if save_path is not None:
            self.hubble.loc[self.sn_names].to_csv(save_path + '{}_hubble.csv'.format(self.name))
    
    def fit_single_sn(self, sn, cut = None):
        """ Helper function to search for early excess in a single supernova.
        
        Parameters:
        -----------
        sn: str, supernova to fit"""

        data = self.data.loc[sn]
        data = data[data['flux'] / data['flux_err'] > -2]
        params = self.hubble.loc[sn]
        
        fit = lightcurve.Lightcurve(sn, data, params, self.bands, self.name, save_fig = False, verbose = True)
        if cut is None:
            pl_params, gauss_params, classification, self.N.loc[sn, 'N'] = fit.excess_search(verbose = True)
        else:
            pl_result, early_data = fit.fit_model(cut = cut, model = 'powerlaw')
            pl_params = pl_result.x
            gauss_result, early_data = fit.fit_model(cut = cut, model = 'gauss')
            gauss_params = gauss_result.x
            early_data, outliers, t_range, binned, y = fit.analyze("gauss", early_data, gauss_result.x, cut)
            fit.plot(early_data, t_range, binned, y)

    def excess_search(self, save_path = None, save_fig = None, verbose = False):
        """ Search for lightcurves with early excess. To be applied after Hubble fitting and parameter cuts
        
        Parameters:
        ------------
        save_path: str, optional (default = None); path to directory to save lightcurve parameters. If None, parameters will not be saved outside the instance
        save_fig: str, optional (default = None); path to directory to save lightcurve figures.
        verbose: bool (default = False); if True, print progress and parameters
        """
        
       
        targets = self.sn_names
        self.gauss_params = pd.DataFrame(index = targets, columns=['t_exp', 'A_r', 'A_g', 'alpha_r', 'alpha_g', 'mu', 'sigma', 'C_r', 'C_g', 'B_r', 'B_g',
                                                                   'dt_exp', 'dA_r', 'dA_g', 'dalpha_r', 'dalpha_g', 'dmu', 'dsigma', 'dC_r', 'dC_g', 'dB_r', 'dB_g'])
        self.pl_params = pd.DataFrame(index = targets, columns=['t_exp', 'A_r', 'A_g', 'alpha_r', 'alpha_g', 'B_r', 'B_g',
                                                                'dt_exp', 'dA_r', 'dA_g', 'dalpha_r', 'dalpha_g', 'dB_r', 'dB_g'])

        self.gold = []
        self.gold_nd = []
        self.bronze = []
        
        print("Searching for excess in {} supernovae".format(targets.size))
        not_bronze = []
        if self.name == "yao":
            # manual control
            not_bronze = ['ZTF18aaunfqq', 'ZTF18aaxwjmp', 'ZTF18abbpeqo', 'ZTF18aazblzy', 'ZTF18abfhaji', 'ZTF18abjstcm', 'ZTF18abjvhec', 'ZTF18abwmuua', 'ZTF18abxygvv']

        # initialize way to save
        for sn in targets: 
            if verbose: print('Fitting {}'.format(sn))
            if sn not in not_bronze:
                data = self.data.loc[sn]
                data = data[data['flux'] / data['flux_err'] > -2] # remove large negative outliers
                params = self.hubble.loc[sn]
                
                fit = lightcurve.Lightcurve(sn, data, params, self.bands, self.name, save_fig = save_fig, verbose = verbose)
                pl_params, gauss_params, classification, self.N.loc[sn, 'N'] = fit.excess_search()

                # Update fit parameters
                if pl_params is not None:
                    self.pl_params.loc[sn] = pl_params
                if gauss_params is not None:
                    self.gauss_params.loc[sn] = gauss_params

                
                # add to the appropriate list
                if classification == "gold":
                    self.gold += [sn]
                elif classification == "bronze":
                    self.bronze += [sn]
                elif classification == "gold_nd":
                    self.gold_nd += [sn]
                else:
                    pass
        
        # write files
        if save_path is not None:
            with open(save_path + '{}_gold.txt'.format(self.name), 'w') as f:
                for line in self.gold:
                    f.write(f"{line}\n")
            
            with open(save_path + '{}_gold_nd.txt'.format(self.name), 'w') as f:
                for line in self.gold_nd:
                    f.write(f"{line}\n")

            with open(save_path + '{}_bronze.txt'.format(self.name), 'w') as f:
                for line in self.bronze:
                    f.write(f"{line}\n")
            
            if save_path is not None:
                self.gauss_params.to_csv(save_path + '{}_gauss_params.csv'.format(self.name))
                self.pl_params.to_csv(save_path + '{}_pl_params.csv'.format(self.name))
    
    
    def compare_excess(self, save_fig = None):
        """ Compare excess and non-excess populations across Hubble residuals.
        
        Parameters:
        -----------
        save_fig: str, optional (default = None), path to directory to save resulting figure."""

        # Compute Hubble residuals
        mu_errs = self.hubble['dmu']
        z = self.hubble['z']
        self.hubble['distmod'] = z.apply(cosmo.distmod)
        y = self.hubble.apply(lambda row: ((row['mu'] * u.mag) - (row['distmod'])).value, axis=1)
        fig, ax = plt.subplots(1,2, figsize=(10,5),width_ratios=[3,1],sharey=True)

        # Scatter points
        for sn in self.sn_names:
            if sn in self.gold:
                ax[0].errorbar(z[sn], y[sn], yerr = mu_errs[sn], markersize = 9, c = 'gold', zorder = 10, marker = 's', capsize = 5)
            elif sn in self.bronze:
                ax[0].errorbar(z[sn], y[sn], yerr = mu_errs[sn], fmt = 'o', markersize = 9, c = 'red', zorder = 10,  capsize = 5)
            else:
                ax[0].errorbar(z[sn], y[sn], yerr = mu_errs[sn], fmt = 'o', alpha = 0.3, markersize = 9, color = 'k', capsize=5)
        
        offset = np.average(y, weights = 1 / mu_errs ** 2) # subtract out mean value
        ax[0].axhline(y = offset, ls = '--', c = 'b')

        gold_avg, gold_err = avg_and_error(y[self.gold], mu_errs[self.gold])
        nd_avg, nd_err = avg_and_error(y[self.gold_nd], mu_errs[self.gold_nd])
        excess_avg, excess_err = avg_and_error(y[self.excess], mu_errs[self.excess])
        noexcess_avg, noexcess_err = avg_and_error(y[self.nd], mu_errs[self.nd])

        # Plot side histograms
        ax[1].hist(y[self.excess], alpha = 0.3, color = 'r', label = 'All Excess', orientation = 'horizontal')
        hist = ax[1].hist(y[self.nd], alpha = 0.3,color = 'k', label = 'All No Excess', orientation = 'horizontal')
        x = np.arange(0, np.max(hist[0] + 2))
        ax[1].fill_between(x, y1 = excess_avg - excess_err, y2 = excess_avg + excess_err, color = 'r', alpha = 0.3)
        ax[1].fill_between(x, y1 = noexcess_avg - noexcess_err, y2 = noexcess_avg + noexcess_err, color = 'k', alpha = 0.3)
        ax[1].axhline(y = excess_avg, ls = '--', c = 'r')
        ax[1].axhline(y = noexcess_avg, ls = '--', c = 'k')
        ax[1].legend()
        ax[1].set(xlim = (0, np.max(hist[0] + 1)))
        ax[1].set_xticks([])

        # Plot formatting and text
        pos = {'yao': 0.015, 'burke': 0.015, 'ztf': 0.01}
        heights = {'yao': [0.4, 0.6], 'burke': [0.4, 0.6], 'ztf': [-0.7, -0.45]}
        ylims = {'yao': [-0.5, 0.68], 'burke': [-0.5, 0.68], 'ztf': [-0.75, 0.8]}
        titles = {'yao': 'Yao et al. 2019', 'ztf': 'Dhawan et al. 2022', "burke": 'Burke et al. 2022b'}

        align = 'left'
        text_pos = pos[self.name]
        text_heights = np.linspace(heights[self.name][0], heights[self.name][1], 4)
        ax[0].set_ylim(ylims[self.name][0], ylims[self.name][1])
        ax[0].text(text_pos, text_heights[1], 'All Excess: ${:.3f} \pm {:.3f}$'.format(excess_avg, excess_err), horizontalalignment = align,)
        ax[0].text(text_pos, text_heights[2], 'All No Excess: ${:.3f} \pm {:.3f}$'.format(noexcess_avg, noexcess_err), horizontalalignment = align,)
        ax[0].text(text_pos, text_heights[0], 'Gold Excess: ${:.3f} \pm {:.3f}$'.format(gold_avg, gold_err), horizontalalignment = align,)
        ax[0].text(text_pos, text_heights[3], 'Gold No Excess: ${:.3f} \pm {:.3f}$'.format(nd_avg, nd_err), horizontalalignment = align,)
        ax[0].set_ylabel('$\mu_{SALT3} - \mu_{\Lambda_{CDM}}$ (mag)')
        ax[0].set_xlabel('z')
        plt.legend()
        print('Overall scatter: ', np.std(y))
        
        if self.name in titles.keys():
            fig.suptitle(titles[self.name])
        
        if save_fig is not None:
            plt.savefig(+ './{}_hubble.pdf'.format(self.name), bbox_inches = 'tight', pad_inches = 0)

    def set_N(self, sn, N):
        """ Set the best cut date for a given supernova.
        
        Parameters:
        -----------
        sn: str, supernova name
        N: int, best cut date for the supernova"""
        
        self.N.loc[sn] = N


    def compare_mass(self, save_fig = None):
        """ Compare host galaxy masses between excess and non-excess supernovae.
        
        Parameters:
        -----------
        save_fig: str, optional (default = None), path to directory to save resulting figure"""

        # Compute statistics
        gold_avg, gold_err = mass_statistics(self.gold, self.masses)
        nd_avg, nd_err = mass_statistics(self.gold_nd, self.masses)
        excess_avg, excess_err = mass_statistics(self.excess, self.masses)
        noexcess_avg, noexcess_err = mass_statistics(self.nd, self.masses)

        # Plot histograms
        cn = ["#68affc", "#304866"][0]
        ce = ["#68affc", "#304866"][1]
        plt.subplots(1, 1, figsize=(8,4))
        hist = plt.hist(self.masses.loc[self.nd], edgecolor = cn, alpha = 0.5, color = cn, label = 'All No Excess', bins = 10)
        plt.hist(self.masses.loc[self.excess], bins = hist[1], edgecolor = ce, alpha=0.5, color = ce, label = 'All Excess')
        plt.legend()

        center = {'yao': 8.2, 'burke': 8.2, 'ztf': 6.8}
        ranges = {'yao': [11, 15], 'burke': [11, 15], 'ztf': [22, 43]}
        titles = {'yao': 'Yao et al. 2019', 'ztf': 'Dhawan et al. 2022', "burke": 'Burke et al. 2022b'}

        # Figure labels and titles
        center = center[self.name]
        positions = np.linspace(ranges[self.name][0], ranges[self.name][1], 4)
        plt.text(center, positions[0], 'Gold No Excess: ${:.2f} \pm {:.2f}$'.format(nd_avg, nd_err), horizontalalignment='left',)
        plt.text(center, positions[1], 'No Excess: ${:.2f} \pm {:.2f}$'.format(noexcess_avg, noexcess_err), horizontalalignment='left',)
        plt.text(center, positions[2], 'All Excess: ${:.2f} \pm {:.2f}$'.format(excess_avg, excess_err), horizontalalignment='left',)
        plt.text(center, positions[3], 'Gold Excess: ${:.2f} \pm {:.2f}$'.format(gold_avg, gold_err), horizontalalignment='left',)
        plt.axvline(x = excess_avg, c = ce, ls = '--')
        plt.axvspan(xmin = excess_avg - excess_err, xmax = excess_avg + excess_err,color = ce, alpha = 0.5)
        plt.axvline(x = noexcess_avg, c = cn, ls='--')
        plt.axvspan(xmin = noexcess_avg - noexcess_err, xmax = noexcess_avg + noexcess_err, color = cn, alpha = 0.5)
        plt.legend()
        plt.xlabel('Host Galaxy Mass ($\log_{10} M_{*} / M_\odot$)')
        plt.yticks([])
        
        # Compute non-Gaussian statistics
        print('gold: ', 'median', self.masses.loc[self.gold].median()[0], 'MAD', stats.median_abs_deviation(self.masses.loc[self.gold])[0])
        print('all excess: ', 'median', self.masses.loc[self.excess].median()[0], 'MAD', stats.median_abs_deviation(self.masses.loc[self.excess])[0])
        print('all no excess: ', 'median', self.masses.loc[self.nd].median()[0], 'MAD', stats.median_abs_deviation(self.masses.loc[self.nd].dropna())[0])
        print('gold no excess: ', 'median', self.masses.loc[self.gold_nd].median()[0], 'MAD', stats.median_abs_deviation(self.masses.loc[self.gold_nd])[0])

        print('Levene :', stats.levene(self.masses.loc[self.excess]['mass'].dropna(), self.masses.loc[self.nd]['mass'].dropna()))
        print('Bartlett :', stats.bartlett(self.masses.loc[self.excess]['mass'].dropna(), self.masses.loc[self.nd]['mass'].dropna()))
        
        if self.name in titles.keys():
            plt.suptitle(titles[self.name])

        if save_fig is not None:
            plt.savefig(save_fig + './{}_mass.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)
            
    def compute_bump_properties(self):
        """ Compute properties of the bump in the light curve.
        
        Parameters
        ----------- 
        result: object output from scipy.minimize (created by self.fit_model)
        bands: list (string) of band names
        """

        # Update computation of bump properties if necessary
        for sn in self.excess:
            data = self.data.loc[sn]
            data = data[data['flux'] / data['flux_err'] > -2] # remove large negative outliers
            params = self.hubble.loc[sn]

            fit = lightcurve.Lightcurve(sn, data, params, self.bands, self.name, save_fig = False, verbose = False)
            result, early_data = fit.fit_model(cut = self.N.loc[sn, 'N'], model = 'gauss')
            J = result.jac
            cov = np.linalg.inv(J.T.dot(J))
            var = np.sqrt(np.diagonal(cov))

            self.gauss_params.loc[sn, ['t_exp', 'A_r', 'A_g', 'alpha_r', 'alpha_g', 'mu', 'sigma', 'C_r', 'C_g', 'B_r', 'B_g',
                                                                   'dt_exp', 'dA_r', 'dA_g', 'dalpha_r', 'dalpha_g', 'dmu', 'dsigma', 'dC_r', 'dC_g', 'dB_r', 'dB_g']] = list(result.x) + list(var)
            bics, outliers = fit.bic(models = ['gauss', 'powerlaw'], cut = self.N.loc[sn, 'N'])
            self.gauss_params.loc[sn, 'BIC'] = bics[1] - bics[0]

        # Compute flux ratios and colors
        gauss_params = self.gauss_params.loc[self.excess]
        bump_r = gauss_params.apply(flux_from_amp, axis = 1, args = ("mu", "sigma", "C_r"))
        bump_g = gauss_params.apply(flux_from_amp, axis = 1, args = ("mu", "sigma", "C_g"))
        err_r = gauss_params["C_r"] / gauss_params["dC_r"] 
        err_g = gauss_params["C_g"] / gauss_params["dC_g"]

        gauss_params["color"] = -2.5 * np.log10(bump_g / bump_r)
        dgr = ((err_r / bump_r) ** 2 + (err_g / bump_g) ** 2).pow(1./2.)
        gauss_params["dcolor"] = -2.5 * 0.434 * dgr / gauss_params["color"]    
        
        ten_r = gauss_params["A_r"] * 10 ** gauss_params["alpha_r"] + gauss_params["B_r"]
        ten_g = gauss_params["A_g"] * 10 ** gauss_params["alpha_g"] + gauss_params["B_g"]

        gauss_params["fr"] = bump_r / ten_r
        gauss_params["dfr"] = (gauss_params["dC_r"] / gauss_params["C_r"]) * gauss_params["fr"]
        gauss_params["fg"] = bump_g / ten_g
        gauss_params["dfg"] = (gauss_params["dC_g"] / gauss_params["C_g"]) * gauss_params["fg"]

        # TO-DO: figure out how not to overwrite everything (lol)
        self.gauss_params = gauss_params

    def analyze_bump_shapes(self, save_fig = None, add_uncertainty = False):
        """
        Analyze and plot bump shapes for all gold supernovae.

        Parameters:
        -----------
        save_fig: str, path to directory to save figure (default = None)
        add_uncertainty: bool, optional. If True, sample random curves from distribution.
        """
        fig, ax = plt.subplots(2,1, figsize=(6,5))
        
        for sn in self.gold:
            row = self.gauss_params.loc[sn]
            x = np.linspace(-3, 6, 200)
            
            # Sample 10 curves from the Gaussian distribution
            for i in range(10):
                mu = row['mu']
                sigma = row['sigma']
                
                if add_uncertainty:
                    mu += np.random.normal(0, 1) * row['dmu']
                    sigma += np.random.normal(0, 1) * row['dsigma']
                y = stats.norm.pdf(x, loc = mu, scale = sigma)
            
                y_g = y * row['fg'] / stats.norm.pdf(mu, loc = mu, scale = sigma,)
                ax[0].plot(x, y_g, c = 'g', alpha = 0.1, )
                y_r = y * row['fr'] / stats.norm.pdf(mu, loc = mu, scale = sigma,)
                ax[1].plot(x, y_r, c = 'r', alpha = 0.1,)
        
        # Plot curves
        ax[0].plot(x, y_g, c = 'g', alpha=1,  label = 'g')
        ax[1].plot(x, y_r, c = 'r', alpha=1,  label = 'r')
        ax[0].legend()
        ax[1].legend()
        
        fig.supylabel('$f_{bump}$ / $f_{10}$')
        ax[1].set_xlabel('JD since First Light')
            
        mu_avg, mu_err = avg_and_error(self.gauss_params['mu'], self.gauss_params['dmu'])
        print(r'$\mu$', mu_avg, mu_err)
        ax[0].axvspan(mu_avg - mu_err, mu_avg + mu_err, color='k', alpha=0.3)
        ax[1].axvspan(mu_avg - mu_err, mu_avg + mu_err, color='k', alpha=0.3)

        sig_avg, sig_err = avg_and_error(self.gauss_params['sigma'], self.gauss_params['dsigma'])
        print(r'$\sigma$', sig_avg, sig_err)

        if save_fig:
            plt.savefig(save_fig + './{}_curves.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)

    def analyze_bump_amps(self, c1 = 'r', c2 = 'g', save_fig = None):
        """
        Analyze and plot bump amplitudes in two bands.

        Parameters:
        -----------
        c1: str, band 1 (default = 'r')
        c2: str, band 2 (default = 'g')
        save_fig: str, path to directory to save figure (default = None)
        """
        
        plt.subplots(figsize=(6,5))
        gauss_params = self.gauss_params.loc[self.excess]

        # Compute bump colors and properties
        r_avg, r_err = avg_and_error(gauss_params['f{}'.format(c1)], gauss_params['df{}'.format(c1)])
        print(c1, ':', r_avg, r_err)
        g_avg, g_err = avg_and_error(gauss_params['f{}'.format(c2)], gauss_params['df{}'.format(c2)])
        print(c2, ':', g_avg, g_err)

        diff = g_avg - r_avg
        d_err = np.sqrt(r_err**2 + g_err**2)
        print('$f{} - f{} = {} \pm {}$'.format(c1, c2, diff, d_err))

        color_avg, color_err = avg_and_error(gauss_params['color'], gauss_params['dcolor'])
        print('$\log_{{10}}(G - R) = {} \pm {}$'.format(color_avg, color_err))

        # Plot bump amplitudes between bands
        plt.errorbar(gauss_params['f{}'.format(c1)], gauss_params['f{}'.format(c2)], yerr = gauss_params['df{}'.format(c2)], 
                    xerr = gauss_params['df{}'.format(c1)], fmt = 'o', c = 'k', markersize = 5, alpha = 0.8)
        plt.axhspan(g_avg - g_err, g_avg + g_err, alpha = 0.5, color = c2)
        plt.axvspan(r_avg - r_err, r_avg + r_err, hatch = 'X',facecolor = c1, edgecolor = 'k', alpha = 0.5)
        plt.xlim(-0.01, 0.24)
        plt.ylim(-0.01, 0.24)
        plt.xlabel('$f_{bump,r}$ / $f_{10,r}$')
        plt.ylabel('$f_{bump,g}$ / $f_{10,g}$')

        if save_fig is not None:
            plt.savefig(save_fig + './{}_bumps.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)

    def analyze_PL(self, save_fig = None, subset = "excess"):
        """
        Analyze and plot power-law slopes for all gold supernovae.

        Parameters:
        -----------
        save_fig: str, path to directory to save figure (default = None)
        subset: str, subset of supernovae to analyze (default = "excess")
        """
        if subset == "excess":
            pl_params = self.gauss_params.loc[self.excess].dropna()
        elif subset == "all":
            pl_params = self.gauss_params.dropna()
        
        pl_params = pl_params[np.logical_and(pl_params['dalpha_r'] < 1, pl_params['dalpha_g'] < 1)]
        plt.subplots(figsize=(6,5))

        # Compute power law slopes by color
        r_avg, r_err = avg_and_error(pl_params['alpha_r'], pl_params['dalpha_r'])
        print(r_avg, r_err)

        g_avg, g_err = avg_and_error(pl_params['alpha_g'], pl_params['dalpha_g'])
        print(g_avg, g_err)

        diff = g_avg - r_avg
        d_err = np.sqrt(r_err**2 + g_err**2)
        print(r'$\alpha_g - \alpha_r = {} \pm {}$'.format(diff, d_err))

        # Plot power law slopes
        plt.errorbar(pl_params['alpha_r'], pl_params['alpha_g'], yerr = pl_params['dalpha_g'], 
                    xerr = pl_params['dalpha_r'], fmt = 'o', c = 'k', markersize = 5)
       
        x = np.linspace(1.1, 2.75)
        plt.axhspan(g_avg - g_err, g_avg + g_err, alpha = 0.5, color = 'g', label = r'$\overline{\alpha}_g$ $(1\sigma)$')
        plt.axvspan(r_avg - r_err, r_avg + r_err, alpha = 0.5, hatch = 'X',facecolor = 'r', edgecolor = 'k', label = r'$\overline{\alpha}_r$ $(1\sigma)$')

        plt.plot(x,x,'--', c='k', alpha=0.5, label=r'$\alpha_g = \alpha_r$')
        plt.legend()
        plt.xlabel(r'$\alpha_r$')
        plt.ylabel(r'$\alpha_g$')

        if save_fig is not None:
            plt.savefig(save_fig + './{}_slopes.pdf'.format(self.name), bbox_inches = 'tight', pad_inches=0)
    
    def analyze_width_correlation(self, save_fig = None):
        """Analyze correlation between bump width and amplitude.
        
        Parameters:
        -----------
        save_fig: str, path to directory to save figure (default = None)"""

        plt.subplots(figsize=(6, 5))
        plt.scatter(self.gauss_params['sigma'], self.gauss_params['fr'], marker = 's', c = 'r', label = 'r', alpha = 0.5)
        plt.scatter(self.gauss_params['sigma'], self.gauss_params['fg'], marker = 's', c = 'g', label = 'g', alpha = 0.5)

        fit_r = np.polyfit(np.array(self.gauss_params['sigma'], dtype = float), np.array(self.gauss_params['fr'], dtype = float), 1)
        plt.plot(self.gauss_params['sigma'], np.poly1d(fit_r)(self.gauss_params['sigma']), alpha = 0.5, color = 'r')
        fit_g = np.polyfit(np.array(self.gauss_params['sigma'], dtype = float), np.array(self.gauss_params['fg'], dtype = float), 1)
        plt.plot(self.gauss_params['sigma'], np.poly1d(fit_g)(self.gauss_params['sigma']), alpha = 0.5, color = 'g')

        plt.xlabel('excess width ' + r'$\sigma$')
        plt.ylabel(r'$f_{bump} / f_{10}$')
        
        print('r', stats.pearsonr(self.gauss_params['sigma'], self.gauss_params['fr']))
        print('g', stats.pearsonr(self.gauss_params['sigma'], self.gauss_params['fg']))

        if save_fig is not None:
            plt.savefig(save_fig + './{}_width_correlation.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)

    def compare_fit_params(self, params = ['x1', 'c'], save_fig = None):
        """Compare SALT3 parameters between SN with and without excess.
        
        Parameters:
        -----------
        params: list (str) of parameters to compare (default = ['x1', 'c'])
        save_fig: str, path to directory to save figure (default = None)"""
        
        fig, ax = plt.subplots(1, len(params), figsize=(len(params) * 5,5))
        for i, param in enumerate(params):
            gold_avg, gold_err = avg_and_error(self.hubble.loc[self.gold, param], self.hubble.loc[self.gold, "d{}".format(param)])
            print('Gold {}: '.format(param), gold_avg, gold_err)

            excess_avg, excess_err = avg_and_error(self.hubble.loc[self.excess, param], self.hubble.loc[self.excess, "d{}".format(param)])
            print('Excess {}: '.format(param), excess_avg, excess_err)

            noexcess_avg, noexcess_err = avg_and_error(self.hubble.loc[self.nd, param], self.hubble.loc[self.nd, "d{}".format(param)])
            print('No Excess {}: '.format(param), noexcess_avg, noexcess_err)

            colors = ["#b2e2e2", "#66c2a4", "#238b45"]
            bins = ax[i].hist(self.hubble.loc[self.gold, param], alpha = 0.5, color = colors[2], label = 'Gold', bins = 10)
            ax[i].hist(self.hubble.loc[self.excess, param], alpha = 0.5, color = colors[1], label = 'Excess', bins = bins[1])
            ax[i].hist(self.hubble.loc[self.nd, param], alpha = 0.5, color = colors[0], label = 'No Excess', bins = bins[1])
            ax[i].legend()
            ax[i].set_xlabel(param)
            ax[i].set_yticks([])

        if save_fig is not None:
            plt.savefig(save_fig + '{}_SALT3_comparison.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)

    def load_from_saved(self, save_path):
        """Load computed parameters from a saved directory.
        
        Parameters:
        -----------
        save_path: str, path to directory containing saved parameters"""

        self.hubble = pd.read_csv(save_path + '{}_hubble.csv'.format(self.name), index_col='SN')
        self.sn_names = self.hubble.index.unique()
        print('Loaded Hubble diagram parameters')
        
        if os.path.exists(save_path + '{}_gold.txt'.format(self.name)):
            print('Loaded excess search results')
            with open(save_path + '{}_gold.txt'.format(self.name), 'r') as f:
                self.gold = [line.strip() for line in f.readlines()]
                
            with open(save_path + '{}_gold_nd.txt'.format(self.name), 'r') as f:
                self.gold_nd = [line.strip() for line in f.readlines()]

            with open(save_path + '{}_bronze.txt'.format(self.name), 'r') as f:
                self.bronze = [line.strip() for line in f.readlines()]

        if os.path.exists(save_path + '{}_gauss_params.csv'.format(self.name)):
            print('Loaded lightcurve fit parameters')
            self.gauss_params = pd.read_csv(save_path + '{}_gauss_params.csv'.format(self.name), index_col = 0)
            self.pl_params = pd.read_csv(save_path + '{}_pl_params.csv'.format(self.name), index_col = 0)
        

    def make_paper_table(self):
        """Create a LaTeX table of fit parameters, as seen in the paper."""

        for sn in self.excess:
            sn_row = self.hubble.loc[sn]
            gauss_row = self.gauss_params.loc[sn]
            mb = -2.5 * np.log10(sn_row['x0']) + 10.635
            dmb = 2.5 * 0.434 * sn_row['dx0'] / sn_row['x0']
            # name, z, t_exp, M_B, x_1, c, BIC, bump_r, bump_g, N
            print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(sn.split('18')[1], 
                                                                                np.round(sn_row['z'], 3), 
                                                                                np.round(sn_row['t0'] - 2400000.5, 1),
                                                                                table_errs(mb, dmb), 
                                                                                table_errs(sn_row['x1'], sn_row['dx1']),
                                                                                table_errs(sn_row['c'], sn_row['dc']), 
                                                                                gauss_row['BIC'], 
                                                                                table_errs(gauss_row['fr'], gauss_row['dfr']),
                                                                                table_errs(gauss_row['fg'], gauss_row['dfg']),
                                                                                self.N.loc[sn, 'N']))

    def end_to_end(self, verbose = False, save_path = None):
        """Compute Hubble fit and excess search, end to end.
        
        Parameters:
        -----------
        verbose: bool; if True, print out progress and fit results
        save_path: str, optional; path to directory to save Hubble parameters and figures. If None, parameters will not be saved outside the instance"""

        # Processing steps
        self.fit_salt(verbose = verbose, save_path = save_path)
        self.salt_stats()
        self.spectral_filter(verbose = verbose)
        self.pv_correction()
        self.param_cuts(verbose = verbose)
        self.fit_hubble(save_path = save_path, verbose = verbose) # includes Tripp fitting step

        # Light curve fitting
        self.excess_search(save_path = save_path, verbose = verbose)

        # Analysis steps
        self.compare_excess()
        self.compare_mass()