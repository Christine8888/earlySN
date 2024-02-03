import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math
import sncosmo
import ztfdr
from ztfdr import release
from astropy.cosmology import Planck18 as cosmo 
from astropy.table import Table, unique
import dustmaps.sfd
dustmaps.sfd.fetch()
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
import scipy.optimize as opt 
from scipy import stats 

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
    mu_base[(z>0) & (z <= 0.033)] -= bigM[0]
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
    err = fit_sigma_mu2(params[:6], z, x_0, errors, cov, dint=params[6])
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

class Dataset(object):
    def __init__(self, base_path, bands, default = None, path_to_data = None, path_to_params = None, path_to_masses = None, index_col = None, name = ""):
        """ Constructor for the Dataset object, with option to use default built-in datasets or upload data from files. 
        To create a Dataset object, user must either use a default option or provide paths to all necessary files.
        
        Parameters
        ----------
        base_path: str, path to this package
        bands: List(str), list of bandpasses in dataset
        default: str, "yao" (for Yao et al. 2019) or "dhawan" (for Dhawan et al. 2022)
        path_to_data: str, system path to lightcurve data
        path_to_params: str, system path to lightcurve params
        index_col: str or int, column to use as DataFrame index
        name: str, ID for Dataset """

        self.name = ""
        self.bands = bands
        if name != None:
            self.name = name
        
        if default != None: # build dataset from default sample
            self.name = default
            if default == 'yao':
                self.data = pd.read_csv(base_path + '/data/yao_data.csv', index_col = 'SN')
                self.params = pd.read_csv(base_path + '/data/yao_params.csv', index_col = 'SN')
                self.masses = pd.read_csv(base_path + '/data/yao_masses.csv', index_col = 0).rename(columns={'0':'mass'})
            elif default == 'dhawan':
                self.data = pd.read_csv(base_path + '/data/dhawan_data.csv', index_col = 'SN')
                self.params = pd.read_csv(base_path + '/data/dhawan_params.csv', index_col = 'SN')
                self.masses = pd.read_csv(base_path + '/dhawan_masses.csv', index_col = 0).rename(columns={'0':'mass'})

        if path_to_data != None and path_to_params != None and index_col != None: # build dataset from path
            self.data = pd.read_csv(path_to_data, index_col = index_col)
            self.params = pd.read_csv(path_to_params, index_col = index_col)
            if path_to_masses != None:
                self.masses = pd.read_csv(path_to_masses, index_col = index_col)
            else:
                self.masses = None
                print('No masses found!')
        
        self.sn_names = pd.Series(self.data.index.unique()) # will hold all the valid SN

    def fit_salt(self, save_path = None, save_fig = None, verbose = False):
        """ Fit SALT3 parameters (z, t0, x0, x1, c) to lightcurves in the Dataset using SNCosmo. Also performs dust extinction modeling. 
        
        Parameters:
        -----------
        save_path: str, optional (default = None), path to directory to save SALT3 parameters. If None, parameters will not be saved outside the instance
        save_fig: str, optional (default = None), path to directory to save figures. If None, no figures will be saved
        verbose: bool (default = True), if True, print progress and parameters"""

        self.hubble = pd.DataFrame(index = self.sn_names, columns = ['z', 't0', 'dt0', 'x0', 'dx0', 'x1', 'dx1', 'c', 'dc', 'fmax', 'cov'])
        
        for i in range(len(self.sn_names))[:10]:
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
                max_fluxes[band] = float(fitted_model.bandflux(band,result['parameters'][np.array(result['param_names']) == 't0'],zp_sn,'ab'))
            self.hubble.loc[sn, 'fmax'] = [max_fluxes]
            
            # Make lightcurve figure
            if save_fig is not None:
                fig = sncosmo.plot_lc(tdata, model = fitted_model, errors = result.errors)
                plt.savefig(save_fig + "{}_{}_SALT3.pdf".format(self.name, sn), format='pdf', bbox_inches='tight')
                plt.close()
            
            self.hubble['cv'] = self.hubble['cv'].astype(object)
            self.hubble.loc[sn, 'cv'] = [result.covariance]
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
            print('{}: {{0:.3f}} $\pm$ {{0:.3f}}'.format(param, self.hubble[param].mean(), self.hubble[param].std()))
        
        if return_vals:
            return np.array([[self.hubble[param].mean(), self.hubble[param].std()] for param in params])

    def spectral_filter(self, spectra = None, ok_sn = ['SN Ia']):
        """ Load and apply a spectral type filter to the dataset's supernova. Usually should keep only normal SN Ia, rejecting peculiar spectral types.
        
        Parameters:
        -----------
        spectra: pandas Series or str, optional. If Series, index should match SN names. If str, spectra should indicate the column in self.params that contains the spectra.
        ok_sn: List(str) of spectral class labels that are OK. Any other SN will be rejected"""

        # Load in spectra
        if self.name == 'yao':
            ok_sn = ['normal   ', '91T-like ', '99aa-like']
            spectra = self.params['Subtype']
        
        elif self.name == 'dhawan':
            ok_sn = ['SN Ia', 'SN Ia?','SN Ia-norm', 'SN Ia 91T-like', 'SN Ia 91T', 'SN-91T', 'SNIa-99aa']
            spectra = pd.read_csv(DR1_PATH+'/samplefiles/Yr1_AllSNe_SampleFile_WithClassSpecCoords.txt', 
                      index_col=0,header=None)
            spectra = spectra[4]
        
        elif type(spectra) == str: # input is the name of a column in params
            spectra = self.params[spectra]
        
        elif spectra is None:
            print("No spectra included!")
            return
        
        # Remove all supernovae not satisfying spectral class requirements
        iterate = reversed(range(len(self.sn_names)))
        for i in iterate:
            sn = self.sn_names.loc[i]
            if spectra.loc[sn] not in ok_sn:
                self.sn_names.drop(i)

    
    def pv_correction(self, pecs):
        # TO-DO: currently have nothing to load in peculiar velocities
        """ Correct velocities and redshifts based on input peculiar velocities
        
        Parameters:
        -----------
        pecs: Pandas Series or str. If Series, index should match SN names. If str, pecs should indicate the column in self.params containing the peculiar velocities."""
        if pecs is not None:
            if type(pecs) == str:
                pecs = self.params[pecs]
            
            for i in range(len(self.sn_names)):
                sn = self.sn_names[i]
                pv = pecs.loc[sn]['vpec']
                z_cmb = pecs.loc[sn]['zcmb']
                self.hubble.loc[sn, 'z'] = (1.0 + z_cmb) / (1.0 + (pv / 3e8)) - 1.0

    def reset_all_cuts(self):
        """ Reset all cuts on spectral type, redshift, etc. and restore SN index to that of the original dataset."""

        self.sn_names = pd.Series(self.data.index.unique())

    def param_cuts(self, z_max = 0.1, min_points = 3, dx1_max = 1.0, dt0_max = 1.0, dc_max = 0.3, x1_max = 3.0, c_max = 0.3):
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

        iterate = reversed(range(len(self.sn_names))) # work backwards to avoid indexing issues
        for i in iterate:
            sn = self.sn_names.loc[i]
            sn_data = self.hubble.loc[sn]

            if sn_data['z'] > z_max:
                self.sn_names.drop(i)
                continue
            
            if len(sn_data[np.abs(sn_data['jd'] - self.hubble[sn, 't0']) < 10]) < min_points:
                self.sn_names.drop(i)
                continue
            
            if self.hubble[sn, 'dx1'] > dx1_max:
                self.sn_names.drop(i)
                continue

            if self.hubble[sn, 'dt0'] > dt0_max:
                self.sn_names.drop(i)
                continue

            if self.hubble[sn, 'dc'] > dc_max:
                self.sn_names.drop(i)
                continue

            if self.hubble[sn, 'x1'] > x1_max or self.hubble[sn, 'x1'] < -1 * x1_max:
                self.sn_names.drop(i)
                continue

            if self.hubble[sn, 'c'] > c_max or self.hubble[sn, 'x1'] < -1 * c_max:
                self.sn_names.drop(i)
                continue
            

    def fit_Tripp(self, verbose = False, mass_step = True):
        """ Fit Tripp equation with optional host-galaxy mass step, get distance modulus (mu) by redshift (z), and compare to Planck18 cosmology.
        
        Parameters:
        -----------
        verbose: bool (default = False), option to print out progress and fit results
        mass_step: bool (default = True), option to model host-galaxy mass step; if False, default range for parameter gamma is set to [-0.01, 0.01].
        """
        
        # Set up parameters for vectorized calculations
        z = self.hubble.loc[self.sn_names, 'z'].to_numpy()
        x_0 = self.hubble.loc[self.sn_names, 'x_0'].to_numpy()
        x_1 = self.hubble.loc[self.sn_names, 'x_1'].to_numpy()
        c = self.hubble.loc[self.sn_names, 'c'].to_numpy()
        e = self.hubble.loc[self.sn_names, ['dt0', 'dx0', 'dx1', 'dc']].to_numpy()
        cv = self.hubble.loc[self.sn_names, 'cv'].to_numpy() # check this
        masses = self.masses.loc[self.sn_names, 'mass'].to_numpy()

        # Optimize initial fit
        if verbose: print('Fitting Hubble residuals')

        x0_guess = [0.14, 3.0, 0.0, 19.36, 19.36, 19.36, 0.1]
        
        # Include optional mass step; if no mass step, restrict bounds tightly
        if mass_step:
            result = opt.minimize(cost_mu, x0 = x0_guess, args=(z, x_0, x_1, c, e, cv, masses))
        else: 
            gamma_bounds = ((0.0, 0.4), (2, 4), (-0.01, 0.01), (-20, -18), (-20, -18), (-20, -18), (0.0, 0.3))
            result = opt.minimize(cost_mu, x0 = x0_guess, bounds = gamma_bounds, args=(z, x_0, x_1, c, e, cv, masses))

        # Print output
        errs = np.sqrt(np.diagonal(result.hess_inv))

        if verbose: 
            result_string = ""
            for i in range(len(result.x)): # TO-DO: add in parameter names
                result_string += result.x[i]
                result_string += " $\pm$ "
                result_string += errs[i]
                result_string += " $\pm$, "
            print('Tripp parameters: {}'.format(result.x))

        mu = fit_mu(result.x, z, x_0, x_1, c, masses)
        mu_errs = fit_sigma_mu2(result.x, z, x_0, e, cv)

        return result, mu, mu_errs

    def fit_hubble(self, save_path = None, savefig = None, verbose = False, mass_step = True, outlier_cut = 5.0):
        """ Fit for Hubble residuals, including sample cuts, Tripp modeling, and outlier rejection.
        
        Parameters:
        -----------
        save_path: str, optional (default = None), path to directory to save SALT3 parameters. If None, parameters will not be saved outside the instance
        save_fig: str, optional (default = None), path to directory to save figures. If None, no figures will be saved
        verbose: bool (default = True); if True, print progress and parameters
        mass_step: bool (default = True), option to model host-galaxy mass step; if False, default range for parameter gamma is set to [-0.01, 0.01]
        outlier_cut: float (default = 5.0), minimum sigma difference from Planck18 distance modulus to label supernova as outlier
        """

        # TO-DO: build parameter cuts (?) directly into this function

        result, mu, mu_errs = self.fit_Tripp(verbose = verbose, mass_step = mass_step)
        z = self.hubble.loc[self.sn_names, 'z'].to_numpy()

        # Calculate and report outliers
        mu_resids = np.array(mu) - np.array(cosmo.distmod(z))
        mu_resids /= mu_errs

        if verbose: print('Outliers: ', list(self.sn_names[np.abs(mu_resids) > outlier_cut].index))
        self.sn_names = self.sn_names[np.abs(mu_resids) < outlier_cut]
        if verbose: print("Repeating Tripp fit without outliers")

        # Repeat analysis without outliers
        result, mu, mu_errs = self.fit_Tripp()

        if savefig is not None:
            plt.errorbar(z, mu, yerr = mu_errs, fmt='o', markersize=3, label='ZTF18')#, yerr=mu_errs)
            plt.plot(np.sort(z), cosmo.distmod(np.sort(z)), label='Planck18')
            plt.legend()
            plt.savefig(savefig + '/{}_hubble_diagram.pdf'.format(self.name))

        # Save distance moduli (mu) and uncertainties (mu_errs)
        self.hubble.loc[self.sn_names, 'mu'] = mu
        self.hubble.loc[self.sn_names, 'dmu'] = mu_errs

        if save_path is not None:
            self.hubble.loc[self.sn_names].to_csv(save_path + '{}_hubble.csv')

    def excess_search():
        print('Not yet implemented')
    
    def load_masses():
        print('Not yet implemented')
    
    def compare_excess():
        print('Not yet implemented')

