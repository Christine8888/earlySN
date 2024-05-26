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
#dustmaps.sfd.fetch()
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
import scipy.optimize as opt 
from scipy import stats 
from earlySN import lightcurve
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

def avg_and_error(data, errs):
    err_weights = 1 / errs ** 2
    avg = np.average(data, weights = err_weights)
    err = np.sqrt(1 / np.sum(errs))

    return avg, err

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

def mass_statistics(sn_names, masses):
    std = np.std(masses.loc[sn_names].dropna())[0]
    avg = np.average(masses.loc[sn_names].dropna())
    err = std * np.sqrt(1/masses.loc[sn_names].dropna().shape[0])
    print(masses.loc[sn_names].dropna().shape[0])
    return avg, err 

class Dataset(object):
    def __init__(self, base_path, bands, default = None, path_to_data = None, path_to_params = None, path_to_masses = None, index_col = None, name = ""):
        """ Constructor for the Dataset object, with option to use default built-in datasets or upload data from files. 
        To create a Dataset object, user must either use a default option or provide paths to all necessary files.
        
        Parameters
        ----------
        base_path: str, path to this package
        bands: List(str), list of bandpasses in dataset
        default: str, "yao" (for Yao et al. 2019) or "dhawan" (for Dhawan et al. 2022) or "burke" (for Burke et al. 2022b)
        path_to_data: str, system path to lightcurve data
        path_to_params: str, system path to lightcurve params
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
                self.pecs = pd.read_csv(base_path + '/data/yao_pvs.txt', index_col = 0).rename(columns = {'0': 'v'})
            elif default == 'dhawan':
                self.data = pd.read_csv(base_path + '/data/dhawan_data.csv', index_col = 'SN')
                self.params = pd.read_csv(base_path + '/data/dhawan_params.csv', index_col = 'SN')
                self.masses = pd.read_csv(base_path + '/data/dhawan_masses.csv', index_col = 0).rename(columns={'0': 'mass'})
                self.pecs = pd.read_csv(base_path + '/data/dhawan_pvs.txt', index_col = 0).rename(columns = {'0': 'v'})

        if path_to_data != None and path_to_params != None and index_col != None: # build dataset from path
            self.data = pd.read_csv(path_to_data, index_col = index_col)
            self.params = pd.read_csv(path_to_params, index_col = index_col)
            if path_to_masses != None:
                self.masses = pd.read_csv(path_to_masses, index_col = index_col)
            else:
                self.masses = None
                print('No masses found!')
        
        self.hubble = None
        self.sn_names = pd.Series(self.data.index.unique()) # will hold all the valid SN

    def fit_salt(self, save_path = None, verbose = False):
        """ Fit SALT3 parameters (z, t0, x0, x1, c) to lightcurves in the Dataset using SNCosmo. Also performs dust extinction modeling. 
        
        Parameters:
        -----------
        save_path: str, optional (default = None), path to directory to save SALT3 parameters and figures. If None, parameters will not be saved outside the instance
        verbose: bool (default = True), if True, print progress and parameters"""

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
                max_fluxes[band] = float(fitted_model.bandflux(band,result['parameters'][np.array(result['param_names']) == 't0'],zp_sn,'ab'))
            self.hubble.loc[sn, 'fmax'] = [max_fluxes]
            
            # Make lightcurve figure
            if save_path is not None:
                fig = sncosmo.plot_lc(tdata, model = fitted_model, errors = result.errors)
                plt.savefig(save_path + "{}_{}_SALT3.pdf".format(self.name, sn), format='pdf', bbox_inches='tight')
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
        if self.name == 'yao' or self.name == "burke":
            ok_sn = ['normal   ', '91T-like ', '99aa-like']
            spectra = self.params['Subtype']
        
        elif self.name == 'dhawan':
            ok_sn = ['SN Ia', 'SN Ia?','SN Ia-norm', 'SN Ia 91T-like', 'SN Ia 91T', 'SN-91T', 'SNIa-99aa']
            spectra = pd.read_csv(self.base_path + '/data/dhawan_data.csv', index_col=0)
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

    
    def pv_correction(self):
        """ Correct velocities and redshifts based on input peculiar velocities
        
        Parameters:
        -----------
        pecs: Pandas Series or str. If Series, index should match SN names. If str, pecs should indicate the column in self.params containing the peculiar velocities."""
        pecs = self.pecs
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
        x_0 = self.hubble.loc[self.sn_names, 'x0'].to_numpy()
        x_1 = self.hubble.loc[self.sn_names, 'x1'].to_numpy()
        c = self.hubble.loc[self.sn_names, 'c'].to_numpy()
        e = self.hubble.loc[self.sn_names, ['dt0', 'dx0', 'dx1', 'dc']].to_numpy()
        cv = self.hubble.loc[self.sn_names, 'cov'].to_numpy() # check this
        masses = self.masses.loc[self.sn_names, 'mass'].to_numpy()

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
                result_string += result.x[i]
                result_string += " $\pm$ "
                result_string += errs[i]
                result_string += " $\pm$, "
            print('Tripp parameters: {}'.format(result_string))

        mu = fit_mu(result.x, z, x_0, x_1, c, masses)
        mu_errs = fit_sigma_mu2(result.x, z, x_0, e, cv)

        return result, mu, mu_errs

    def fit_hubble(self, save_path = None, savefig = None, verbose = False, mass_step = True, outlier_cut = 5.0):
        """ Fit for Hubble residuals, including sample cuts, Tripp modeling, and outlier rejection.
        
        Parameters:
        -----------
        save_path: str, optional (default = None), path to directory to save SALT3 parameters. If None, parameters will not be saved outside the instance
        save_path: str, optional (default = None), path to directory to save figures. If None, no figures will be saved
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

    def excess_search(self, save_path = None):
        targets = self.sn_names
        self.gauss_params = pd.DataFrame(index = targets, columns=['t_exp', 'A_r', 'A_g', 'alpha_r', 'alpha_g', 'mu', 'sigma', 'C_r', 'C_g', 'B_r', 'B_g',
                                                                   'dt_exp', 'dA_r', 'dA_g', 'dalpha_r', 'dalpha_g', 'dmu', 'dsigma', 'dC_r', 'dC_g', 'dB_r', 'dB_g'])
        self.pl_params = pd.DataFrame(index = targets, columns=['t_exp', 'A_r', 'A_g', 'alpha_r', 'alpha_g', 'B_r', 'B_g',
                                                                'dt_exp', 'dA_r', 'dA_g', 'dalpha_r', 'dalpha_g', 'dB_r', 'dB_g'])
        
        self.N = pd.DataFrame(index = targets, columns=['N'])

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
            if sn not in not_bronze:
                data = self.data.loc[sn]
                data = data[data['flux'] > data['flux_err'] > -2] # remove large negative outliers
                
                params = self.hubble.loc[sn]
                fit = lightcurve.Lightcurve(sn, data, params, self.bands, self.name, save_path = None, verbose = False)
                pl_params, gauss_params, classification, self.N[sn] = fit.excess_search()
                
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
            with open(save_path + '/{}_gold.txt'.format(self.name), 'w') as f:
                for line in self.gold:
                    f.write(f"{line}\n")
            
            with open(save_path + '/{}_gold_nd.txt'.format(self.name), 'w') as f:
                for line in self.gold_nd:
                    f.write(f"{line}\n")

            with open(save_path + '/{}_bronze.txt'.format(self.name), 'w') as f:
                for line in self.bronze:
                    f.write(f"{line}\n")

    # basically all of this is from the notebook -- NOTE TO SELF
    def compare_excess(self):
        mu = self.hubble['mu']
        mu_errs = self.hubble['dmu']
        z = self.hubble['z']

        color = ["red" if sn in self.gold else "black" for sn in self.sn_names]
        color = pd.DataFrame({"SN": self.sn_names, "color": color}).set_index("SN", inplace=True)['color'] 
        alpha = [1 if sn in self.gold else 0.3 for sn in self.sn_names]
        alpha = alpha.astype(np.float)
        alpha = pd.DataFrame({"SN": self.sn_names, "alpha": alpha}).set_index("SN", inplace=True)['alpha']

        # Plot for Final Article
        y = mu * u.mag - (cosmo.distmod(z))
        fig, ax = plt.subplots(1,2, figsize=(10,5),width_ratios=[3,1],sharey=True)

        offset = np.average(y, weights=1/mu_errs**2).value

        for sn in self.sn_names:
            if sn in self.gold:
                ax[0].errorbar(z[sn], y[sn], yerr = mu_errs[sn] * u.mag, markersize = 9, c = 'gold', zorder = 10, marker = 's', capsize = 5)
            else:
                ax[0].errorbar(z[sn], y[sn], yerr = mu_errs[sn] * u.mag, fmt = 'o', alpha = alpha[sn], markersize = 9, color = color[sn], capsize=5)
            
            
        ax[0].axhline(y=offset, ls='--', c='b')#np.sort(z), np.ones(len(z))*offset,

        gold_avg, gold_err = avg_and_error(y[self.gold], mu_errs[self.gold])
        nd_avg, nd_err = avg_and_error(y[self.gold_nd], mu_errs[self.gold_nd])
        excess_avg, excess_err = avg_and_error(y[self.excess], mu_errs[self.excess])
        noexcess_avg, noexcess_err = avg_and_error(y[self.nd], mu_errs[self.nd])
        
        #print('Excess Avg. Residual: {}'.format(excess_avg))
        #print('Error: {}'.format(excess_err))
        #print('No Excess Avg. Residual: {}'.format(noexcess_avg))
        #print('Error: {}'.format(noexcess_err))
        #print('Pop. Avg. Residual: {}'.format(np.average(y, weights = 1/mu_errs**2)))
        #print('Error: {}'.format(np.sqrt(1/np.sum(1/mu_errs[no_excess]**2))))

        ax[1].hist(y[self.excess], alpha = 0.3, color = 'r', label = 'All Excess', orientation = 'horizontal')
        hist = ax[1].hist(y[self.nd], alpha = 0.3,color = 'k', label = 'All No Excess', orientation = 'horizontal')
        ax[1].axhline(y = excess_avg, ls = '--', c = 'r')
        x = np.arange(0, np.max(hist[0] + 2))
        ax[1].fill_between(x, y1 = excess_avg - excess_err, y2 = excess_avg + excess_err, color = 'r', alpha = 0.3)
        ax[1].fill_between(x, y1 = noexcess_avg - noexcess_err, y2 = noexcess_avg + noexcess_err, color = 'k', alpha = 0.3)
        ax[1].axhline(y = noexcess_avg, ls = '--', c = 'k')
        ax[1].legend()
        ax[1].set(xlim = (0, np.max(hist[0] + 1)))
        ax[1].set_xticks([])

        pos = {'yao': 0.015, 'burke': 0.015, 'ztf': 0.01}
        heights = {'yao': [0.4, 0.6], 'burke': [0.4, 0.6], 'ztf': [-0.7, -0.45]}
        ylims = {'yao': [-0.5, 0.68], 'burke': [-0.5, 0.68], 'ztf': [-0.75, 0.8]}

        align = 'left'
        text_pos = pos[self.name]
        text_heights = np.linspace(heights[self.name][0], heights[self.name][1], 4)
        ax[0].set_ylim(ylims[self.name][0], ylims[self.name][1])

        ax[0].text(text_pos, text_heights[1], 'All Excess: ${:.3f} \pm {:.3f}$'.format(excess_avg, excess_err), horizontalalignment=align,)
        ax[0].text(text_pos, text_heights[2], 'All No Excess: ${:.3f} \pm {:.3f}$'.format(noexcess_avg, noexcess_err), horizontalalignment=align,)
        ax[0].text(text_pos, text_heights[0], 'Gold Excess: ${:.3f} \pm {:.3f}$'.format(gold_avg, gold_err), horizontalalignment=align,)
        ax[0].text(text_pos, text_heights[3], 'Gold No Excess: ${:.3f} \pm {:.3f}$'.format(nd_avg, nd_err), horizontalalignment=align,)
        plt.legend()
        print(np.std(y))

        titles = {'yao': 'Yao et al. 2019', 'ztf': 'Dhawan et al. 2022', "burke": 'Burke et al. 2022b'}
        fig.suptitle(titles[self.name])

        ax[0].set_ylabel('$\mu_{SALT3} - \mu_{\Lambda_{CDM}}$ (mag)')
        ax[0].set_xlabel('z')

        plt.savefig(+'./{}_hubble.pdf'.format(self.name), bbox_inches = 'tight', pad_inches = 0)

    def load_tiers(self):
        burke_gold = ['ZTF18aaxsioa', 'ZTF18abcflnz', 'ZTF18abssuxz', 
          'ZTF18abxxssh', 'ZTF18aavrwhu', 'ZTF18abfhryc',]
        burke_bronze = ['ZTF18aawjywv', 'ZTF18aaqcozd', 'ZTF18abdfazk',
         'ZTF18abimsyv', 'ZTF18aazsabq']
        also_in_literature = ['ZTF18aapqwyv', 'ZTF18aaqqoqs', 'ZTF18aayjvve', 'ZTF18abckujq', 'ZTF18abcrxoj',
                     'ZTF18abdfazk', 'ZTF18abdfwur', 'ZTF18abfhryc', 'ZTF18abgxvra', 'ZTF18abpamut',]
        
        if self.name == "burke":
            self.gold = burke_gold
            self.bronze = burke_bronze
            
        self.excess = self.bronze + self.gold
        self.nd = [sn for sn in self.sn_names if sn not in self.excess]
        self.gold_nd = [sn for sn in self.gold_nd if (sn not in burke_gold and sn not in burke_bronze and sn not in also_in_literature)]

    def set_N(self, sn, N):
        self.N.loc[sn] = N


    def compare_mass(self):
        cn = ["#68affc", "#304866"][0]
        ce = ["#68affc", "#304866"][1]

        fig, ax = plt.subplots(1, 1, figsize=(8,4))
        std = np.std(self.masses)[0]

        gold_avg, gold_err = mass_statistics(self.gold, self.masses)
        nd_avg, nd_err = mass_statistics(self.gold, self.masses)
        excess_avg, excess_err = mass_statistics(self.excess, self.masses)
        noexcess_avg, noexcess_err = mass_statistics(self.nd, self.masses)
        
        print('No Excess Avg. Mass: {}'.format(excess_avg))
        print('Error: {}'.format(excess_err))
        print('No Excess Avg. Mass: {}'.format(noexcess_avg))
        print('Error: {}'.format(noexcess_err))

        hist = plt.hist(self.masses.loc[self.nd], edgecolor = cn, alpha = 0.5, color = cn, label = 'All No Excess', bins = 10)
        plt.hist(self.masses.loc[self.excess], bins = hist[1], edgecolor = ce, alpha=0.5, color = ce, label = 'All Excess')
        plt.legend()

        center = {'yao': 8.2, 'burke': 8.2, 'ztf': 6.8}
        ranges = {'yao': [11, 15], 'burke': [11, 15], 'ztf': [22, 43]}
        titles = {'yao': 'Yao et al. 2019', 'ztf': 'Dhawan et al. 2022', "burke": 'Burke et al. 2022b'}

        center = center[self.name]
        positions = np.linspace(ranges[self.name][0], ranges[self.name][1], 4)

        # burke/yao: 8.9, 8, 10
        # ztf: 8, 12, 20

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
        plt.suptitle(titles[self.name])

        plt.savefig('./{}_mass.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)
            
    def compute_bump_properties(self):
        """ Compute properties of the bump in the light curve.
        
        Parameters
        ----------- 
        result: object output from scipy.minimize (created by self.fit_model)
        bands: list (string) of band names
        """

        gauss_params = self.gauss_params.loc[self.excess]

        bump_r = lightcurve.flux_from_amp(gauss_params["mu"], gauss_params["sigma"], gauss_params["C_r"])
        bump_g = lightcurve.flux_from_amp(gauss_params["mu"], gauss_params["sigma"], gauss_params["C_g"])
        err_r = gauss_params["C_r"] / gauss_params["dC_r"] 
        err_g = gauss_params["C_g"] / gauss_params["dC_g"]

        gauss_params["color"] = -2.5 * np.log10(bump_g / bump_r)
        dgr = np.sqrt((err_r / bump_r) ** 2 + (err_g / bump_g) ** 2)
        gauss_params["dcolor"] = -2.5 * 0.434 * dgr / gauss_params["color"]    
        
        ten_r = gauss_params["A_r"] * 10 ** gauss_params["alpha_r"] + gauss_params["B_r"]
        ten_g = gauss_params["A_g"] * 10 ** gauss_params["alpha_g"] + gauss_params["B_g"]

        gauss_params["fr"] = bump_r / ten_r
        gauss_params["dfr"] = (gauss_params["dC_r"] / gauss_params["C_r"]) * gauss_params["fr"]
        gauss_params["fg"] = bump_g / ten_g
        gauss_params["dfg"] = (gauss_params["dC_g"] / gauss_params["C_g"]) * gauss_params["fg"]

        self.gauss_params = gauss_params

    def analyze_bump_shapes(self):
        gauss_params = self.gauss_params.loc[self.gold]

        fig, ax = plt.subplots(2,1, figsize=(6,5))
        
        for sn in gauss_params.index.unique():
            row = gauss_params.loc[sn]
            x = np.linspace(-3, 6, 200)
            
            for i in range(10):
                mu = row['mu'] #+ np.random.normal(0, 1) * row['dmu']
                sigma = row['sigma'] #+ np.random.normal(0, 1) * row['dsigma']
                y = stats.norm.pdf(x, loc = mu, scale = sigma)
            
                y_g = y * row['fg'] / stats.norm.pdf(mu, loc = mu, scale = sigma,)
                ax[0].plot(x, y_g, c = 'g', alpha = 0.1, )
                
                
                y_r = y * row['fr'] / stats.norm.pdf(mu, loc = mu, scale = sigma,)
                ax[1].plot(x, y_r, c = 'r', alpha = 0.1,)
        
        ax[0].plot(x, y_g, c = 'g', alpha=1,  label = 'g')
        ax[1].plot(x, y_r, c = 'r', alpha=1,  label='r')
            
        ax[0].legend()
        ax[1].legend()
        
        fig.supylabel('$f_{bump}$ / $f_{10}$')
        ax[1].set_xlabel('JD since First Light')
            
        mu_avg, mu_err = avg_and_error(gauss_params['mu'], gauss_params['dmu'])
        print(mu_avg, mu_err)
        ax[0].axvspan(mu_avg - mu_err, mu_avg + mu_err, color='k', alpha=0.3)
        ax[1].axvspan(mu_avg - mu_err, mu_avg + mu_err, color='k', alpha=0.3)

        sig_avg, sig_err = avg_and_error(gauss_params['sigma'], gauss_params['dsigma'])
        print(sig_avg, sig_err)

        plt.savefig('./{}_curves.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)

    def analyze_bump_amps(self):
        fig, ax = plt.subplots(figsize=(6,5))
        gauss_params = self.gauss_params.loc[self.gold]

        # TODO: adapt to multiple band sizes (?) 
        r_avg, r_err = avg_and_error(gauss_params['fr'], gauss_params['dfr'])
        print('r:', r_avg, r_err)
        g_avg, g_err = avg_and_error(gauss_params['fg'], gauss_params['dfg'])
        print('g:', g_avg, g_err)

        diff = g_avg - r_avg
        d_err = np.sqrt(r_err**2 + g_err**2)

        print('$fg - fr = {} \pm {}$'.format(diff, d_err))

        color_avg, color_err = avg_and_error(gauss_params['color'], gauss_params['dcolor'])
        print('$\log_{{10}}(G - R) = {} \pm {}$'.format(color_avg, color_err))

        plt.errorbar(gauss_params['fr'], gauss_params['fg'], yerr = gauss_params['dfg'], 
                    xerr = gauss_params['dfr'], fmt = 'o', c = 'k', markersize = 5, alpha = 0.8)
        
        plt.axhspan(g_avg - g_err, g_avg + g_err, alpha = 0.5, color = 'g')
        plt.axvspan(r_avg - r_err, r_avg + r_err, hatch = 'X',facecolor = 'r', edgecolor = 'k', alpha = 0.5)

        #plt.plot(x,x,'--', c='b')
        plt.xlim(-0.01, 0.24)
        plt.ylim(-0.01, 0.24)
        plt.xlabel('$f_{bump,r}$ / $f_{10,r}$')
        plt.ylabel('$f_{bump,g}$ / $f_{10,g}$')

        plt.savefig('./{}_bumps.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)

    def analyze_PL(self):
        pl_params = self.pl_params.loc[self.gold]
        
        fig, ax = plt.subplots(figsize=(6,5))

        err_weights = 1 / pl_params['dalpha_r'] ** 2
        r_avg = np.average(pl_params['alpha_r'], weights = err_weights)
        r_err = np.sqrt(1 / np.sum(err_weights))
        print(r_avg, r_err)

        err_weights = 1 / pl_params['dalpha_g'] ** 2
        g_avg = np.average(pl_params['alpha_g'], weights = err_weights)
        g_err = np.sqrt(1 / np.sum(err_weights))
        print(g_avg, g_err)

        diff = g_avg - r_avg
        d_err = np.sqrt(r_err**2 + g_err**2)

        print('$\alpha_g - \alpha_r = {} \pm {}$'.format(diff, d_err))

        plt.errorbar(pl_params['alpha_r'], pl_params['alpha_g'], yerr = pl_params['dalpha_g'], 
                    xerr = pl_params['dalpha_r'], fmt = 'o', c = 'k', markersize = 5)
       
        x = np.linspace(1.1, 2.75)
        plt.axhspan(g_avg - g_err, g_avg + g_err, alpha = 0.5, color = 'g', label = '$\overline{\alpha}_g$ $(1\sigma)$')
        plt.axvspan(r_avg - r_err, r_avg + r_err, alpha = 0.5, hatch = 'X',facecolor = 'r', edgecolor = 'k', label = '$\overline{\alpha}_r$ $(1\sigma)$')

        plt.plot(x,x,'--', c='k', alpha=0.5, label='$\alpha_g = \alpha_r$')
        plt.legend()
        plt.xlabel('$\alpha_r $')
        plt.ylabel('$\alpha_g $')

        plt.savefig('./{}_slopes.pdf'.format(self.name), bbox_inches='tight', pad_inches=0)
    
    def compare_fit_params(self, params = ['x1', 'c']):
        for param in params:
            err_weights = 1 / self.hubble[self.gold, "d{}".format(param)] ** 2
            gold_avg = np.average(self.hubble[self.gold, param], weights = err_weights)
            gold_err = np.sqrt(1 / np.sum(err_weights))
            print('Gold {}: '.format(param), gold_avg, gold_err)

            err_weights = 1 / self.hubble[self.excess, "d{}".format(param)] ** 2
            excess_avg = np.average(self.hubble[self.excess, param], weights = err_weights)
            excess_err = np.sqrt(1 / np.sum(1 / err_weights))
            print('Excess $x_1$: ', excess_avg, excess_err)

            err_weights = 1 / self.hubble[self.nd, "d{}".format(param)] ** 2
            noexcess_avg = np.average(self.hubble[self.nd, param], weights = err_weights)
            noexcess_err = np.sqrt(1 / np.sum(1 / err_weights))
            print('No Excess $x_1$: ', noexcess_avg, noexcess_err)
    
    def save_all(self):
        # need: to be able to pick up from 'wherever we left off'
        print("Not yet implemented")

    def make_table(self):
        print("Not yet implemented")

    def end_to_end(self, verbose, save_path):
        # Processing steps
        self.fit_salt(verbose, save_path)
        self.spectral_filter()
        self.pv_correction()
        self.param_cuts()
        self.fit_hubble() # includes Tripp fitting step

        # Light curve fitting
        self.excess_search()
        self.load_tiers()

        # Analysis steps
        self.compare_excess()
        self.compare_mass()