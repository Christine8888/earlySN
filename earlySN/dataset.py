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
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    bigM = params[3:]
    m_shift = np.zeros(x_0.shape[0])
    if masses is not None:
        m_shift[masses['mass'] > 10] = 1
    mu_base = -2.5 * np.log10(x_0) + 10.635 + (alpha * x_1) - (beta * c) + gamma*m_shift
    mu_base[(z>0) & (z <= 0.033)] -= bigM[0]
    mu_base[(z > 0.033) & (z <= 0.067)] -= bigM[1]
    mu_base[(z > 0.067) & (z <= 0.1)] -= bigM[2]
    return mu_base

def cost_mu(params, z, x_0, x_1, c, errors, cov, masses):
    expected = np.array(cosmo.distmod(z))
    fit = np.array(fit_mu(params[:6], z, x_0, x_1, c, masses))
    err = fit_sigma_mu2(params[:6], z, x_0, errors, cov, dint=params[6])
    n = z.shape[0]
    return np.sum(-(fit-expected)**2/(2*(err**2)) - np.log(np.sqrt(2*math.pi*err**2))) * -1

def fit_sigma_mu2(params, z, x_0, errors, cov, pec = 250, dint = 0.1):
    dmuz = pec/3e5
    dlens = 0.055*z
    dmb = -2.5 * 0.434 * errors[:,1] / x_0 # mb = 2.5log10(x0) + constant
    dx1 = params[0]*errors[:,2]
    dc = params[1]*errors[:,3]
    dcx1 = 2 * params[0] * params[1] * cov[:,2,3]
    dmbx1 = 2 * params[0] * cov[:,1,2] * (-2.5/(x_0 * np.log(10.0)))
    dmbc = 2 * params[1] * cov[:,1,3] * (-2.5/(x_0 * np.log(10.0)))
    err = np.sqrt(dint**2 + dmuz**2 + dlens**2 + dmb**2 + dx1**2 + dc**2 + dcx1 + dmbx1 + dmbc)
    
    return err

class Dataset(object):
    def __init__(self, base_path, bands, default = None, path_to_data = None, path_to_params = None, index_col = None, name = ""):
        """ Constructor for the Dataset object, with option to use default built-in datasets or upload data from files. 
        To create a Dataset object, user must either use a default option or provide paths to all necessary files.
        
        Parameters
        ----------
        base_path: path to this package
        bands: list of bandpasses in dataset
        default: "yao" (for Yao et al. 2019) or "dhawan" (for Dhawan et al. 2022)
        path_to_data: system path to lightcurve data
        path_to_params: system path to lightcurve params
        index_col: column to use as DataFrame index
        name: ID for Dataset """

        self.name = ""
        self.bands = bands
        if name != None:
            self.name = name
        
        if default != None: # build dataset from default sample
            self.name = default
            if default == 'yao':
                self.data = pd.read_csv(base_path + '/data/yao_data.csv', index_col = 'SN')
                self.params = pd.read_csv(base_path + '/data/yao_params.csv', index_col = 'SN')
            elif default == 'dhawan':
                self.data = pd.read_csv(base_path + '/data/dhawan_data.csv', index_col = 'SN')
                self.params = pd.read_csv(base_path + '/data/dhawan_params.csv', index_col = 'SN')


        if path_to_data != None and path_to_params != None and index_col != None: # build dataset from path
            self.data = pd.read_csv(path_to_params, index_col = index_col)
            self.params = pd.read_csv(path_to_data, index_col = index_col)
        
        self.sn_names = pd.Series(self.data.index.unique())

    def fit_salt(self, save_path, save_fig = None, verbose = False):
        """ Fit SALT3 parameters (z, t0, x0, x1, c) to lightcurves in the Dataset using SNCosmo. Also performs dust extinction modeling.
        modeling,  
        
        Parameters:
        -----------
        save_path: path to directory to save SALT3 parameters
        save_fig: optional, path to directory to save figures. If None, no figures will be saved
        verbose: if True, print progress and parameters"""

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
            if save_fig is not None:
                fig = sncosmo.plot_lc(tdata, model = fitted_model, errors = result.errors)
                plt.savefig(save_fig + "{}_{}_SALT3.pdf".format(self.name, sn), format='pdf', bbox_inches='tight')
                plt.close()
            
            #self.hubble.loc[sn, 'cv'] = [result.covariance]
            self.hubble.loc[sn, ('t0', 'x0', 'x1', 'c')] = fitted_model.parameters[1:5] # do not use SALT3 z
            self.hubble.loc[sn, ('dt0', 'dx0', 'dx1', 'dc')] = result.errors['t0'], result.errors['x0'], result.errors['x1'], result.errors['c']
            

        if verbose: self.salt_stats()
        pd.to_csv(save_path + '{}_hubble.csv'.format(self.name))

        return self.hubble
        
    def salt_stats(self):
        print("Not yet implemented")

    def spectral_filter(self, spectra = None, ok_sn = ['SN Ia']):
        if self.name == 'yao':
            ok_sn = ['normal   ', '91T-like ', '99aa-like']
            spectra = self.params['Subtype']
        if self.name == 'dhawan':
            ok_sn = ['SN Ia', 'SN Ia?','SN Ia-norm', 'SN Ia 91T-like', 'SN Ia 91T', 'SN-91T', 'SNIa-99aa']
            spectra = pd.read_csv(DR1_PATH+'/samplefiles/Yr1_AllSNe_SampleFile_WithClassSpecCoords.txt', 
                      index_col=0,header=None)
            spectra = spectra[4]
    
    def pv_correction(self, pecs):
        if pecs is not None:
            for i in range(len(self.sn_names)):
                sn = self.sn_names[i]
                pv = pecs.loc[sn]['vpec']
                z_cmb = pecs.loc[sn]['zcmb']
                self.hubble.loc[sn, 'z'] = (1.0 + z_cmb) / (1.0 + (pv / 3e8)) - 1.0

    def param_cuts(self):
        iterate = reversed(range(len(self.sn_names)))
        for i in iterate:
            sn = self.sn_names.loc[i]
            sn_data = self.hubble.loc[sn]
            if sn_data['z'] > 0.1:
                self.sn_names.drop(i)

    def fit_Tripp(self, mass_step = True):
        print("Not yet implemented")
    
    def fit_hubble(self, save_path, savefig = None, verbose = False):

        print("Not yet implemented")

    def excess_search():
        print('Not yet implemented')
    
    def load_masses():
        print('Not yet implemented')

