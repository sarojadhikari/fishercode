"""
.. module:: cmbfisher
   :synopsis: class for computing simple CMB Fisher matrices.
   
.. moduleauthor:: Saroj Adhikari <adh.saroj@gmail.com>
"""

import numpy as np
from functions import *
from fisher import Fisher
from classy import Class

class CMBFisher(Fisher): 
    """This class is for computing the CMB Fisher matrix given by
    
    .. math::
        
        F_{ij}=\\sum_l \\frac{(2l+1)}{2} \\frac{\\frac{\\partial C_l}{\\partial \\alpha_i} \\frac{\\partial C_l}{\\partial \\alpha_j}}{(C_l+w^{-1} e^{\\sigma^2 l^2})^2}
    
    The :math:`C_l` values are computed from the currently set cosmology.
    
    """
    
    def __init__(self, expt, cosmology, params=[], param_values=[], param_names=[], priors=[], pol=False):
        """Set the experiment and cosmology for CMB Fisher computations.
        Also, set the parameters and priors if specified
        """
        self.experiment=expt
        self.cosmology=cosmology
        self.include_polarization=pol
        Fisher.__init__(self, params, param_values, param_names, priors)
            
    def theoryCls(self, LMAX):
        """get the theoretical :math:`C_l` values for the current cosmology using CLASS code.
        The cosmological parameters are set from the current :class:`.cosmology`.
        The power sepctra are also set as variables in the cosmology class.
        
        If the include_polarization switch is set to True, then it also sets
        
        """
        params={
            'output': 'tCl',
            'l_max_scalars': LMAX,
            'modes': 'st',
            'A_s': self.cosmology.A,
            'n_s': self.cosmology.n,
            'r': self.cosmology.r,
            'h': self.cosmology.h,
            'omega_b': self.cosmology.Ob0*self.cosmology.h**2.0,
            'omega_cdm': self.cosmology.Om0*self.cosmology.h**2.0,
            }
        
        if (self.include_polarization):
            params['output']='tCl,pCl'
            
        cosmo=Class()
        cosmo.set(params)
        cosmo.compute()
        
        self.cosmology.TTCls=1.e12*cosmo.raw_cl(LMAX)['tt']
        
        if (self.include_polarization):
            self.cosmology.TECls=1.e12*cosmo.raw_cl(LMAX)['te']
            self.cosmology.BBCls=1.e12*cosmo.raw_cl(LMAX)['bb']
            self.cosmology.EECls=1.e12*cosmo.raw_cl(LMAX)['ee']
        
        return self.cosmology.TTCls
    
    def getCls(self, ps='tt'):
        """return one of the TT, TE, EE, BB Cls
        """
        if (ps=='te'):
            return self.cosmology.TECls
        elif (ps=='bb'):
            return self.cosmology.BBCls
        elif (ps=='ee'):
            return self.cosmology.EECls
        else:
            return self.cosmology.TTCls
        
    def Cls_deriv(self, param, param_value, ps='tt'):
        """compute the numerical derivative of :math:`C_ls`, the angular temperature power spectrum
        with respect to the parameter specified at the given value
        
        Note: currently, the implementation is for TT power spectrum; one can easily extend to include
        EE and BB by summing over
        """
        v=getattr(self.cosmology, param)
        pv=param_value
        setfunc=getattr(self.cosmology, "set_"+param)
        
        setfunc(pv*(1.+self.diff_percent))
        self.theoryCls(self.experiment.lmax)
        plus_value=self.getCls(ps)
        setfunc(pv*(1.-self.diff_percent))
        self.theoryCls(self.experiment.lmax)
        minus_value=self.getCls(ps)
        finite_diff=plus_value-minus_value
        delta_pv=2*self.diff_percent*pv
        setfunc(v)
        return (finite_diff)/delta_pv
        
    def fisherXX(self, ps='tt'):
        """computes the fisher matrix given the parameters, experiment and cosmology definitions
        """
        # loop over multipoles to form the Fisher matrix
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        dCij=[0.]*self.nparams
        
        if (self.experiment.lmax>256):
            self.theoryCls(self.experiment.lmax)
        else:
            self.theoryCls(256)
        
        ClXX=self.getCls(ps)        
        
        for i in range(self.nparams):
            dCij[i]=self.Cls_deriv(self.parameters[i], self.parameter_values[i], ps)
                    
        for i in range(self.nparams):
            for j in range(self.nparams):
                fmatrix[i][j]=np.sum(np.array([(self.experiment.fsky*(2*l+1)/2)*(dCij[i]*dCij[j])/(ClXX[l]+self.experiment.wT**(-1) *np.exp((self.experiment.sigma*l)**2.0)) for l in range(2, self.experiment.lmax-1)]))
        return np.array(np.matrix(fmatrix))  # numpy array for easy indexing

    def fisher(self, XX=['tt', 'te', 'ee', 'bb']):
        """sum over the specified XX=[TT, TE, EE, BB] fisher matrices to get the
        total CMB fisher matrix
        """
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        for xx in XX:
            fmatrix=fmatrix+self.fisherXX(xx)
        
        self.fisher_matrix=np.matrix(fmatrix)
        return fmatrix

    def test_fisher(self):
        """
        devise a test case that reproduces previously known (upto some tolerable accuracy)
        Fisher matrix computation as a check of this code
        """
        return 0

