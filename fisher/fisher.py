 
"""
.. module:: fisher
   :synopsis: base class that defines the Fisher matrix class, manipulates it 
    and performs simple computations
    
"""

import numpy as np
from functions import *
from matplotlib.patches import Ellipse

class Fisher(object): 
    """
    This is the base fisher matrix class that is inherited by :class:`.ClusterFisher` 
    and :class:`.CMBFisher` classes. 
    """
    def __init__(self, params=[], param_values=[], param_names=[], priors=[]):
        self.parameters=params
        self.parameter_values=param_values
        self.nparams=len(params)
        self.priors=priors
        self.fisher_matrix=0.
        self.marginal_fisher_matrix=0.
        self.covariance_matrix=0.
        self.marginal_covariance_matrix=0.
        self.marginal_param_list=0.
        self.prior_information_added=False
        self.diff_percent=0.01 # for finite difference to compute numerical derivative

        if len(param_names)<1:
            self.param_names=params
        else:
            self.param_names=param_names
    
    def fisher(self):
        """
        method to compute the fisher matrix given the parameters, survey and 
        cosmology definitions for this class. The computations are performed in the inherited classes such as :class:`.CMBFisher` and :class:`.ClusterFisher`. 
        """
        # loop over the parameters (and sum over redshift bins) to form the Fisher matrix
        
        return np.array(self.fisher_matrix)  # numpy array for easy indexing
    
    def add_priors(self, prior_inf=[]):
        """
        add prior information 1/sigma^2 to the fisher matrix; the prior matrix is given in
        percentage, so needs to be multiplied by the parameter values (done below in code).
        """
        if len(prior_inf)>0:
            self.priors=prior_inf
        # else use existing prior information    
        prior_matrix=np.array([[0.]*self.nparams]*self.nparams)
        for i in range(self.nparams):
            if self.priors[i]>0:
                prior_matrix[i][i]=1./(self.priors[i]*self.parameter_values[i])**2.0
        self.fisher_matrix=self.fisher_matrix+np.matrix(prior_matrix)
        self.prior_information_added=True
        return self.fisher_matrix
        
    def covariance(self):
        """
        compute the inverse of fisher matrix
        """
        self.covariance_matrix=self.fisher_matrix.I
        return np.array(self.covariance_matrix)
    
    def marginalize(self, param_list=[0,1]):
        """
        compute and return a new fisher matrix after marginalizing over all 
        other parameters not in the list param_list
        
        also save the marginalized fisher matrix in self.marginal_fisher_matrix
        and save the new covariance matrix in self.marginal_covariance_matrix
        """ 
        npars=self.nparams
        trun_cov_matrix=self.covariance_matrix
        mlist=[]
        for i in range(npars):
            if i not in param_list:
                mlist.append(i)
        trun_cov_matrix=np.delete(trun_cov_matrix, mlist, 0)
        trun_cov_matrix=np.delete(trun_cov_matrix, mlist, 1)
        
        self.marginal_covariance_matrix=trun_cov_matrix
        self.marginal_fisher_matrix=trun_cov_matrix.I
        self.martinal_param_list=param_list
        return trun_cov_matrix.I
    
    def save_fisher_matrix(self, fname):
        """
        save the current fisher matrix and relevant information in the DETF format as name.fisher
        """
        hd=""
        for param in parameters:
            hd=hd+param+" "
            
        np.savetext(fname, self.fisher_matrix, header=hd)
        return 0
    
    def load_fisher_matrix(self, fname):
        """
        load the fisher matrix in file fname. currently, one has to set the parameter names etc
        manually.
        """
        self.fisher_matrix=np.loadtxt(fname)
        return 0
        
    def sigma_i(self, i):
        """
        return the 1-sigma error for the ith parameter (given a 2x2 cov matrix)
        """
        return np.sqrt(np.array(self.covariance_matrix)[i][i])
    
    def corr_coeff(self, i, j):
        """
        return the correlation coefficient between ith and jth parameters
        """
        sigma_ij=self.covariance_matrix.item(i,j)
        return sigma_ij/self.sigma_i(i)/self.sigma_i(j)
     
    def error_ellipse(self, i, j, nstd=1, space_factor=3, clr='b', alpha=0.5):
        """
        return the plot of the error ellipse from the covariance matrix
        use ideas from error_ellipse.m from 
        http://www.mathworks.com/matlabcentral/fileexchange/4705-error-ellipse
        
        * (i,j) specify the ith and jth parameters to be used and
        * nstd specifies the number of standard deviations to plot, the default is 1 sigma. 
        * space_factor specifies the number that divides the width/height, the result of which is then added as extra space to the figure
        """
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::1]
            return vals[order], vecs[:,order]
        
        self.marginalize(param_list=[i,j])
        vals, vecs = eigsorted(self.marginal_covariance_matrix)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        print theta
        width, height = 2* nstd * np.sqrt(vals)
        xypos=[self.parameter_values[i], self.parameter_values[j]]
        ellip = Ellipse(xy=xypos, width=width, height=height, angle=theta, color=clr, alpha=alpha)
        ellip_vertices=ellip.get_verts()
        xl=[ellip_vertices[i][0] for i in range(len(ellip_vertices))]
        yl=[ellip_vertices[i][1] for i in range(len(ellip_vertices))]
        dx=(max(xl)-min(xl))/space_factor
        dy=(max(yl)-min(yl))/space_factor
        xyaxes=[min(xl)-dx, max(xl)+dx, min(yl)-dy, max(yl)+dy]
        return ellip, xyaxes
            
    def plot_error_ellipse(self, i, j, xyaxes_input=0, nstd=1, clr='b', alpha=0.5):
        """
        """
        import matplotlib.pyplot as plt
        ax = plt.gca()
        errorellipse, xyaxes=self.error_ellipse(i,j, nstd=nstd, clr=clr, alpha=alpha)
        ax.add_artist(errorellipse)
        if (xyaxes_input!=0):
            ax.axis(xyaxes_input)
        else:
            ax.axis(xyaxes)
            
        plt.xlabel(self.param_names[i])
        plt.ylabel(self.param_names[j])
        plt.show()
        return plt
        
