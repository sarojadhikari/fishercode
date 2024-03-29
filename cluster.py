"""
.. module:: cluster
    :synopsis: define a class for a Large Scale Structure (LSS) survey
    
"""

#from functions import degsq2rad
import numpy as np
from ..functions import xx, fratio, dndlnM, degsq2rad
from scipy import integrate
from scipy.special import erfc
from ..fisher import Fisher
import multiprocessing as mp

class Survey:
    """ large scale structure survey class.
            
        Args:
            * DeltaOmega: The survey size in square degrees
            * zmin: minimum redshift value of the survey
            * zmax: maximum redshift value of the survey
            * Nbins: number of redshift bins between zmin and zmax
            * name: [None, "eROSITA", "mag1000"] provides the possibility of using known survey geometry
             
    """   
     
    def __init__(self, DeltaOmega=2000., zmin=0.0, zmax=1.0, Nbins=10, name=None, alias=None):
        """The survey class.
           
        """
        self.Mth=1.0e14 # in Mpc/h
        self.sigma_lnM=0.25
        self.name=name

        if (alias):
            self.name=alias
        if name==None:
            self.dOm=degsq2rad(DeltaOmega) # the survey area in degree square is converted to radians square here
            self.zmin=zmin
            self.zmax=zmax
            self.nbins=Nbins
            self.dz=(zmax-zmin)/Nbins
            self.dlogM=0.1
        elif name=="eROSITA":
            self.dOm=degsq2rad(f=0.658)  # 65.8% of the sky (1111.6587)
            self.zmin=0.
            self.zmax=1.
            self.Mth=5.0e13
            self.dz=0.05
            self.dlogM=0.1
        elif name=="mag1000":    # see 1111.6587 sec 9.3
            self.dOm=degsq2rad(f=0.658)
            self.zmin=1.0
            self.zmax=3.0
            self.dz=0.05
            self.Mth=2.2e14
            self.dlogM=0.1
    
    def set_Mth(self, Mt):
        self.Mth=Mt
    
    def set_sigma_lnM(self, slnM):
        self.sigma_lnM=slnM
        

"""
.. module:: clusterfisher
    :synopsis: class for computing cluster number count fisher matrix, inherits the Fisher class
    
"""
#define the ClusterFisher class that calculates and manipulates fisher matrix for cluster analysis


class ClusterFisher(Fisher):
    """This class implements methods to compute the cluster number count, its derivatives
    with respect to various parameters, and the fisher matrix
    """
    def __init__(self, survey, cosmology, params=[], param_values=[], param_names=[], priors=[]):
        """Set the survey and cosmology for Cluster Fisher computations.
        Also, set the parameters and priors if specified
        """
        self.survey=survey
        self.cosmology=cosmology
        Fisher.__init__(self, params, param_values, param_names, priors)    
        
    def Nlm_integrand(self, z, M, Mm, Mmp, gfratio):
        """
        retun the integrand for Nlm; for a given l, m value, it assumes that the survey class
        has specified the redshift and mass range
        """
        nM=dndlnM(M, self.cosmology.rhom(), self.cosmology.sigmaM(M)*gfratio, self.cosmology.sigmaM_M(M)*gfratio)/self.cosmology.sigmaM(M)/gfratio
        
        if (self.cosmology.fnl!=0.0):
            nM=nM*self.cosmology.nGMFfactor(M,gfratio)
        if (self.cosmology.baryon_hmf_correction!=0):
            nM=nM*fratio(0,M)
        
        return nM*self.cosmology.volume_factor_integrand(z)*(erfc(xx(Mm, M, self.survey.sigma_lnM))-erfc(xx(Mmp, M, self.survey.sigma_lnM)))
    
    
    def Nlm(self, l, m):
        """
        compute the number of cluster expected using eq (17) from 1003.0841 in the specified 
        l-th redshift bin and m-th observed mass bin       
        """
        zl=self.survey.zmin + (l-1)*self.survey.dz
        
        # to speed things up (note that this is a good approximation for small redshift bins
        # and must be checked later) we will precompute the growth factor for the average 
        # redshift value in a redshift bin
        
        gf=self.cosmology.growth_factor(zl+self.survey.dz/2.0)
        gf0=self.cosmology.growth_factor(0.)
        gfratio=gf/gf0
                
        Mm=self.survey.Mth
        
        Mm=self.survey.Mth*(1+self.survey.dlogM*np.log(10))**m 
        for i in range(m):
            Mm=Mm+Mm*10**self.survey.dlogM
            
        Mmp=Mm*10**self.survey.dlogM
        result = integrate.nquad(self.Nlm_integrand, [[zl, zl+self.survey.dz], [Mm/1.e2, Mmp*1.e2]], args=[Mm, Mmp, gfratio])
        return result[0]*self.survey.dOm
    
    def Nlm_deriv(self, l, m, param, param_value, output):
        """compute the numerical derivative of Nlm (the cluster number count) w.r.t. the parameter specified
        at the given value
        """
        try:
            v=getattr(self.cosmology, param)
        except:
            v=getattr(self.survey, param)
        
        pv=param_value
        
        try:
            setfunc=getattr(self.cosmology, "set_"+param)
        except:
            setfunc=getattr(self.survey, "set_"+param)
        
        setfunc(pv*(1.+self.diff_percent))
        plus_value=self.Nlm(l, m)
        setfunc(pv*(1.-self.diff_percent))
        minus_value=self.Nlm(l,m)
        finite_diff=plus_value-minus_value
        delta_pv=2*self.diff_percent*pv
        setfunc(v)
        der=(finite_diff)/delta_pv
        output.put((self.parameters.index(param), der))
        return (finite_diff)/delta_pv
    
    def fisher(self):
        """
        computes the fisher matrix given the parameters, survey and cosmology definitions for this class.
        
        The fisher matrix is given by:
        
        .. math::
        
            F_{ij}=\\sum_{l,m} \\frac{\\partial N_{l,m}}{\\partial \\alpha_i} \\frac{\\partial N_{l,m}}{\\partial \\alpha_j} \\frac{1}{N_{l,m}}
            
        """
        # loop over the parameters (and sum over redshift bins) to form the Fisher matrix
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        dN=np.array([0.]*self.nparams)
        ls=int((self.survey.zmax-self.survey.zmin)/self.survey.dz)
        ms=int((15-np.log10(self.survey.Mth))/(np.log10(1+self.survey.dlogM*2.3)))
        for l in range(ls):
            for m in range(ms):
                Nbar=self.Nlm(l, m)
                output=mp.Queue()
                processes=[mp.Process(target=self.Nlm_deriv, args=(l, m, self.parameters[i], self.parameter_values[i], output)) for i in range(self.nparams)]

                for p in processes:
                    p.start()
                for p in processes:
                    p.join()

                dN=[output.get() for p in processes]
                dN.sort()
                dN=[r[1] for r in dN]
                #dN[i]=self.Nlm_deriv(l, m, self.parameters[i], self.parameter_values[i])
                #print dN
 
                for i in range(self.nparams):
                    for j in range(self.nparams):
                        fmatrix[i][j]=fmatrix[i][j]+dN[i]*dN[j]/Nbar
                        
        self.fisher_matrix=np.matrix(fmatrix)
        self.add_priors()
        return np.array(self.fisher_matrix)  # numpy array for easy indexing

    def test_fisher(self):
        """
        devise a test case that reproduces previously known (upto some tolerable accuracy)
        Fisher matrix computation as a check of this code
        """
        return 0
