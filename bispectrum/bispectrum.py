
from ..fisher import Fisher
from ..functions import top_hat
from scipy.integrate import quad, nquad
import numpy as np

class ibkLFisher(Fisher):
    """ implements methods to compute squeezed limit approximation of the reduced
    bispectrum that gives the position dependent power spectrum, and methods to 
    compute fisher matrix
    """
    def __init__(self, survey, cosmology, params=[], param_values=[], param_names=[], priors=[]):
        """
        """
        self.survey=survey
        self.cosmology=cosmology
        Fisher.__init__(self, params, param_values, param_names, priors)
        self.sigmaSqs = np.array([self.sigmadLSq(self.survey.Lboxes[i]) for i in range(self.survey.Nsub)])  # compute all the sigmaLsq values at once
        self.sigmaSqsfNL = np.array([self.sigmadLSqfNL(self.survey.Lboxes[i]) for i in range(self.survey.Nsub)])
        
    
    def DeltaibSq(self, k=0.5, box=0):
        Lbox = self.survey.Lboxes[box]  # get the physical length of the box # specified
        Vfactor = np.power(Lbox/self.survey.Lsurvey, 3.0)
        kmin = 2.*np.pi/Lbox
        #kmax = self.survey.kmax
        b1 = self.survey.b1fid
        Kp = self.Kaiser_factor_p(b1)
        #NkL = np.power(k/kmin, 2.0)  # check this
        NkL = np.power(Lbox, 3.0)*np.power(k, 2.0)*kmin/4./np.pi/np.pi
        sigma=b1*np.sqrt(Kp*self.sigmaSqs[box])
        cpower = Kp*b1*b1*self.cosmology.power_spectrumz(k, self.survey.z)
        term1 = np.power(sigma, 2.0) + self.survey.Pshot/np.power(Lbox, 3.0)
        term2 = cpower + self.survey.Pshot
        
        return Vfactor * term1 * np.power(term2, 2.0)/NkL/np.power(sigma, 4.0)/np.power(cpower, 2.0)
        

    def sigmadLSqfNL(self, Lbox=600.):
        integrand = lambda k: np.power(k*top_hat(k, Lbox/2.0), 2.0)*self.cosmology.power_spectrumz(k, self.survey.z)/self.cosmology.alpha(k, self.survey.z)
        results = quad(integrand, 0., np.infty)
        return results[0]/(2.*np.pi**2.0)
    
    def sigmadLSq(self, Lbox=600.):
        integrand = lambda k: np.power(k*top_hat(k, Lbox/2.0), 2.0)*self.cosmology.power_spectrumz(k, self.survey.z)
        #integrand = lambda kx, ky, kz: np.power(np.sinc(kx*Lbox/2.0)*np.sinc(ky*Lbox/2.0), 2.0)*self.cosmology.power_spectrumz(np.sqrt(kx**2.0+ky**2.0+kz**2.0), self.survey.z)
        #lb = self.survey.kmin/2.0
        #ub = self.survey.kmax*2.0
        #results = nquad(integrand, [[lb, ub], [lb, ub], [lb, ub]])
        results = quad(integrand, 0., np.infty)
        #return results[0]/np.power(2.*np.pi, 3.0)
        return results[0]/(2.*np.pi**2.0)
    
    def ibSPT(self, k, b1):
        return (68./21 - (1./3)*self.dlnk3Pkdlnk(k))/b1
    
    def ibb2(self, k, b1, b2):
        return 2.*b2/np.power(b1, 2.0)
    
    def ibfNL(self, k, b1, fNL=0., box=0):
        return 4.*fNL*self.sigmaSqsfNL[box]/self.sigmaSqs[box]/b1
        
    def ibtotal(self, k, b1, b2, fNL, box=0):
        Kb = self.Kaiser_factor_b(b1)
        #Kb=1.0
        return Kb*(self.ibfNL(k, b1, fNL, box=box)+self.ibSPT(k, b1)+self.ibb2(k, b1, b2))
    
    def Kaiser_factor_p(self, b1):
        """return the Kaiser factor 1 + 2./3 beta + 1./5 beta^2 for the power spectrum
        """
        f = self.cosmology.growth_rate_f(self.survey.z)
        beta = f/b1
        return 1. + (2./3)*beta + (1./5)*np.power(beta, 2.0)
    
    def Kaiser_factor_b(self, b1):
        """return the Kaiser factor 1 + 2./3 beta + 1./9 beta^2
        where beta = f/b1
        """
        f = self.cosmology.growth_rate_f(self.survey.z)
        beta = f/b1
        return 1.+ (2./3)*beta + (1./9)*np.power(beta, 2.0)
        
    def ibk_deriv(self, k, param="fNL", box=0):
        """ find the derivatives around the fiducial values defined in the survey class
        since the fid fNL parameter can be 0, lets use param+0.01 for finite difference
        of fNL, and fac=1.01 for the bias parameters that have non-zero fid values
        """
        fac=1.01
        sv=self.survey
        ibtfid=self.ibtotal(k, sv.b1fid, sv.b2fid, sv.fNLfid, box=box)
        if param=="fNL":
            ibkdif = self.ibtotal(k, sv.b1fid, sv.b2fid, sv.fNLfid+0.01, box=box)-ibtfid
            return ibkdif/(0.01)
        if param=="b1":
            ibkdif = self.ibtotal(k, sv.b1fid*fac, sv.b2fid, sv.fNLfid, box=box)-ibtfid
            return ibkdif/((fac-1.0)*self.survey.b1fid)
        if param=="b2":
            ibkdif = self.ibtotal(k, sv.b1fid, sv.b2fid*fac, sv.fNLfid, box=box)-ibtfid
            return ibkdif/((fac-1.0)*self.survey.b2fid)
            
    def fisher(self, skip=1.):
        """skip defines the binning scheme
        """
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        
        for i in range(self.nparams):
            for j in range(self.nparams):
                total=0.0
                for box in range(len(self.survey.Lboxes)):
                    kmin = 2.*np.pi/self.survey.Lboxes[box]
                    klist = np.arange(kmin, self.survey.kmax, skip*kmin)
                    dibk_list = np.array([self.ibk_deriv(k, param=self.param_names[i], box=box)*self.ibk_deriv(k, param=self.param_names[j], box=box)/self.DeltaibSq(k, box=box) for k in klist])
                    total = total + np.sum(dibk_list)
                fmatrix[i][j]=total

        self.fisher_matrix=np.matrix(fmatrix)
        return self.fisher_matrix
        
    def dlnk3Pkdlnk(self, k, fac=1.01):
        """return the logarithmic derivative of the linear matter power spectrum
        """
        Q2 = np.log(np.power(k*fac, 3.0)*self.cosmology.power_spectrumz(k*fac, self.survey.z))
        Q1 = np.log(np.power(k, 3.0)*self.cosmology.power_spectrumz(k, self.survey.z))
        return (Q2-Q1)/(np.log(k*fac)-np.log(k))
    #def ibkL_integrand(self, z, b1, b2, 
        
    def conv_power_spectrumz(self, k, L=600.):
        """return the convoled power at (k, self.survey.z) for a cubic box of side L
        """
        fac = 2.*np.pi/(np.power(2.*np.pi*L, 3.0))
        kmin = 2.*np.pi/L
        integrand = lambda q, mu: q*q * np.power(top_hat(q, L), 2.0)* self.cosmology.power_spectrumz(np.sqrt(k*k+q*q-2*k*q*mu), self.survey.z)
        results = nquad(integrand, [[kmin, self.survey.kmax*5.], [-1.,1.]])
        return fac*results[0]

class itkLFisher(ibkLFisher):
    """ similar to ibkLFisher but for the squeezed trispectrum
    """
    
    def DeltaitSq(self, k=0.5, box=0):
        Lbox = self.survey.Lboxes[box]  # get the physical length of the box # specified
        Vfactor = np.power(Lbox/self.survey.Lsurvey, 3.0)
        kmin = 2.*np.pi/Lbox
        b1 = self.survey.b1fid
        Kp = self.Kaiser_factor_p(b1)
        NkL = np.power(Lbox, 3.0)*np.power(k, 2.0)*kmin/4./np.pi/np.pi
        sigma=b1*np.sqrt(Kp*self.sigmaSqs[box])
        #cpower = b1*b1*self.cosmology.power_spectrumz(k, self.survey.z)
        cpower = Kp*b1*b1*self.cosmology.power_spectrumz(k, self.survey.z)
        term1 = np.power(sigma, 2.0) + self.survey.Pshot/np.power(Lbox, 3.0)
        term2 = cpower + self.survey.Pshot
        
        return Vfactor * term1 * np.power(term2, 3.0)/NkL/np.power(sigma, 4.0)/np.power(cpower, 4.0)
   
    def itgNL(self, k, b1, gNL, box=0):
        return 2.*gNL*self.sigmaSqsfNL[box]/self.sigmaSqs[box]/np.power(b1, 2.0)/self.cosmology.alpha(k, self.survey.z)
        
    def ittotal(self, k, b1, b2, gNL, box=0):
        """return the total integrated trispectrum in the squeezed limit, and
        when the rest of the three modes form a equilateral triangle with
        wave number amplitude k
        """
        return self.itgNL(k, b1, gNL, box=box)
        
    def itk_deriv(self, k, param="fNL", box=0):
        """ find the derivatives around the fiducial values defined in the survey class
        since the fid fNL parameter can be 0, lets use param+0.01 for finite difference
        of fNL, and fac=1.01 for the bias parameters that have non-zero fid values
        """
        fac=1.01
        sv=self.survey
        ittfid=self.ittotal(k, sv.b1fid, sv.b2fid, sv.fNLfid, box=box)
        if param=="fNL":
            itkdif = self.ittotal(k, sv.b1fid, sv.b2fid, sv.fNLfid+0.01, box=box)-ittfid
            return itkdif/(0.01)
        if param=="b1":
            itkdif = self.ittotal(k, sv.b1fid*fac, sv.b2fid, sv.fNLfid, box=box)-ittfid
            return itkdif/((fac-1.0)*self.survey.b1fid)
        if param=="b2":
            itkdif = self.ittotal(k, sv.b1fid, sv.b2fid*fac, sv.fNLfid, box=box)-ittfid
            return itkdif/((fac-1.0)*self.survey.b2fid)
            
    def fisher(self, skip=1.):
        """skip defines the binning scheme
        """
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        
        for i in range(self.nparams):
            for j in range(self.nparams):
                total=0.0
                for box in range(len(self.survey.Lboxes)):
                    kmin = 2.*np.pi/self.survey.Lboxes[box]
                    klist = np.arange(kmin, self.survey.kmax, skip*kmin)
                    dibk_list = np.array([self.itk_deriv(k, param=self.param_names[i], box=box)*self.itk_deriv(k, param=self.param_names[j], box=box)/self.DeltaitSq(k, box=box) for k in klist])
                    total = total + np.sum(dibk_list)
                fmatrix[i][j]=total

        self.fisher_matrix=np.matrix(fmatrix)
        return self.fisher_matrix

class Survey:
    """ large-scale structure survey class with fixed z but kmin/kmax and V specified
    """
    
    def __init__(self, z=0.57, kmax=0.17, b1fid=1.95, b2fid=0.5, fNLfid=0., Lsurvey=1500., ngbar=0.2248, Lboxes=[100., 200., 300., 400.]):
        self.z=z
        self.kmax=kmax+0.00001
        self.b1fid=b1fid
        self.b2fid=b2fid
        self.ngbar=ngbar
        self.Pshot = 1./self.ngbar
        
        self.fNLfid=fNLfid
        self.Lsurvey=Lsurvey
        self.kmin = 2.*np.pi/self.Lsurvey
        self.Lboxes=Lboxes
        self.Nsub = len(self.Lboxes)
    