
from fisher import Fisher
from functions import top_hat
from scipy.integrate import quad, nquad
import numpy as np

from os.path import isfile

from mpi4py import MPI

QLIMIT=1000  # limit on the integration cycles
GKMIN=0.0001
GKMAX=1.0

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
        self.survey.Pshot = 1./self.survey.ngbar
        self.fNLfid=0.0
        self.b1fid=0.0
        self.b2fid=0.0
        self.b3fid=0.0
        
        for pn in range(len(params)):
            if params[pn]=="fNL":
                self.fNLfid=param_values[pn]
            if params[pn]=="b1":
                self.b1fid=param_values[pn]
            if params[pn]=="b2":
                self.b2fid=param_values[pn]
            if params[pn]=="b3":
                self.b3fid=param_values[pn]
        
        Fisher.__init__(self, params, param_values, param_names, priors)
        Nsubs = range(self.survey.Nsub)
        
        self.sigmaSqs = np.array([self.sigmadLSq(self.survey.Lboxes[i]) for i in Nsubs])  # compute all the sigmaLsq values at once
        self.sigmaSqsfNL = np.array([self.sigmadLSqfNL(self.survey.Lboxes[i]) for i in Nsubs])
        self.sigmaSqsWL = np.array([self.sigmaWLSq(self.survey.Lboxes[i]) for i in Nsubs])
        self.sigmaPSq = np.array([self.sigmaWLPSq(self.survey.Lboxes[i]) for i in Nsubs])
        self.sigmaWLa = np.array([self.sigmaWLalpha(self.survey.Lboxes[i]) for i in Nsubs])
        
    
    def DeltaibSq(self, k=0.5, cp=1.0, box=0):
        Lbox = self.survey.Lboxes[box]  # get the physical length of the box # specified
        Vfactor = np.pi*np.power(Lbox/self.survey.Lsurvey, 3.0)/6.0
        Volume = 4.*np.pi*np.power(Lbox/2.0, 3.0)/3.
        kmin = 2.*np.pi/Lbox
        b1 = self.b1fid
        Kp = self.Kaiser_factor_p(b1)
        NkL = 2.*np.pi*np.power(k/kmin, 2.0)  # check this
        sigma=b1*np.sqrt(Kp*self.sigmaSqs[box])
        #cpower = Kp*b1*b1*self.cosmology.power_spectrumz(k, self.survey.z)/50. # fifty is ad-hoc test correction for the conv-power spectrum
        cpower = Kp*b1*b1*cp
        term1 = np.power(sigma, 2.0) + Kp*self.survey.Pshot/Volume
        term2 = cpower + self.survey.Pshot*Kp
        
        return Vfactor * term1 * np.power(term2, 2.0)/NkL/np.power(sigma, 4.0)/np.power(cpower, 2.0)
        
    def sigmaWLPSq(self, Lbox=600.):
        """
        """
        integrand = lambda k: np.power(k*top_hat(k, Lbox/2.0)*self.cosmology.power_spectrumz(k, self.survey.z), 2.0)
        results = quad(integrand, GKMIN, GKMAX, limit=QLIMIT)
        return results[0]/(2.*np.pi**2.0)
        
    def sigmaWLSq(self, Lbox=600.):
        """the integrand only has the window function and not the power spectrum
        """
        integrand = lambda k: np.power(k*top_hat(k, Lbox/2.0), 2.0)
        results = quad(integrand, GKMIN, GKMAX, limit=QLIMIT)
        return results[0]/(2.*np.pi**2.0)

    def sigmadLSqfNL(self, Lbox=600.):
        integrand = lambda k: np.power(k*top_hat(k, Lbox/2.0), 2.0)*self.cosmology.power_spectrumz(k, self.survey.z)/self.cosmology.alpha(k, self.survey.z)
        results = quad(integrand, GKMIN, GKMAX, limit=QLIMIT)
        return results[0]/(2.*np.pi**2.0)

    def sigmaWLalpha(self, Lbox=600.):
        integrand = lambda k: np.power(k*top_hat(k, Lbox/2.0), 2.0)*self.cosmology.alpha(k, self.survey.z)
        results = quad(integrand, GKMIN, GKMAX, limit=QLIMIT)
        return results[0]/(2.*np.pi**2.0)
    
    def sigmadLSq(self, Lbox=600.):
        integrand = lambda k: np.power(k*top_hat(k, Lbox/2.0), 2.0)*self.cosmology.power_spectrumz(k, self.survey.z)
        results = quad(integrand, GKMIN, GKMAX, limit=QLIMIT)
        return results[0]/(2.*np.pi**2.0)
    
    def ibSPT(self, k, b1):
        return (47./21 - (1./3)*self.dlnPkdlnk(k))/b1
    
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

        ibtfid=self.ibtotal(k, self.b1fid, self.b2fid, self.fNLfid, box=box)
        if param=="fNL":
            ibkdif = self.ibtotal(k, self.b1fid, self.b2fid, self.fNLfid+0.01, box=box)-ibtfid
            return ibkdif/(0.01)
        if param=="b1":
            ibkdif = self.ibtotal(k, self.b1fid*fac, self.b2fid, self.fNLfid, box=box)-ibtfid
            return ibkdif/((fac-1.0)*self.b1fid)
        if param=="b2":
            ibkdif = self.ibtotal(k, self.b1fid, self.b2fid*fac, self.fNLfid, box=box)-ibtfid
            return ibkdif/((fac-1.0)*self.b2fid)
            
    def fisher(self, skip=1.):
        """skip defines the binning scheme
        """
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        
        for box in range(len(self.survey.Lboxes)):
            
            klist, plist = self.cpower_saveload(box, skip=skip)

            for i in range(self.nparams):
                for j in range(self.nparams):
                    #total=0.0
                    dibk_list = np.array([self.ibk_deriv(klist[ki], param=self.parameters[i], box=box)*self.ibk_deriv(klist[ki], param=self.parameters[j], box=box)/self.DeltaibSq(klist[ki], plist[ki], box=box) for ki in range(len(klist))])
                    #total = total + np.sum(dibk_list)
                    fmatrix[i][j]=fmatrix[i][j]+np.sum(dibk_list)

        self.fisher_matrix=np.matrix(fmatrix)
        return self.fisher_matrix
    
    def cpower_saveload(self, box, skip=1.):
        kmin = 2.*np.pi/self.survey.Lboxes[box]
        klist = np.arange(kmin, self.survey.kmax, skip*kmin)
        cpfname = "conv_data/cpower_"+str(self.survey.Lboxes[box])+"_"+str(self.survey.kmax)+".npy"
        gfratio = self.cosmology.growth_factor(self.survey.z)/self.cosmology.growth_factor(0.0)
        if isfile(cpfname):
            plist = np.load(cpfname)
            if (self.survey.z!=0.0):
                plist = plist*gfratio
        else:
            plist = np.array([self.conv_power_spectrumz(k, L=self.survey.Lboxes[box])/gfratio for k in klist])
            np.save(cpfname, plist)
        return klist, plist
    
    def dlnPkdlnk(self, k, fac=1.001):
        """return the logarithmic derivative of the linear matter power spectrum
        """
        Q2 = np.log(self.cosmology.power_spectrumz(k*fac, self.survey.z))
        Q1 = np.log(self.cosmology.power_spectrumz(k, self.survey.z))
        return (Q2-Q1)/(np.log(k*fac)-np.log(k))
    #def ibkL_integrand(self, z, b1, b2, 
    
    def mpi_conv_power(self, klist, L=600.):
        listall=[]
        #MPI.Init()
        comm = MPI.COMM_WORLD
        Ncores = comm.Get_size()
        rank = comm.Get_rank()
        
        Nk = len(klist)
        bdown = int(Nk/Ncores)
        print (Ncores, Nk, bdown)

        for i in range(0, bdown):
            # first scatter klist
            if rank == 0:
                kdata = klist[i*Ncores:(i+1)*Ncores]
            else:
                kdata = None
                
            kdata = comm.scatter(kdata, root=0)            
            kdata = self.conv_power_spectrumz(kdata, L)
            comm.Barrier()
            
            kdata = comm.gather(kdata, root=0)
            
            if (kdata != None):
                listall = np.append(listall, kdata)
            
            comm.Barrier()
        # do anything that is left
        if (bdown*Ncores < Nk):
            kdata = klist[bdown*Ncores:Nk]
            kdata = self.conv_power_spectrumz(kdata, L)
            listall = np.append(listall, kdata)
            comm.Barrier()
        
        print (listall)    
        return listall.flatten()
    
    def conv_power_spectrumz(self, k, L=600.):
        """return the convoled power at (k, self.survey.z) for a cubic box of side L
        """
        Volume=4.*np.pi*np.power(L/2.0, 3.0)/3.
        fac = Volume/(4.*np.pi*np.pi)
        kmin = 2.*np.pi/L
        integrand = lambda q, mu: q*q * np.power(top_hat(q, L), 2.0)* self.cosmology.power_spectrumz(np.sqrt(k*k+q*q-2*k*q*mu), self.survey.z)
        options={'limit':QLIMIT}
        results = nquad(integrand, [[kmin/10., self.survey.kmax*10.], [-1.,1.]], opts=[options, options])
        return fac*results[0]

class itkLFisher(ibkLFisher):
    """ similar to ibkLFisher but for the squeezed trispectrum
    """
    
    def DeltaitSq(self, k=0.5, cp=1.0, box=0):
        Lbox = self.survey.Lboxes[box]  # get the physical length of the box # specified
        Vfactor = np.pi*np.power(Lbox/self.survey.Lsurvey, 3.0)/6.0
        Volume = 4.*np.pi*np.power(Lbox/2.0, 3.0)/3.
        kmin = 2.*np.pi/Lbox
        b1 = self.b1fid
        Kp = self.Kaiser_factor_p(b1)
        #NkL = np.power(Lbox, 3.0)*np.power(k, 2.0)*kmin/4./np.pi/np.pi
        NkL = 2.*np.pi*np.power(k/kmin, 2.0)
        sigma=b1*np.sqrt(Kp*self.sigmaSqs[box])
        #cpower = b1*b1*self.cosmology.power_spectrumz(k, self.survey.z)
        #cpower = Kp*b1*b1*self.cosmology.power_spectrumz(k, self.survey.z)/60. # 60 is ad-hoc at the moment
        cpower = Kp*b1*b1*cp
        term1 = np.power(sigma, 2.0) + Kp*self.survey.Pshot/Volume
        term2 = cpower + Kp*self.survey.Pshot
        
        return Vfactor * term1 * np.power(term2, 3.0)/NkL/np.power(sigma, 4.0)/np.power(cpower, 4.0)
   
    def itgNL(self, k, b1, gNL, box=0):
        term1 = 3*self.sigmaSqsfNL[box]
        #term2 = self.cosmology.power_spectrumz(k, self.survey.z)*self.sigmaWLa[box]/np.power(self.cosmology.alpha(k, self.survey.z), 2.0)
        term2=0.
        #print (term1, term2)
        return (6*gNL/(self.cosmology.alpha(k, self.survey.z)* b1*b1*self.sigmaSqs[box]))*(term1 + term2)
        #return 6.*gNL*self.sigmaSqsfNL[box]/self.sigmaSqs[box]/np.power(b1, 2.0)/self.cosmology.alpha(k, self.survey.z)
    

    def itSPT(self, k, b1, box=0):
        return (579./98 - (32./21.)*self.dlnPkdlnk(k))/np.power(b1, 2.0)
    
    def itSPTcon(self, k, b1, box=0):
        return (4./7.)*(73./7.-2.*self.dlnPkdlnk(k))/np.power(b1, 2.0)
        
    def itL2(self, k, b1, b2, box=0):
        logP = self.dlnPkdlnk(k)
        cpower = self.cosmology.power_spectrumz(k, self.survey.z)
        term1 = 7.*(98.-11.*logP)
        term2 = cpower* (242.-28.*logP)*self.sigmaSqsWL[box]/self.sigmaSqs[box]
        term3 = (68.+7.*logP)*self.sigmaPSq[box]/self.sigmaSqs[box]/cpower
        return (b2/b1**3.0)*(term1 + term2 + term3)/42.
        
    def itL3(self, k, b1, b2, box=0):
        return 12.*(b2**2.0/b1**4.0)*(1.+self.cosmology.power_spectrumz(k, self.survey.z)*self.sigmaSqsWL[box]/self.sigmaSqs[box])
        
    def itL4(self, k, b1, b3, box=0):
        return 3.*(b3/b1**3.0)*(1.+self.cosmology.power_spectrumz(k, self.survey.z)*self.sigmaSqsWL[box]/self.sigmaSqs[box]/3.)
        
    def ittotal(self, k, b1, b2, b3, gNL, box=0):
        """return the total integrated trispectrum in the squeezed limit, and
        when the rest of the three modes form a equilateral triangle with
        wave number amplitude k
        """
        return self.itgNL(k, b1, gNL, box=box) + self.itSPT(k, b1, box)+ self.itL4(k, b1, b3, box) +self.itL3(k, b1, b2, box)+self.itL2(k, b1, b2, box)
        
    def itk_deriv(self, k, param="fNL", box=0):
        """ find the derivatives around the fiducial values defined in the survey class
        since the fid fNL parameter can be 0, lets use param+0.01 for finite difference
        of fNL, and fac=1.01 for the bias parameters that have non-zero fid values
        """
        dif=0.001

        ittfid=self.ittotal(k, self.b1fid, self.b2fid, self.b3fid, self.fNLfid, box=box)
        if param=="fNL":
            itkdif = self.ittotal(k, self.b1fid, self.b2fid, self.b3fid, self.fNLfid+dif, box=box)-ittfid
            return itkdif/(dif)
        if param=="b1":
            itkdif = self.ittotal(k, self.b1fid+dif, self.b2fid, self.b3fid, self.fNLfid, box=box)-ittfid
            return itkdif/(dif)
        if param=="b2":
            itkdif = self.ittotal(k, self.b1fid, self.b2fid+dif, self.b3fid, self.fNLfid, box=box)-ittfid
            return itkdif/(dif)
        if param=="b3":
            itkdif = self.ittotal(k, self.b1fid, self.b2fid, self.b3fid+dif, self.fNLfid, box=box)-ittfid
            return itkdif/(dif)
            
    def fisher(self, skip=1.):
        """skip defines the binning scheme
        """
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        
        for box in range(len(self.survey.Lboxes)):
            print ("box number: ", box, "/", len(self.survey.Lboxes))

            klist, plist = self.cpower_saveload(box, skip=skip)
            
            for i in range(self.nparams):
                for j in range(self.nparams):
                    for box in range(len(self.survey.Lboxes)):
                        dibk_list = np.array([self.itk_deriv(klist[ki], param=self.params[i], box=box)*self.itk_deriv(klist[ki], param=self.params[j], box=box)/self.DeltaitSq(klist[ki], plist[ki], box=box) for ki in range(len(klist))])
                        
                        fmatrix[i][j]=fmatrix[i][j]+np.sum(dibk_list)

        self.fisher_matrix=np.matrix(fmatrix)
        return self.fisher_matrix
    

class Survey:
    """ large-scale structure survey class with fixed z but kmin/kmax and V specified
    """
    
    def __init__(self, z=0.57, kmax=0.17, Lsurvey=1500., ngbar=0.2248, Lboxes=[100., 200., 300., 400., 500.]):
        """
        The Lbox here is the diameter of a cubic volume
        """
        self.z=z
        self.kmax=kmax
        self.ngbar=ngbar
        self.Pshot = 1./self.ngbar
        self.Lsurvey=Lsurvey
        self.kmin = 2.*np.pi/self.Lsurvey
        self.Lboxes=Lboxes
        self.Nsub = len(self.Lboxes)
    
