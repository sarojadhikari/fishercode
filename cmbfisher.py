"""
.. module:: cmb
    :synopsis: define a class for a CMB experiment

"""
import numpy as np
from fishercode import Fisher
import camb

class CMBExperiment(object):
    """ CMB experiment class

        Args:
            * DeltaOmega: The sky covered in square degrees
            * LMAX: maximum multipole to be considered
            * theta: resolution (FWHM, arcmin)
            * sigmaT: temperature noise per pixel (:math:`\mu K`)
            * name: [None, "Planck", "WMAP", "PIXIE"] provides the possibility of using a known experiment

    """

    def __init__(self, DeltaOmega=41253., LMIN=2, LMAX=2000, beamsize=5.0, w=8000.,
                 name=None, alias=None):
        """The CMB experiment class.

        """
        self.lmax=LMAX
        self.lmin=LMIN
        self.lmaxT = self.lmaxP = LMAX
        self.theta=np.deg2rad(beamsize/60.)    # beamsize is specified in arcmin
        self.wT=w   # (1/(pixel size*noise per pixel^2)
        self.name=name
        self.fsky=DeltaOmega/41253

        if (alias):
            self.name=alias
        if name==None:
            self.dOm = np.deg2rad(DeltaOmega)**2.0 # the survey area in degree square is converted to radians square here
        if name=="CVlimited":
            """define a cosmic variance limited CMB temperature and polarization experiment
            with the following properties
            """
            if not(alias):
                self.name="CVlimited"
            self.lmax=2500
            self.lmaxT = self.lmaxP = self.lmax
            self.fsky = 1
            self.wT = np.infty
            self.wP = np.infty

        if name=="Planck": # this really is Planck hiL
            if not(alias):
                self.name="Planck"
            self.lmaxP=1996
            self.lmaxT=2508
            self.lmax = max(self.lmaxP, self.lmaxT)
            self.lmin = 30
            self.fsky = 0.5
            self.frequency = 143
            self.theta=np.deg2rad(7./60.)   # in arcmin
            self.noiseppT=6.0  # noise per pixel for temperature in microK
            self.noiseppP=11.5    # noise per pixel for polarization in microK
            self.wT=1.0/(self.theta*self.noiseppT)**2.0 # units of 1/microK^2
            self.wP=1.0/(self.theta*self.noiseppP)**2.0

        self.lrangeT = range(self.lmin, self.lmaxT+1)
        self.lrangeE = range(self.lmin, self.lmaxP+1)
        self.lrangeC = self.lrangeE

        self.fskyT = self.fskyE = self.fskyP = self.fskyC = self.fsky

        if (self.lmaxT<self.lmaxP):
            self.lrangeC = self.lrangeT

Planck2015expt = CMBExperiment(name="Planck")
            
Planck2015pars = {'H0':     67.8,
              'Obh2':   0.02226,
              'Och2':   0.1186,
              'tau':    0.066,
              'ln10As': 3.062,
              'ns':     0.9677,
              'mnu':    0.06,
              'Neff':   3.046,
              'r':      0.,
              'omk':    0.}

    
class CMBFisher(Fisher):
    """This class is for computing the CMB Fisher matrix given by

    .. math::

        F_{ij}=\\sum_l \\frac{(2l+1)}{2} \\frac{\\frac{\\partial C_l}{\\partial \\alpha_i} \\frac{\\partial C_l}{\\partial \\alpha_j}}{(C_l+w^{-1} e^{\\sigma^2 l^2})^2}

    The :math:`C_l` values are computed from the currently set cosmology.

    """

    def __init__(self, expt=Planck2015expt, cosmology=Planck2015pars, 
                 params=[], param_values=[], param_names=[], priors=[], pol=True):
        """Set the experiment and cosmology for CMB Fisher computations.
        Also, set the parameters and priors if specified
        """
        self.experiment=expt
        self.cosmology=cosmology
        self.include_polarization=pol
        self.Cl_covmat_computed=False
        Fisher.__init__(self, params, param_values, param_names, priors)

    def theoryCls(self, muKunits = True):
        """get the theoretical :math:`C_l` values for the current cosmology using CLASS code.
        The cosmological parameters are set from the current :class:`.cosmology`.
        The power sepctra are also set as variables in the cosmology class.

        If the include_polarization switch is set to True, then it also sets

        the noise are usually set in muK units; so muKunits needs to be True for that
        """
        LMAX = self.experiment.lmax

        # using camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.cosmology['H0'], 
                           ombh2=self.cosmology['Obh2'],
                           omch2=self.cosmology['Och2'], 
                           omk=self.cosmology['omk'],
                           tau=self.cosmology['tau'], 
                           mnu=self.cosmology['mnu'],
                           nnu=self.cosmology['Neff'])
        As = np.exp(self.cosmology['ln10As'])*1.e-10
        pars.InitPower.set_params(As=As, ns=self.cosmology['ns'], r=self.cosmology['r'])
        pars.set_for_lmax(LMAX+200)

        if (self.cosmology['r'] > 0.0):
            pars.WantTensors = True

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, raw_cl=True)
        totCL = powers['total']


        mufac = 1
        if (muKunits):
            mufac = (2.7255E6)**2.0

        self.TTCls = mufac*totCL[:LMAX+1,0]
        self.transfer_functions = camb.get_transfer_functions(pars)

        if (self.include_polarization):
            self.TECls = mufac*totCL[:LMAX+1,3]
            self.BBCls = mufac*totCL[:LMAX+1,2]
            self.EECls = mufac*totCL[:LMAX+1,1]
            # return the TT, TE, EE spectra in that order (CovMatrix assumes this)
            return np.append(self.TTCls[self.experiment.lrangeT],
                            np.append(self.TECls[self.experiment.lrangeC],
                                        self.EECls[self.experiment.lrangeE]))

        return self.TTCls[self.experiment.lrangeT]

    def getCls(self, ps='tt'):
        """return one of the TT, TE, EE, BB Cls
        """
        if (ps=='te'):
            return self.TECls
        elif (ps=='bb'):
            return self.BBCls
        elif (ps=='ee'):
            return self.EECls
        else:
            return self.TTCls

    def Cls_deriv(self, param, param_value):
        """compute the numerical derivative of :math:`C_ls`, the angular temperature power spectrum
        with respect to the parameter specified at the given value
        """
        v = self.cosmology[param]
        pv = param_value
        if (pv==0.):
            # may need to be careful about the step-size
            self.cosmology[param] = self.diff_percent
            delta_pv = 2*self.diff_percent
        else:
            self.cosmology[param] = pv*(1.+self.diff_percent)
            delta_pv = 2*self.diff_percent*pv
        
        plus_value = self.theoryCls()
        
        if (pv==0.):
            self.cosmology[param] = -self.diff_percent
        else:
            self.cosmology[param] = pv*(1.-self.diff_percent)
        
        minus_value = self.theoryCls()
        
        finite_diff = plus_value - minus_value
                
        self.cosmology[param] = v
        return (finite_diff)/delta_pv
        
    def Cov_deriv(self, param, param_value):
        """ compute the numerical derivative of the covariance matrix
        """
        v = self.cosmology[param]
        pv = param_value
        self.cosmology[param] = pv*(1.+self.diff_percent)
        plus_value = self.FullCovarianceMatrix()
        
        self.cosmology[param] = pv*(1.-self.diff_percent)
        minus_value = self.FullCovarianceMatrix()
        
        finite_diff = plus_value-minus_value
        delta_pv=2.*self.diff_percent*pv
        
        # set the default value back
        self.cosmology[param] = v
        return (finite_diff)/delta_pv

    def noise_weight(self, ps='tt'):
        """return the noise weight for the power spectrum specified
        """
        if (self.experiment.name=="CVlimited"):
            return 0.
        if (ps=='tt'):
            return (1./self.experiment.wT)
        elif (ps=='ee' or ps =='bb'):
            return (1./self.experiment.wP)
        else:
            return 0.

    def noise_powers(self):
        """
        the noise power spectra
        """
        amtorad = 0.000290888
        lrange = range(0, self.experiment.lmax+1) # start from zero for easy indexing
        
        if self.experiment.name == "Planck":
            # use the formula and specifications from 1403.5271 to compute noise 
            freqs = [143, 217]
            DTT = [1.5, 3.3] # in muK/K
            DPT = [3.0, 6.6] # in muK/K
            TCMB = 2.7255
            theta = [7., 5.]
            
            TNoiseChannels = []
            ENoiseChannels = []
            
            X = 0
            for freq in freqs:
                wTinv = (DTT[X]*TCMB*theta[X]*amtorad)**2.0
                wEinv = (DPT[X]*TCMB*theta[X]*amtorad)**2.0
                noiseT = wTinv * np.array([np.exp(l*(l+1)*(theta[X]*amtorad)**2.0/8./np.log(2.)) for l in range(0, self.experiment.lmaxT+1)])
                noiseE = wEinv * np.array([np.exp(l*(l+1)*(theta[X]*amtorad)**2.0/8./np.log(2.)) for l in range(0, self.experiment.lmaxT+1)])
                TNoiseChannels.append(noiseT)
                ENoiseChannels.append(noiseE)
                
                X = X + 1
            
            # now combine the two channels
            Tnoise = 1./((1./TNoiseChannels[0]) + (1./TNoiseChannels[1]))
            Enoise = 1./((1./ENoiseChannels[0]) + (1./ENoiseChannels[1]))
            
        else:
            th = self.experiment.theta
            Tnoise = np.array([self.noise_weight('tt')*np.exp((th*l)**2.0/8./np.log(2.)) for l in lrange])
            Enoise = np.array([self.noise_weight('ee')*np.exp((th*l)**2.0/8./np.log(2.)) for l in lrange])
        
        
        
        return np.array([Tnoise, Enoise])

    def fisher(self, include_cov_term=False):
        """computes the fisher matrix given the parameters, experiment and cosmology definitions
        """
        # loop over multipoles to form the Fisher matrix
        fmatrix=np.array([[0.]*self.nparams]*self.nparams)
        dCij=[0.]*self.nparams

        if (self.experiment.lmax>256):
            self.theoryCls(self.experiment.lmax)
        else:
            self.theoryCls(256)

        if not(self.Cl_covmat_computed):
            covmat = self.FullCovarianceMatrix()
        else:
            covmat = self.Cl_covmat

        invcov = np.linalg.inv(covmat)

        for i in range(self.nparams):
            dCij[i]=self.Cls_deriv(self.parameters[i], self.parameter_values[i])

        for i in range(self.nparams):
            for j in range(self.nparams):
                #fijl = dCij[i]@invcov@dCij[i]
                fmatrix[i][j] = dCij[i]@invcov@dCij[j]
                
        if (include_cov_term):
            # add the 0.5*Tr[C^{-1}C_,iC^{-1}C_,j] term to see if there is any effect
            #print ("not implemented")
            dCovij = [0.]*self.nparams
            for i in range(self.nparams):
                dCovij[i] = self.Cov_deriv(self.parameters[i], self.parameter_values[i])
            
            for i in range(self.nparams):
                for j in range(self.nparams):
                    term2ij = 0.5*np.trace(invcov@dCovij[i]@invcov@dCovij[j])
                    print (term2ij/fmatrix[i][j])
                    fmatrix[i][j] = fmatrix[i][j] + term2ij

        self.fisher_matrix = np.matrix(fmatrix)  # numpy array for easy indexing
        return self.fisher_matrix

    def FullCovarianceMatrix(self, turn_off_noise = False, CC=True):
        """
        construct the full "block-diagonal" CMB covariance matrix for data
            {ClTT, ClTE, ClEE}
        that is
            [[TTTT, TTTE, TTEE],
             [TETT, TETE, TEEE],
             [EETT, EETE, EEEE]]

        See for example: astro-ph/9807130
        """
        lrangeT = self.experiment.lrangeT
        lrangeE = self.experiment.lrangeE
        lrangeC = self.experiment.lrangeC

        fskyT = self.experiment.fskyT
        fskyE = self.experiment.fskyE
        fskyC = np.sqrt(fskyT*fskyE)

        Tnoise, Enoise = self.noise_powers()

        if (turn_off_noise):
            # option to turn off noise for testing purposes
            Tnoise = np.zeros(np.shape(Tnoise))
            Enoise = np.zeros(np.shape(Enoise))

        fskyfactorT = np.array([1./((2.*l+1)*fskyT) for l in lrangeT])
        fskyfactorE = np.array([1./((2.*l+1)*fskyE) for l in lrangeE])
        fskyfactorC = np.array([1./((2.*l+1)*fskyC) for l in lrangeC])

        # make sure theory Cls are calculated
        self.theoryCls()

        CTl = self.TTCls + Tnoise

        CovTT = np.diagflat(2.*fskyfactorT*CTl[lrangeT]**2.0)

        if (self.include_polarization):
            CEl = self.EECls + Enoise
            CCl = self.TECls

            CovEE = np.diagflat(2.*fskyfactorE*CEl[lrangeE]**2.0)
            CovCC = np.diagflat(fskyfactorC*(CCl[lrangeC]**2.0 + CTl[lrangeC]*CEl[lrangeC]))

            CovTC = np.zeros((np.size(lrangeT), np.size(lrangeC)))
            np.fill_diagonal(CovTC, 2.*fskyfactorC*CCl[lrangeC]*CTl[lrangeC])

            CovTE = np.zeros((np.size(lrangeT), np.size(lrangeE)))
            np.fill_diagonal(CovTE, 2.*fskyfactorC*CCl[lrangeC]**2.0)

            CovEC = np.zeros((np.size(lrangeE), np.size(lrangeC)))
            np.fill_diagonal(CovEC, 2.*fskyfactorC*CCl[lrangeC]*CEl[lrangeC])

            covmat = np.bmat([[CovTT, CovTC, CovTE],
                            [CovTC.T, CovCC, CovEC.T],
                            [CovTE.T, CovEC, CovEE]])
            if (CC==False):
                covmat = np.bmat([[CovTT, CovTE],
                                [CovTE.T, CovEE]])

            self.Cl_covmat = covmat
            self.Cl_covmat_computed = True
            return covmat

        self.Cl_covmat = CovTT
        self.Cl_covmat_computed = True
        return CovTT

    def test_fisher(self):
        """
        devise a test case that reproduces previously known (upto some tolerable accuracy)
        Fisher matrix computation as a check of this code
        """
        return 0
