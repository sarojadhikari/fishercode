"""
.. module:: cmb
    :synopsis: define a class for a CMB experiment

"""
import numpy as np
from fishercode import Fisher
import camb
from camb import model, initialpower

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
            self.fskyT = self.fskyE = self.fskyP = self.fskyC = self.fsky
            self.wT = np.infty
            self.wP = np.infty

        if name=="Planck": # this really is Planck hiL
            if not(alias):
                self.name="Planck"
            self.lmaxP=1996
            self.lmaxT=2508
            self.lmax = max(self.lmaxP, self.lmaxT)
            self.lmin = 30
            self.fsky=0.7
            self.fskyT = self.fskyE = self.fskyP = self.fskyC = self.fsky
            self.frequency=143
            self.theta=np.deg2rad(7./60.)   # in arcmin
            self.noiseppT=6.0  # noise per pixel for temperature in microK
            self.noiseppP=11.5    # noise per pixel for polarization in microK
            self.wT=1.0/(self.theta*self.noiseppT)**2.0 # units of 1/microK^2
            self.wP=1.0/(self.theta*self.noiseppP)**2.0

        self.lrangeT = range(self.lmin, self.lmaxT+1)
        self.lrangeE = range(self.lmin, self.lmaxP+1)
        self.lrangeC = self.lrangeE

        if (self.lmaxT<self.lmaxP):
            self.lrangeC = self.lrangeT

class CMBFisher(Fisher):
    """This class is for computing the CMB Fisher matrix given by

    .. math::

        F_{ij}=\\sum_l \\frac{(2l+1)}{2} \\frac{\\frac{\\partial C_l}{\\partial \\alpha_i} \\frac{\\partial C_l}{\\partial \\alpha_j}}{(C_l+w^{-1} e^{\\sigma^2 l^2})^2}

    The :math:`C_l` values are computed from the currently set cosmology.

    """

    def __init__(self, expt, cosmology, params=[], param_values=[], param_names=[], priors=[], pol=True):
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
        pars.set_cosmology(H0=self.cosmology.H0, ombh2=self.cosmology.Ob0*self.cosmology.h**2.0,
                           omch2=self.cosmology.Oc0*self.cosmology.h**2.0, omk=0,
                           tau=self.cosmology.tau, mnu=self.cosmology.m_nu[-1])
        pars.InitPower.set_params(As=self.cosmology.As, ns=self.cosmology.n, r=self.cosmology.r)
        pars.set_for_lmax(LMAX+200)

        if (self.cosmology.r > 0.0):
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
        v=getattr(self.cosmology, param)
        pv=param_value
        setattr(self.cosmology, param, pv*(1.+self.diff_percent))
        plus_value = self.theoryCls()

        setattr(self.cosmology, param, pv*(1.-self.diff_percent))
        minus_value = self.theoryCls()

        finite_diff=plus_value-minus_value
        delta_pv=2*self.diff_percent*pv

        # set the default value back
        setattr(self.cosmology, param, v)
        return (finite_diff)/delta_pv

    def noise_weight(self, ps='tt'):
        """return the noise weight for the power spectrum specified
        """
        if (self.experiment.name=="CVlimited"):
            return 0.
        if (ps=='tt'):
            return 1./self.experiment.wT
        elif (ps=='ee' or ps =='bb'):
            return 1./self.experiment.wP
        else:
            return 0.

    def noise_powers(self):
        """
        the noise power spectra
        """
        th = self.experiment.theta
        lrange = range(0, self.experiment.lmax+1) # start from zero for easy indexing
        Tnoise = np.array([self.noise_weight('tt')*np.exp((th*l)**2.0) for l in lrange])
        Enoise = np.array([self.noise_weight('ee')*np.exp((th*l)**2.0) for l in lrange])
        return np.array([Tnoise, Enoise])

    def fisher(self):
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
                fijl = dCij[i]@invcov@dCij[i]
                fmatrix[i][j] = dCij[i]@invcov@dCij[j]

        self.fisher_matrix = np.matrix(fmatrix)  # numpy array for easy indexing
        return self.fisher_matrix

    def FullCovarianceMatrix(self, turn_off_noise = False):
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
