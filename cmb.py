"""
.. module:: cmb
    :synopsis: define a class for a CMB experiment

"""
import numpy as np
from fishercode import Fisher
import camb
from camb import model, initialpower

import multiprocessing as mp
from utilities.functions import degsq2rad, deg2rad

class CMBExperiment(object):
    """ CMB experiment class

        Args:
            * DeltaOmega: The sky covered in square degrees
            * LMAX: maximum multipole to be considered
            * theta: resolution (FWHM, arcmin)
            * sigmaT: temperature noise per pixel (:math:`\mu K`)
            * name: [None, "Planck", "WMAP", "PIXIE"] provides the possibility of using a known experiment

    """

    def __init__(self, DeltaOmega=41253., LMAX=2000, beamsize=5.0, w=8000., name=None, alias=None):
        """The CMB experiment class.

        """
        self.lmax=LMAX
        self.theta=deg2rad(beamsize/60.)    # beamsize is specified in arcmin
        self.wT=w   # (1/(pixel size*noise per pixel^2)
        self.name=name
        self.fsky=DeltaOmega/41253

        if (alias):
            self.name=alias
        if name==None:
            self.dOm = degsq2rad(DeltaOmega) # the survey area in degree square is converted to radians square here
        if name=="CVlimited":
            """define a cosmic variance limited CMB temperature and polarization experiment
            with the following properties
            """
            if not(alias):
                self.name="CVlimited"
            self.lmax=2500
            self.fsky=0.8
            # one does not need to worry about wT, wP, and noise per pixel; the weight for the
            # noise term is set to zero for this experiment in the CMBFisher class.

        if name=="Planck":
            if not(alias):
                self.name="Planck"
            self.lmax=2000
            self.fsky=0.7
            self.frequency=143
            self.theta=deg2rad(7./60.)   # in arcmin
            self.noiseppT=6.0  # noise per pixel for temperature in microK
            self.noiseppP=11.5    # noise per pixel for polarization in microK
            self.wT=1.0/(self.theta*self.noiseppT)**2.0
            self.wP=1.0/(self.theta*self.noiseppP)**2.0

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
        # using camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.cosmology.H0, ombh2=self.cosmology.Ob0*self.cosmology.h**2.0,
                           omch2=self.cosmology.Oc0*self.cosmology.h**2.0, omk=0,
                           tau=self.cosmology.tau, mnu=self.cosmology.m_nu[-1])
        pars.InitPower.set_params(As=self.cosmology.As, ns=self.cosmology.n, r=self.cosmology.r)
        pars.set_for_lmax(LMAX)

        if (self.cosmology.r > 0.0):
            pars.WantTensors = True

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars)
        totCL = powers['total']

        self.cosmology.TTCls = totCL[:LMAX+1,0]
        self.cosmology.ells=np.arange(LMAX+1)

        if (self.include_polarization):
            self.cosmology.TECls = totCL[:LMAX+1,3]
            self.cosmology.BBCls = totCL[:LMAX+1,2]
            self.cosmology.EECls = totCL[:LMAX+1,1]

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
        """
        v=getattr(self.cosmology, param)
        pv=param_value
        setattr(self.cosmology, param, pv*(1.+self.diff_percent))

        self.theoryCls(self.experiment.lmax)
        plus_value=self.getCls(ps)
        setattr(self.cosmology, param, pv*(1.-self.diff_percent))

        self.theoryCls(self.experiment.lmax)
        minus_value=self.getCls(ps)
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

    def fisherXX(self, ps, output):
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
                fijl=np.array([((2*l+1)/2)*(dCij[i][l]*dCij[j][l])/(ClXX[l]+self.noise_weight(ps)*np.exp((self.experiment.theta*l)**2.0))**2.0 for l in range(2, self.experiment.lmax-1)])
                fmatrix[i][j]=self.experiment.fsky*np.sum(fijl)

        output.put(np.array(np.matrix(fmatrix)))
        return np.array(np.matrix(fmatrix))  # numpy array for easy indexing

    def fisher(self, XX=['tt', 'te', 'ee', 'bb']):
        """sum over the specified XX=[TT, TE, EE, BB] fisher matrices to get the
        total CMB fisher matrix
        """
        if (('te' in XX) or ('ee' in XX) or ('bb' in XX)):
            self.include_polarization = True

        fmatrix=np.array([[0.]*self.nparams]*self.nparams)

        output=mp.Queue()
        processes=[mp.Process(target=self.fisherXX, args=(xx, output)) for xx in XX]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        fmats=[output.get() for p in processes]
        fmatrix=sum(fmats)

        self.fisher_matrix=np.matrix(fmatrix)
        return self.fisher_matrix


    def test_fisher(self):
        """
        devise a test case that reproduces previously known (upto some tolerable accuracy)
        Fisher matrix computation as a check of this code
        """
        return 0
