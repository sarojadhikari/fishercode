"""
.. module:: cmbexperiment
    :synopsis: define a class for a CMB experiment
    
"""

from functions import degsq2rad, deg2rad
import numpy as np

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
            self.fsky=0.7
            
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
            
