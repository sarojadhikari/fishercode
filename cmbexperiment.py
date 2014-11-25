"""
.. module:: cmbexperiment
    :synopsis: define a class for a CMB experiment
    
"""

from functions import degsq2rad, deg2rad

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
        self.sigma=deg2rad(beamsize)
        self.wT=w   # (1/(pixel size*noise per pixel^2)
        self.name=name
        self.fsky=DeltaOmega/41253
        
        if (alias):
            self.name=alias
        if name==None:
            self.dOm = degsq2rad(DeltaOmega) # the survey area in degree square is converted to radians square here


