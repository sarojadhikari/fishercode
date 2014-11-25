"""
.. module:: survey
    :synopsis: define a class for a Large Scale Structure (LSS) survey
    
"""

from functions import degsq2rad

class survey:
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
        
