
from fisher import Fisher

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
        
    def ibkL_integrand(self, z, b1, b2, 
        

class Survey:
    """ large-scale structure survey class with fixed z but kmin/kmax and V specified
    """
    
    def __init__(self, z=0.57, kmax=0.17, b1fid=1.95, b2fid=0.5, fNLfid=0, Lboxes=[100., 200., 300., 400., 500., 600.]):
        self.z=z
        self.kmax=kmax
        self.b1fid=b1fid
        self.b2fid=b2fid
        self.fNLfid=fNLfid
        self.Lboxes=Lboxes
        self.Nsubvolumes = len(self.Lboxes)
        self.
    