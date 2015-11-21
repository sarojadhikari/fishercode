import sys
from cosmology import cosmo
import bispectrum
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


st = datetime.now()
fidcosmo=cosmo()
fidcosmo.set_Om0(0.27)
fidcosmo.set_Ob0(0.047)
fidcosmo.f_baryon=0.17
fidcosmo.set_sigma8(0.79)


######## spherex (1412.4872) - Example ##########
#===============================================#
zmin = 0.1
zmax = 3.0
Nbins = 15
KMAX = 0.2
zstep = (zmax-zmin)/Nbins

def ng(z):
    ng1 = np.log10(3E-2)
    ng2 = np.log10(1.E-5)
    lin = ng2+(3.0-z)*(ng1-ng2)/(zmax - zmin)
    return np.power(10., lin)

Lbs=[1000.]
cnt = 0

for z in np.arange(zmin+zstep/2, zmax, zstep):
    Ls = fidcosmo.volume_factor(z, zstep)**(1./3)
    # select subvolumes according to Ls i.e. if the volume is large enough --- for example for larger z --- it maybe useful to make big subvolumes.
    if (z<1.4):
        Lbs = [200., 300.]
    elif (z<1.8):
        Lbs = [200., 1500.]
    else:
        Lbs = [200., 2000.]
    
    TotalSV = np.sum([4.*np.pi*np.power(Lbs[i]/2.0, 3.0) for i in range(len(Lbs))])
    VolumeRatio = int(np.power(Ls, 3.0)/TotalSV)  
    
    if VolumeRatio>=1.0:
        ngb = ng(z)
        survey=bispectrum.Survey(z=z, Lsurvey=Ls, ngbar=ngb, kmax=KMAX, Lboxes=Lbs)
        #bf=bispectrum.ibkLFisher(survey, fidcosmo, params=["b1", "b2", "fNL"], param_names=["$b_1$", "$b_2$", r"$f_{\rm NL}$"], param_values=[1.95, 0.5, 0.0])
        bf=bispectrum.itkLFisher(survey, fidcosmo, params=["b1", "b2", "b3", "fNL"], param_names=["$b_1$", "$b_2$", "$b_3$", r"$g_{\rm NL}$"], param_values=[1.95, 0.5, 0.0, 0.0])
        bf.fisher()
        
        if (cnt==0):
            total_fisher = bf.fisher_matrix * VolumeRatio
            cnt=cnt+1
        else:
            total_fisher = total_fisher + bf.fisher_matrix * VolumeRatio
            cnt=cnt+1
        
        print (cnt)
    else:
        print("sub-volumes larger than the redshift bin volume")
            
#np.savetxt("fmbk_SPHEREx_1.95_0.5_0.01_0.17_3.txt", total_fisher)
        
bf.fisher_matrix=total_fisher
np.savetxt("fmtk_SPHEREx_1.95_0.5_0.0_0.2_3.txt", total_fisher)
bf.covariance()

bf.plot_error_matrix([0,1,2,3])
plt.show()

sys.exit()

