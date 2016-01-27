from cosmology import cosmo
import bispectrum
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#import seaborn as sns
#sns.set_context("poster")

st = datetime.now()
fidcosmo=cosmo()
fidcosmo.set_Om0(0.27)
fidcosmo.set_Ob0(0.047)
fidcosmo.f_baryon=0.17
fidcosmo.set_sigma8(0.79)

#plt.style.use("ggplot")

######## spherex (1412.4872) - Example ##########
#===============================================#
zmin = 0.1
zmax = 3.0
Nbins = 30
KMAX = 0.2
zstep = (zmax-zmin)/Nbins

def ng(z):
    ng1 = np.log10(3E-2)
    ng2 = np.log10(1.E-5)
    lin = ng2+(3.0-z)*(ng1-ng2)/(zmax - zmin)
    return np.power(10., lin)

Lbs=[1000.]
TotalSV = np.sum([4.*np.pi*np.power(Lbs[i]/2.0, 3.0) for i in range(len(Lbs))])
cnt = 0

z=2.0
Ls = fidcosmo.volume_factor(z, zstep)**(1./3)
ngb = ng(z)
survey=bispectrum.Survey(z=z, Lsurvey=Ls, ngbar=ngb, kmax=KMAX, Lboxes=Lbs)
bf=bispectrum.itkLFisher(survey, fidcosmo, params=["b1", "b2", "b3", "fNL"], param_names=["$b_1$", "$b_2$", "$b_3$", r"$g_{\rm NL}$"], param_values=[1.95, 0.5, 0.1, 0.1])

kmin = 2.*np.pi/Lbs[0]

klist = np.arange(kmin, KMAX, kmin/10.)
itSPTcon = bf.itSPTcon(klist, 1.95)
itSPT = bf.itSPT(klist, 1.95)
itL2 = bf.itL2(klist, 1.95, 0.5)
itL3 = bf.itL3(klist, 1.95, 0.5)
itL4 = bf.itL4(klist, 1.95, 0.1)
itgNL = bf.itgNL(klist, 1.95, 100000)

LW=2.0

plt.plot(klist, itSPTcon, lw=LW,  label=r"$it_{L, c}^{(1)}$")
plt.plot(klist, itSPT, "--", lw=LW, label=r"$it_{L, d}^{(1)}$")
plt.plot(klist, itL2, ":", lw=LW, label=r"$it_L^{(2)}$")
plt.plot(klist, itL3, ls="dashdot", lw=LW, label=r"$it_L^{(3)}$")
plt.plot(klist, itL4, ls="dashed", lw=LW, label=r"$it_L^{(4)}$")
plt.plot(klist, itgNL, "-", lw=LW, label=r"$it_L^{\left(g_{\rm NL}^{\rm local}\right)}$")


plt.title(r"$L="+str(Lbs[0])+r"\,{\rm Mpc/h},\, z="+str(z)+r",\, g_{\rm NL}=10^5$")
plt.ylabel(r"reduced integrated trispectra, $it_L$")
plt.xlabel(r"$k\, ({\rm Mpc/h})^{-1}$")
plt.xscale('log')

plt.xlim(kmin, KMAX)
plt.legend(loc=0, ncol=2, framealpha=0.4)
plt.show()
