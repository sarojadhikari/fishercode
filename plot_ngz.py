import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update("")

zmin = 0.1
zmax = 3.0

def ng(z):
    ng1 = np.log10(3E-2)
    ng2 = np.log10(1.E-5)
    lin = ng2+(3.0-z)*(ng1-ng2)/(zmax - zmin)
    return np.power(10., lin)

zlist = np.arange(zmin, zmax, zmin/10.)
nlist = np.array([ng(z) for z in zlist])

plt.plot(zlist, nlist, "b-", linewidth=1.75, alpha=0.75)
plt.yscale('log')
plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"$\bar{n}_g \left({\rm Mpc/h}\right)^{-3}$")
plt.title("Assumed number density for SPHEREx")

plt.show()
