# Emiliano/Roman paper comparison saved file
from fisher import Fisher
import matplotlib.pyplot as plt
fs = Fisher()
#fs.load_fisher_matrix("fmtk_0.3Gpc_0.003_0.3_3456.txt")
fs.load_fisher_matrix("fmtk_3.8373Gpc_0.2248_0.17_2345.txt")
fs.parameters=["b1", "b2", "b3", "fNL"]
fs.params=fs.parameters
fs.param_names=["$b_1$", "$b_2$", "$b_3$", r"$g_{\rm NL}$"]
#fs.parameter_values=[1., 0., 0.0, 0.0]
fs.parameter_values=[1.95, 0.5, 0.0, 0.0]
fs.nparams=4

fs.covariance()
fs.plot_error_matrix([0,1,2,3], nstd=1)
#plt.text(-6.5, 5E6, r"$V=0.3 {\rm Gpc/h}$")
#plt.text(-6.5, 4E6, r"$\bar{n}=0.003 (h^{-1}{\rm Mpc})^{-3}$")
