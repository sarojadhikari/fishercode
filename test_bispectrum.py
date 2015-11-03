import sys
from cosmology import cosmo
import bispectrum
import matplotlib.pyplot as plt
from datetime import datetime


st = datetime.now()
fidcosmo=cosmo()
fidcosmo.set_Om0(0.27)
fidcosmo.set_Ob0(0.047)
fidcosmo.f_baryon=0.17
fidcosmo.set_sigma8(0.79)

###### EXAMPLE 1 - Pos Dep Power Spectrum - 1504.03322 #########
#=====================BOSS FULL SURVEY=========================#
#Ls = (3.8373)**(1./3)*1000
#survey=bispectrum.Survey(Lsurvey=Ls, ngbar=0.2248, kmax=0.17, Lboxes=[200., 300., 400., 500.])
#bf=bispectrum.ibkLFisher(survey, fidcosmo, params=["b1", "b2", "fNL"], param_names=["$b_1$", "$b_2$", r"$f_{\rm NL}$"], param_values=[1.95, 0.5, 0.0])
#bf.fisher()
#bf.save_fisher_matrix("fmbk_3.8373Gpc_1.95_0.5_0.2248_0.17_2345.txt")
#bf.covariance()
#print (bf.fisher_matrix)
#bf.plot_error_matrix([0,1,2])
#plt.show()
#
#print ("time taken", datetime.now()-st)
##print bf.sigma_i(0), bf.sigma_i(1), bf.sigma_i(2)
#sys.exit()

#Ls = (3.8373)**(1./3)*1000
#survey=bispectrum.Survey(Lsurvey=Ls, ngbar=0.2248, kmax=0.17, Lboxes=[200., 300.])
#bf=bispectrum.itkLFisher(survey, fidcosmo, params=["b1", "b2", "fNL"], param_names=["$b_1$", "$b_2$", r"$g_{\rm NL}$"], param_values=[1.95, 0.5, 0.0])
#bf.fisher()
#bf.save_fisher_matrix("fmtk_new_3.8373Gpc_1.95_0.5_0.2248_0.17_23.txt")
#bf.covariance()
#print (bf.fisher_matrix)
#bf.plot_error_matrix([0,1,2])
#plt.show()
#
#print ("time taken", datetime.now()-st)
##print bf.sigma_i(0), bf.sigma_i(1), bf.sigma_i(2)
#sys.exit()

######## EXAMPLE 2 -  Chi-Ting's Thesis ########################
#=====================HETDEX===================================#
#Ls = (3.039)**(1./3)*1000
#survey=fc.bispectrum.Survey(Lsurvey=Ls, z=2.5, ngbar=0.5, kmax=0.3, Lboxes=[100., 200., 300., 400., 500., 600.])
#bf=fc.bispectrum.ibkLFisher(survey, fidcosmo, params=["b1", "b2", "fNL"], param_names=["$b_1$", "$b_2$", r"$f_{\rm NL}$"], param_values=[2.2, 0.67, 0.0])
#bf.fisher()
#bf.covariance()
#
#sys.exit()

####### EXAMPLE 3 - Pos Dep Bispectrum ##########
# ===============~SDSS Geometry from ================ #
Ls = (0.3)**(1./3)*1000
survey=bispectrum.Survey(Lsurvey=Ls, ngbar=0.003, kmax=0.2, Lboxes=[300., 400.])

#bf=fc.bispectrum.ibkLFisher(survey, fidcosmo, params=["b1", "b2", "fNL"], param_names=["$b_1$", "$b_2$", r"$f_{\rm NL}$"], param_values=[1.95, 0.5, 0.0])
bf=bispectrum.itkLFisher(survey, fidcosmo, params=["b1", "b2", "b3", "fNL"], param_names=["$b_1$", "$b_2$", "$b_3$", r"$g_{\rm NL}$"], param_values=[1., 0.0, 0.0, 0.0])

bf.fisher(skip=1.0) # skip should be 1 for now
bf.save_fisher_matrix("fmtk_new_0.2Gpc_0.003_0.3_34.txt")
bf.covariance()
bf.plot_error_matrix([0,1,2,3])
plt.show()
print (bf.sigma_i(0), bf.sigma_i(1), bf.sigma_i(2), bf.sigma_i(3))
print ("time taken", datetime.now()-st)
