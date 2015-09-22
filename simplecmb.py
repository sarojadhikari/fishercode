# simple fisher information matrix computation in python
# implement a useful cosmology python class in doing so! :) ==> result: cosmo class in cosmology.py
# useful functions in functions.py

from cosmology import cosmo
from cmb.cmbexperiment import CMBExperiment
from cmb.cmbfisher import CMBFisher 

planck13=cosmo() # the default Planck 2013 cosmology
expt=CMBExperiment(name="Planck")
#cf=CMBFisher(expt, planck13, 
#             params=["n", "r", "h", "Ob0", "A", "Om0", "tau"], 
#             param_values=[0.96, 0.1, 0.7, 0.04, planck13.A, planck13.Om0, planck13.tau], 
#             param_names=['$n_s$', '$r$', '$h$', '$\Omega_b$', '$A$', '$\Omega_m$', '$\tau$'])
#
cf=CMBFisher(expt, planck13, 
             params=["n", "r", "A"], 
             param_values=[0.96, 0.05, planck13.A], 
             param_names=['$n_s$', '$r$', '$A$'])
             
cf.include_polarization=False
cf.fisher('tt')
#if (cf.prior_information_added):
#   print "prior information added!", cf.priors
cf.covariance()

