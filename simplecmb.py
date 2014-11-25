# simple fisher information matrix computation in python
# implement a useful cosmology python class in doing so! :) ==> result: cosmo class in cosmology.py
# useful functions in functions.py

from cosmology import cosmo
from cmbexperiment import CMBExperiment
from cmbfisher import CMBFisher 

planck13=cosmo() # the default Planck 2013 cosmology
expt=CMBExperiment()
expt.lmax=2000
cf=CMBFisher(expt, planck13, 
             params=["n", "r", "h", "Ob0", "A"], 
             param_values=[0.9, 0.7, 0.7, 0.04, planck13.A], 
             param_names=['$n_s$', '$r$', '$h$', '$\Omega_b$', '$A$'])
cf.include_polarization=True
cf.fisher()
#if (cf.prior_information_added):
#   print "prior information added!", cf.priors
cf.covariance()

