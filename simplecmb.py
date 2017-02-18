# simple fisher information matrix computation in python
# implement a useful cosmology python class in doing so! :) ==> result: cosmo class in cosmology.py
# useful functions in functions.py

from cosmology.cosmoparams import Planck2015
from cmb import CMBExperiment, CMBFisher

planck15=Planck2015() # the default Planck 2013 cosmology
expt=CMBExperiment(name="Planck")
#cf=CMBFisher(expt, planck13,
#             params=["n", "r", "h", "Ob0", "A", "Om0", "tau"],
#             param_values=[0.96, 0.1, 0.7, 0.04, planck13.A, planck13.Om0, planck13.tau],
#             param_names=['$n_s$', '$r$', '$h$', '$\Omega_b$', '$A$', '$\Omega_m$', '$\tau$'])
#
cf=CMBFisher(expt, planck15,
             params=["n", "r", "A"],
             param_values=[0.96, 0.05, planck15.A],
             param_names=['$n_s$', '$r$', '$A$'])

cf.include_polarization=False
cf.fisher('tt')
#if (cf.prior_information_added):
#   print "prior information added!", cf.priors
cf.covariance()
