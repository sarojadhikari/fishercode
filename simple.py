# simple fisher information matrix computation in python
# implement a useful cosmology python class in doing so! :) ==> result: cosmo class in cosmology.py
# useful functions in functions.py

from cosmology import cosmo
from cluster import survey
from cluster import ClusterFisher

s1=survey(name='eROSITA')

planck13=cosmo() # the default Planck 2013 cosmology
planck13.baryon_hmf_correction=1

cf=ClusterFisher(s1, planck13, 
                 params=['tau', 'n','fnl', 'sigma8', 'Mth', 'sigma_lnM', 'h'],
                 param_values=[planck13.tau, planck13.n, 10.0, 0.8, s1.Mth, s1.sigma_lnM, planck13.h], 
                 param_names=["$\\tau$", "$n_s$", "$f_{\\rm NL}$", "$\\sigma_8$","$M_{\\rm th}$", "$\\sigma_{\ln{M}}$", "$h$"], 
                 priors=[0, 0.05, 0, 0, 0.2, 0.2, 0.05])
#cf=ClusterFisher(params=['fnl','sigma8', 'Mth'], param_values=[20,0.8, s1.Mth], priors=[0, 0.02, 0.1]) 
#cf=ClusterFisher(s1, planck13, params=['fnl','sigma8'], param_values=[20,0.8], param_names=["$f_{\\rm NL}$", "$\\sigma_8"], priors=[0, 0.1]) 
s1.zmin=0.1
s1.zmax=0.5
s1.dlogM=0.2
cf.fisher()
if (cf.prior_information_added):
    print "prior information added!", cf.priors
cf.covariance()

