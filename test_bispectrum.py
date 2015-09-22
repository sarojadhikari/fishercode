import fishercode as fc

fidcosmo=fc.cosmo()
fidcosmo.set_Om0(0.27)
fidcosmo.set_Ob0(0.047)
fidcosmo.f_baryon=0.17385
fidcosmo.set_sigma8(0.79)
survey=fc.bispectrum.Survey()

bf=fc.bispectrum.ibkLFisher(survey, fidcosmo, params=["fNL", "b1"], param_names=["fNL", "b1"])
bf.fisher(skip=5.0)
