"""
.. module:: cosmology
   :synopsis: cosmology definitions and basic computations
   
.. moduleauthor:: Saroj Adhikari <adh.saroj@gmail.com>

"""

import numpy as np
from scipy import integrate
from functions import top_hat, Hermite3

class cosmo(object):
    """ 
    define the cosmology and provide many methods to compute basic cosmological quantities 
    using the currently set cosmological parameters
    """
    Ob0=0.048252
    Om0=0.30712
    H0=67.77
    n=0.9611
    r=0.
    sigma8=0.8288
    tau=0.0952
    z_reion=11.52
    t0=13.7965
    Tcmb0=2.7255
    Neff=3.046
    flat=True
    m_nu=[0., 0., 0.06]
    f_baryon=0.014
    h=H0/100.
    A=1.0
    fnl=0.0    # cosmology class allows for the primordial skewness to be non-zero.
    
    def __init__(self):
        self.primordial_power_spectrum=self.pps
        self.normalize() # normalize the primordial amplitude A for the given sigma8
    
    def set_r(self, r):
        self.r=r

    def set_A(self, A):
        """set the amplitude A of the primordial power spectrum.
        Note that this will not change the value of sigma_8
        """
        self.A=A
    
    def set_sigma8(self, s8):
        self.sigma8=s8
        self.normalize()
    
    def set_fnl(self, fNL):
        self.fnl=fNL
    
    def set_Ob0(self, Ob):
        self.Ob0=Ob
    
    def set_Om0(self, Om):
        self.Om0=Om
    
    def set_n(self, ns):
        self.n=ns
    
    def set_h(self, h):
        self.h=h
    
    def set_tau(self, tau):
        self.tau=tau
                
    def rhom(self):
        """
        return the matter density for the current cosmology
        """
        mpc_to_cm=3.0856e24
        crit_dens=1.8791e-29*self.h*self.h*pow(mpc_to_cm,3.0) # in grams Mpc^{-3}
        M_sun=1.989e33 # in grams
        return crit_dens*self.Om0/(M_sun/self.h) # in M_sun/h Mpc^{-3}
        
    def pps(self, A, k, k0):
        """
        return the dimensionless primordial power spectrum value at a wave number k, 
        given amplitude A and the current cosmology, using pivot wavenumber k0
        
        .. math:: 
        
            \\mathcal{P}(k) = A \\left( \\frac{k}{k_0} \\right)^{n_s-1}
            
        """
        return  A*(k/k0)**(self.n-1)
    
    def power_spectrum0(self, A, k):
        """
        returns the matter power spectrum value at wave number k, given A at z=0
        """
        return A*(2.*np.pi**2.0)*k**self.n * (2998./self.h)**(3.+self.n)*(self.transfer_function(k)*self.growth_factor(0.0))**2.0
    
    def power_spectrum_bbks(self, A, k):
        """
        return the matter power spectrum using bbks transfer function
        """
        return A*(2.*np.pi**2.0)*k**self.n * (2998./self.h)**(3.+self.n)*(self.transfer_function_bbks(k)*self.growth_factor(0.0))**2.0
        
    def sigma_sq_integrand(self, k, R):
        return k*k/(2.0*np.pi**2.0)*self.power_spectrum0(self.A, k)*top_hat(k,R)**2.0
    
    def RtoM(self, R):
        """
        convert R in :math:`h^{-1}` Mpc to the corresponding M in :math:`h^{-1}M_\odot`
        """
        return 4.0*self.rhom()*np.pi*R*R*R/3.0
    
    def MtoR(self, M):
        """
        convert M in M_sun/h to the corresponding R in Mpc
        """
        return (3.0*M/4.0/self.rhom()/np.pi)**(1.0/3.0)
    
    def sigmaM(self, M):
        """
        """
        R=self.MtoR(M)
        return self.sigmaR(R)
    
    def sigmaM_M(self, M):
        """
        return the derivative sigmaM,M
        """
        plus_value=self.sigmaM(M*1.01)
        minus_value=self.sigmaM(M*0.99)
        finite_diff=plus_value-minus_value
        delta_M=0.02*M
        return (finite_diff/delta_M)
    
    def sigmaR(self, R):
        """
        compute sigma_R by integrating...
        """
        result=integrate.fixed_quad(self.sigma_sq_integrand, 0.0, 40./R, args=(R,))
        return np.sqrt(result[0])
    
    def normalize(self):
        """
        nomrmalize the amplitude of primordial fluctuations A so as to produce 
        the sigma8 of the current cosmology
        
        """
        self.A = self.A*(self.sigma8/self.sigmaR(8.0))**2.0
    
    def alpha(self, k,z):
        """ 
        return the product of transfer function and the growth factor at a wavenumber k and redshift
        z with other appropriate factors; alpha relates the primordial gravitational potential to the overdensity
        """
        c=299792.458 # speed of light in km/s
        return 2.0*k*k*self.transfer_function(k)*self.growth_factor(z)*c*c/(3.0*self.Om0*self.H0**2.0)
    
    def growth_factor_integrand(self, z):
        """
        return the growth factor integrand
        """
        hubblez=100*self.h*np.sqrt(self.Om0*pow(1+z,3.0)+(1-self.Om0))
        return (1+z)/pow(hubblez/(100.0*self.h),3.0)
        
    def growth_factor(self, z):
        """
        return the growth factor D(z)
        """
        if (z==0):
            return 0.78
        result=integrate.quad(self.growth_factor_integrand, z, 1000) # therefore only valid at z<<1000.
        hubblez=100*self.h*np.sqrt(self.Om0*pow(1+z,3.0)+(1-self.Om0))
        return (5.0*self.Om0*hubblez/(2*100.*self.h))*result[0]
    
    def E(self, z):
        """
        return the function :math:`E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}` 
        """
        return np.sqrt(self.Om0*(1+z)**3.0+(1-self.Om0))

    def comoving_distance_integrand(self, z):
        return 1./self.E(z)
    
    def comoving_distance(self, z):
        """
        return the comoving distance :math:`D(z) = \int \frac{c dt}{a}`
        as a function of the redshift
        """
        c=299792.458 # speed of light in km/s
        result=(c/self.H0)*integrate.quad(self.comoving_distance_integrand, 0.0, z)[0]
        return result
        
    
    def fnl2M3(self, fnl):
        return fnl*0.0003 # this is wealkly cosmology dependent -- ignore the cosmology dependence for now
    
    def nGMFfactor(self, M, gfratio=1.0):
        """
        return the factor with which to multiply the mass function if the cosmology is non-Gaussian
        """
        M3=self.fnl2M3(self.fnl)
        deltac=1.46
        nuc=deltac/(self.sigmaM(M)*gfratio)
        return (1.0+M3*(Hermite3(nuc))/6.0)
    
    def baryon_factor(self, M):
        """
        return the Vellisg et. al. factor for change in HMF due to their model:
        """
        return 1.0
    
    def volume_factor_integrand(self, z):
        """
        return the volume factor integrand
        """
        DH=3033 # in Mpc/h
        DC=DH*integrate.quad(self.E, 0.0, z)[0]
        DA=DC/(1+z)
        return DH*(1+z)**2.0*DA**2.0/self.E(z)
    
    def volume_factor(self, z, dz):
        """
        return the volume factor V_i
        """
        result=integrate.quad(self.volume_factor_integrand, z-dz/2.0, z+dz/2.0)
        return result[0]       

    def transfer_function_bbks(self, k):
        """
        return the fitting formula for transfer function by 
        Bardeen, Bond, Kaiser, and Szalay (1986, BBKS)
        
        Eq 7.71 of Dodelson
        """
        q=k/self.Om0/self.h
        return np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(16.2*q)**2.0+(5.47*q)**3.0+(6.71*q)**4.0)**(-0.25)
        
    def transfer_function(self, k):
        """
        return the transfer function T(k) for the current cosmology
        this python version was simply taken from the C code (tf_fit.c) for EH transfer function 
        by just making the formulae compatible with python/numpy.
        """
        # first set parameters as in TFset_parameters
        #=============================================
        Tcmb=self.Tcmb0
        theta_cmb=Tcmb/2.7
        f_baryon=self.f_baryon
        omhh=self.Om0*self.h**2.0
        obhh=omhh*f_baryon
        #h=self.h
        
        z_equality=2.50e4*omhh/theta_cmb**4.0
        k_equality=0.0746*omhh/theta_cmb**2.0
        
        z_drag_b1=0.313*omhh**(-0.419)*(1.+0.607*omhh**0.674)
        z_drag_b2=0.238*omhh**0.223
        z_drag=1291*(omhh**0.251)/(1+0.659*omhh**0.828)*(1+z_drag_b1*obhh**z_drag_b2)
        
        R_drag=31.5*obhh/theta_cmb**4.0*(1000/(1+z_drag))
        R_equality=31.5*obhh/theta_cmb**4.0*(1000/z_equality)
        
        sound_horizon=2./3./k_equality*np.sqrt(6./R_equality)*np.log((np.sqrt(1+R_drag)+np.sqrt(R_drag+R_equality))/(1+np.sqrt(R_equality)))
        
        k_silk = 1.6*pow(obhh,0.52)*pow(omhh,0.73)*(1+pow(10.4*omhh,-0.95))
        
        alpha_c_a1=(46.9*omhh)**0.670*(1+(32.1*omhh)**(-0.532))
        alpha_c_a2=(12.0*omhh)**0.424*(1+(45.0*omhh)**(-0.582))
        alpha_c=alpha_c_a1**(-f_baryon)*(alpha_c_a2)**(-f_baryon**3.0)
        
        beta_c_b1 = 0.944/(1+pow(458*omhh,-0.708))
        beta_c_b2 = pow(0.395*omhh, -0.0266)
        beta_c = 1.0/(1+beta_c_b1*(pow(1-f_baryon, beta_c_b2)-1))

        y = z_equality/(1+z_drag);
        alpha_b_G = y*(-6.*np.sqrt(1+y)+(2.+3.*y)*np.log((np.sqrt(1+y)+1)/(np.sqrt(1+y)-1)))
        alpha_b = 2.07*k_equality*sound_horizon*pow(1+R_drag,-0.75)*alpha_b_G

        beta_node = 8.41*pow(omhh, 0.435)
        beta_b = 0.5+f_baryon+(3.-2.*f_baryon)*np.sqrt(pow(17.2*omhh,2.0)+1)

        #k_peak = 2.5*3.14159*(1+0.217*omhh)/sound_horizon
        #sound_horizon_fit = 44.5*np.log(9.83/omhh)/np.sqrt(1+10.0*pow(obhh,0.75))

        #alpha_gamma = 1-0.328*np.log(431.0*omhh)*f_baryon + 0.38*np.log(22.3*omhh)*f_baryon**2.0;
        
        # the TFfit_onek code starts from here
        # ====================================        
        q = k/13.41/k_equality
        xx = k*sound_horizon

        T_c_ln_beta = np.log(2.718282+1.8*beta_c*q)
        T_c_ln_nobeta = np.log(2.718282+1.8*q)
        T_c_C_alpha = 14.2/alpha_c + 386.0/(1+69.9*pow(q,1.08))
        T_c_C_noalpha = 14.2 + 386.0/(1+69.9*pow(q,1.08))

        T_c_f = 1.0/(1.0+pow(xx/5.4, 4.0))
        T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*(q*q)) + (1-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*(q*q))
    
        s_tilde = sound_horizon*pow(1+pow(beta_node/xx, 3.0),-1./3.)
        xx_tilde = k*s_tilde

        T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*(q*q));
        T_b = np.sin(xx_tilde)/(xx_tilde)*(T_b_T0/(1+pow(xx/5.2,2.0))+
		alpha_b/(1+pow(beta_b/xx,3.0))*np.exp(-pow(k/k_silk,1.4)));
    
        f_baryon = obhh/omhh;
        T_full = f_baryon*T_b + (1-f_baryon)*T_c;
        return T_full

