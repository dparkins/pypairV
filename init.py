
import numpy as np

class cosmo_params(object):
	"""a class for holding the specific values of the cosmological parameters."""
	def __init__(self, label):
		self.label = label
		self.model = 'LCDM'
		self.omegamI = 0.2792 #omega_matter
		self.omegavI=0.0 #omega_lambda
		self.omegaQI=0.7208  #omega_darkenergy
		self.wQI=-1.0 #equation of state of dark energy
		self.wQpI=0.0 #derivative thereof
		self.hI=0.701 #Hubble constant in unit of 100 km/s/Mpc
		self.omegakI=1.0-self.omegamI-self.omegavI-self.omegaQI #omega_curvature
		self.hubble = 67.11
		self.iwmodeI=3 #iwmode controls parametrization of z dependence of w:
		# iwmode=1 means constant w=wQI, iwmode=2 means w=wQI + wQpI*z
		# iwmode=3 means w=wQI + (1-a)*wQpI
		self.omegabI=0.046 # Omegab is omega_baryon, for Eisenstein and Hu transfer function (iBBKS=4)
		self.Theta27=2.725/2.7 #the temperature of the CMB in unit of 2.7 Kelvin.
		self.omeganuI=0.0


class power_params(object):
	"""a class for holding parameters relating to the computation of the power spectra and correlation functions"""
	def __init__(self, label):
		self.label = label
		self.zmaxact=5.0 #This controls the maximum redshift of SNe of interest.
		#This affects the table of growth factor that we compute once for all.
		self.sigma8=0.817 #
		self.an=0.96 #
		self.antol=0.05 #antol is used for figuring out derivative of an itself.
		# This is used only when the nonlinear power spectrum is computed.
		# Don't worry about it otherwise (i.e. it has nothing to do with
		# the so called running of the spectral index).
		# iBBKS specifies the analytic transfer function.
		# We recommend iBBKS=4, the Hu & Eisenstein transfer function.
		# (iBBKS=1 is BBKS, iBBKS=2 is Ma [which works only for omegam=1, omegab=0.05], 
		#  iBBKS=3 is EBW)
		self.iBBKS=4
		# The following (iGammaI to ToS) are control parameters for the power spectrum
		# that are basically obsolete (they are only used for iBBKS .ne. 4).
		# # Set iGammaI = 1 if want to set gammaeff by hand (instead of = omegam*h)
		self.iGammaI=1
		self.gammaeff=0.21
		# The following is for CHDM models, using iBBKS=2.
		self.qrms=18.0
		self.ToS=0.0

		self.pnorm=1.0 #normaliation of power spectrum
		# Control parameters for nonlinear power spectrum*
		self.ipvelLINEAR=1 # Set ipvelLINEAR=1 if wants linear power spectrum, or to 0 if wants nonlinear
		# The following parameter is not used at all (unless I want to
		# precompute a table of nonlinear P as a function of ak and z). 
		self.zminact=1.0e-5
		# Same for following parameter.
		self.nzint=100
		# For pNLini: we recommend iSmith=1, iJMW=0, iPD=0, especially
		#             if baryon wiggles are present.
		self.dakf=0.0001

		self.iJMW=0 # Set iJMW=1 if want to use JMW power spectrum, otherwise to 0
		self.iPD=0 # Set iPD=1 if want to use PD power spectrum, otherwise to 0
		self.iSmith=1 # Set iSmith=1 if want to use Smith et al. power spectrum, otherwise to 0.
		self.ipNLinibigtable=0
		#This controls the number of k bins I used to compute the integrals over P(k).
		self.nplot=1000
		# This controls limits of integration for integral over P(k).
		self.akrpass=np.zeros(self.nplot)
		self.prpass=np.zeros(self.nplot)
		self.akplotmaxP=1.0e6
		self.akplotminP=1.0e-8
		# This should be k roughly where k*P(k) peaks. 
		# For linear P(k), kpeak is about 0.1, for nonlinear P(k), kpeak
		# ranges from 0.1 to 1.0. Therefore, to be safe, I choose akeffmaxmax=1.0d0. 
		# This helps determine up to what k I do the relevant integral over k.
		self.akeffmaxmax=1.0
		# This controls limit of k integration over P(k) for imethod=1
		# Using aklargeangfact=10.0 would be faster, and almost as accurate.
		# self.aklargeangfact=10.0
		self.aklargeangfact=25.0
		# This controls limit of k integration over P(k) for imethod=2
		self.aklargeangfact2=100.
		#growth parameters
		self.num_growth = 100
		self.z_growth = np.zeros(self.num_growth)
		# Pifunc and Sigfunc parameters
		# This controls size of table for Pifunc and Sigfunc (used if imethod=2)
		self.nPStab = 500
		self.Pifunctab = np.zeros(self.nPStab)
		self.Sigfunctab = np.zeros(self.nPStab)
		self.rtab = np.zeros(self.nPStab)
		# This controls the minimum and maximum r's in the table, in Mpc (no h's).
		# Should choose rPSmin to be fairly small, because I will approximate ar=0 with it.
		self.rPSmin=0.1
		self.rPSmax=1.e+4
		# parameters holding the coefficients for the Eisenstein and Hu's transfer function
		self.keqEH = 0. # equality k
		self.sEH = 0. # exponent
		self.alphacEH = 0. # alpha associated with CDM
		self.betacEH  = 0. # beta associated with CDM
		self.betanodeEH  = 0. # beta with the node
		self.alphabEH  = 0. # alpha associated with baryons
		self.betabEH  =  0. # beta associated with baryons
		self.ksilkEH  =  0. # silk damping scale
		self.fracBEH  =  0. # baryon fraction
		self.fracCEH  =  1. # CDM fraction



