import numpy as np
from init import cosmo_params
from growthq import fgQQ

def phiPD(x,aneff,cosmo):
      APD = 0.482*(1.0+aneff/3.0)**(-0.947)
      BPD = 0.226*(1.0+aneff/3.0)**(-1.778)
      alphaPD = 3.310*(1.0+aneff/3.0)**(-0.244)
      betaPD = 0.862*(1.0+aneff/3.0)**(-0.287)
      VPD = 11.55*(1.0+aneff/3.0)**(-0.423)
  
      g=growthfac(0.0,cosmo)
      g3=g**3

      fac=1.0 + BPD*betaPD*x + (APD*x)**(alphaPD*betaPD)
      fac=fac/(1.0 + ( (APD*x)**alphaPD*g3/(VPD*x**0.5) )**betaPD)
      phiPD = x*(fac**(1.0/betaPD))

      return phiPD

def growthfac(z,cosmo):
      '''this gives an approximate growth factor (from Carroll et al. 
      ann. rev. of A. and A. 1992)
      put z to 0 to get what people call g (omega)
      note that this growthfactor is not normalized to 1 at z=0 !'''

      if np.abs(cosmo.omegamI+cosmo.omegavI+cosmo.omegakI+cosmo.omegaQI-1.0) >1.e-10:
#      if ((omegam+omegav+omegak+omegaQ).ne.1.0) then
         print('omegam omegav omegak do not sum to 1')
         raise SystemExit


      if cosmo.omegaQI == 0.0:
            AA=1.0/(1.0+z)
            AA3=AA**3

            omegamAA=cosmo.omegam/(AA+cosmo.omegam*(1.0-AA)+cosmo.omegav*(AA3-AA))
            omegavAA=AA3*cosmo.omegav/(AA+cosmo.omegam*(1.0-AA)+cosmo.omegav*(AA3-AA))
            growthfac=2.5/(1.0+z)*omegamAA/(omegamAA**(4.0/7.0)-omegavAA+(1.0+omegamAA/2.0)*(1.0+omegavAA/70.))
      else:
            growthfac=fgQQ(z)/(1.+z)

      return growthfac


def Tgrowthfac(z,cosmo):

      Tgrowthfac=growthfac(z,cosmo)/growthfac(0.,cosmo)

      return Tgrowthfac


def phiJMW(x):
      top=1.0+0.6*x+x*x-0.2*x**3-1.5*(x**3.5)+x**4
      bottom=1.0+0.0037*x**3
      
      phiJMW=x*(top/bottom)**0.5
      return phiJMW

