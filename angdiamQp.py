import numpy as np
import scipy.integrate as sp
from astropy.cosmology import LambdaCDM
import astropy.units as u

# This is based on angdiamQ.f, which assumed wQ=constant.
# Here, I want to allow for a time-varying wQ.
# What does that mean?
# It means that p = w rho.
# Recall that d(rho * a^3) = - p d(a^3)
# i.e. d(rho a^3) = - w*rho*(3 a^2 da)
# i.e. 3*rho*a^2*da + a^3*drho = -3*w*rho*a^2*da
# i.e. drho/rho = (-3*w*a^2*da - 3*a^2*da)/a^3
#               = -3*(w+1)*da/a
#
# if iwmode=1, w=constant, we have dln(rho) = dln(a^{-3(1+w)})
#              i.e. rho ~ a^-3(1+w)
#
# if iwmode=2, w=wQ+wQp*z, we have w=wQ+wQp*(1/a - 1)
#              w=(wQ-wQp) + wQp/a
#              therefore, dln(rho) = -3*[(1+wQ-wQp) + wQp/a]*da/a
#                                  = -3*(1+wQ-wQp)dlna - 3*wQp*da/a^2
#                                  = dln(a^-3*(1+wQ-wQp)) + d(3*wQp/a)
#                         dln(rho) = d(ln(a^-3*(1+wQ-wQp)) + 3*wQp/a)
#                         ln(rho) = ln(a^-3*(1+wQ-wQp)) + 3*wQp/a + constant
#                         rho ~ a^-3*(1+wQ-wQp) * exp(3*wQp/a)
#                             ~ (1+z)^3(1+wQ-wQp) * exp(3*wQp*z)   [the factor of exp(3*wQp) is just an overall constant]
#               the above agrees with expression given in Kim's 0304509.
#               it also has the nice property that rho(z=0) = 1.
#
# if iwmode=3, w=wQ+wQp*(1-a) = wQ+wQp - wQp*a
#              therefore, dln(rho) = -3*[(1+wQ+wQp) - wQp*a]*da/a
#              i.e. dln(rho) = d(lna^-3(1+wQ+wQp)) + d(3*wQp*a)
#                   ln(rho) = ln(a^-3(1+wQ+wQp)) + 3*wQp*a
#                   rho ~ a^-3(1+wQ+wQp) * exp(3*wQp*a)
#                       ~ (1+z)^3(1+wQ+wQp) * exp(3*wQp/(1+z)) * exp(-3*wQp)  [extra factor of exp(-3*wQp) is there to
#                                                                              make rho(z=0) = 1]
#                       ~ (1+z)^3(1+wQ+wQp) * exp(-3*wQp*z/(1+z))
#
# ------------------------------------------------------------------------------------------------
import numpy as np
from init import cosmo_params


def HzzQ(z, cosmo):
    """gives Hubble constant at z"""

    h00 = 100.0 * cosmo.hI

    if abs(cosmo.omegamI + cosmo.omegakI + cosmo.omegavI + cosmo.omegaQI - 1) > 1.0e-10:
        print("Omegas do not sum to one")
        print("error in HzzQ")
        print(cosmo.omegamI + cosmo.omegakI + cosmo.omegavI + cosmo.omegaQI)
        raise SystemExit

    if cosmo.iwmodeI == 1:
        hz = h00 * np.sqrt(
            cosmo.omegamI * (1.0 + z) ** 3
            + cosmo.omegakI * (1.0 + z) ** 2
            + cosmo.omegavI
            + cosmo.omegaQI * (1.0 + z) ** (3.0 * (1.0 + cosmo.wQI))
        )
    elif cosmo.iwmodeI == 2:
        hz = h00 * np.sqrt(
            cosmo.omegamI * (1.0 + z) ** 3
            + cosmo.omegakI * (1.0 + z) ** 2
            + cosmo.omegavI
            + cosmo.omegaQI
            * (
                (1.0 + z) ** (3.0 * (1.0 + cosmo.wQI - cosmo.wQpI))
                * np.exp(3.0 * cosmo.wQpI * z)
            )
        )
    elif cosmo.iwmodeI == 3:
        hz = h00 * np.sqrt(
            cosmo.omegamI * (1.0 + z) ** 3
            + cosmo.omegakI * (1.0 + z) ** 2
            + cosmo.omegavI
            + cosmo.omegaQI
            * (
                (1.0 + z) ** (3.0 * (1.0 + cosmo.wQI + cosmo.wQpI))
                * np.exp(-3.0 * cosmo.wQpI * z / (1.0 + z))
            )
        )

    return hz


def chiRQ(z, cosmo):
    """integrates chi directly
    see comments in rrchiQ

    this gives the radial comoving distance to an object at z
    where z is input in rrchi

    chi in unit of Mpc, if want h^-1 Mpc, then
    take chi and multiply by h."""

    if abs(cosmo.omegamI + cosmo.omegakI + cosmo.omegavI + cosmo.omegaQI - 1) > 1.0e-10:
        print("Omegas do not sum to one")
        print("error in chiRQ")
        print(cosmo.omegamI + cosmo.omegakI + cosmo.omegavI + cosmo.omegaQI)
        raise SystemExit

    # xstart = 0.0
    # xend = z
    # chi, err = sp.quad(derivsQ, xstart, xend, args=(cosmo))

    # chi *= 2.99792458e5
    astro = LambdaCDM(H0=cosmo.hI * 100.0, Om0=cosmo.omegamI, Ode0=cosmo.omegaQI)
    chi = astro.comoving_distance(z)
    return chi.value


def derivsQ(z, cosmo):
    """function to integrate under to get chi"""

    hz = HzzQ(z, cosmo)

    derivsQ = 1.0 / hz

    return derivsQ
