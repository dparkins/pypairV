import numpy as np
import pandas as pd
import sys
import time
from init import cosmo_params, power_params
from powerplot import initializePower, pNLini, poutLINEARnoh, poutNONLINEARnoh
from universal import Tgrowthfac, growthfac, phiPD
from angdiamQp import chiRQ, HzzQ

cspeed = 2.99792458e5  # speed of light

# This code is a python port of a code written by Lam Hui. When using it, please
# acknowledge the paper
# L. Hui, P. B. Greene, Phys. Rev. D73, 123526 (2006)
# [arXiv:astro-ph/0512159].

# When running this code for the first time, set ireadPiSigtable=0
# and iwritePiSigtable=0
# When running this code again with exactly the same cosmological
# parameters, but merely with different angular separations and
# redshifts, it would be much faster to set ireadPiSigtable=1
# and iwritePiSigtable=0, which takes advantage of a pre-computed
# table of xiV (which was written out when ireadPiSigtable=0 and
# iwritePiSigtable=1).

# Set the following to 1 if want to read Pifunc and Sigfunc from an existing table
# (see comments in subroutine initpairVV). Otherwise set to 0.
ireadPiSigtable = 0
# Set the following to 1 if want to write out such a table for later use.
iwritePiSigtable = 1


# -----------------------------------------------------------------
# This subroutine returns the magnitude covariance due to peculiar
# motion Cv (and dtheta) and the peculiar velocity
# two-point correlation xiV (Cv is defined in equation 4 and
# xiV is defined in equation 5 of notes.pdf.),
# for given RA, DEC and z of a pair of SNe or any other objects.
#
# For a precise definition of Cv. See notes.pdf
#
# The argument ipair just labels the pairs.
# Two different methods are allowed.
# Method 1 uses the observer-centric approach.
# Method 2 uses the separation-centric approach.
# The latter approach is faster if one needs to compute Cv
# for a large number of pairs.
#
# dtheta is in radians.
def pairVV(RA1, DEC1, z1, RA2, DEC2, z2, init, cosmo, power, imethod):
    Cvfacsave = 0.0  # Initialize Cvfacsave

    # This controls what angular separation (in radian) is small enough
    # that we will regard the pair to refer to the same SN.
    dthetatol = np.pi / 180.0 / 60.0 / 60.0
    veldisptol = 1.0e-3
    dztol = 1.0e-8
    if imethod == 1:
        dtheta = angsep(RA1, DEC1, RA2, DEC2)

        if init:
            initializePower(cosmo, power)

        fac1, Dprimeoverc1, chi1 = cosmocompute(z1, cosmo, power)
        fac2, Dprimeoverc2, chi2 = cosmocompute(z2, cosmo, power)

        if power.ipvelLINEAR != 1:
            if init:
                pNLini(0.0)

        angr1, angr2, rsep = angsep2(dtheta, chi1, chi2, cosmo)
        if init and power.veldisp <= veldisptol:
            Cvfac = Cvintegrate0(cosmo, power)
            Cvfacsave = Cvfac

        if dtheta < dthetatol and abs(z1 - z2) < dztol:
            if power.veldisp > veldisptol:
                Cv = (
                    (5.0 / np.log(10.0)) ** 2
                    * fac1
                    * fac2
                    * (power.veldisp / cspeed) ** 2
                )
                xiV = power.veldisp**2
                return Cv, dtheta, xiV
            else:
                Cvfac = Cvfacsave

        else:
            Cvfac = cvintegrate1(chi1, chi2, dtheta, cosmo, power)

        Cv = (
            (5.0 / np.log(10.0)) ** 2
            * fac1
            * fac2
            * Dprimeoverc1
            * Dprimeoverc2
            * Cvfac
        )
        xiV = Dprimeoverc1 * Dprimeoverc2 * Cvfac * cspeed**2

    elif imethod == 2:
        dtheta = angsep(RA1, DEC1, RA2, DEC2)

        if init:
            out = initializePower(cosmo, power)

        fac1, Dprimeoverc1, chi1 = cosmocompute(z1, cosmo, power)
        fac2, Dprimeoverc2, chi2 = cosmocompute(z2, cosmo, power)

        if power.ipvelLINEAR != 1:
            if init:
                out = pNLini(0.0)

        if init:
            initpairVV(cosmo, power)

        angr1, angr2, rsep = angsep2(dtheta, chi1, chi2, cosmo)

        Cv = np.zeros(len(z2))
        xiV = np.zeros(len(z2))
        Cv = np.where(
            (dtheta < dthetatol)
            & (np.abs(z1 - z2) < dztol)
            & (power.veldisp > veldisptol),
            (5.0 / np.log(10.0)) ** 2 * fac1 * fac2 * (power.veldisp / cspeed) ** 2,
            0.0,
        )
        xiV = np.where(
            (dtheta < dthetatol)
            & (np.abs(z1 - z2) < dztol)
            & (power.veldisp > veldisptol),
            power.veldisp**2,
            0.0,
        )

        # If the SNe are the same (tiny dtheta and dz) and you want to add a dispersion (disp > 0):
        # if dtheta < dthetatol and np.abs(z1 - z2) < dztol:
        #     if power.veldisp > veldisptol:
        #         Cv = (
        #             (5.0 / np.log(10.0)) ** 2
        #             * fac1
        #             * fac2
        #             * (power.veldisp / cspeed) ** 2
        #         )
        #         xiV = power.veldisp**2
        #         return Cv, dtheta, xiV, rsep
        # print(RA1, DEC1, RA2, DEC2, z1, z2, rsep)
        Pifunc, Sigfunc = PiSig(power, rsep)
        Cvfac = (
            np.cos(angr1) * np.cos(angr2) * Pifunc
            + (np.cos(dtheta) - np.cos(angr1) * np.cos(angr2)) * Sigfunc
        )
        Cv += (
            (5.0 / np.log(10.0)) ** 2
            * fac1
            * fac2
            * Dprimeoverc1
            * Dprimeoverc2
            * Cvfac
        )
        xiV += Dprimeoverc1 * Dprimeoverc2 * Cvfac * cspeed**2
        # Cv = (
        #     (5.0 / np.log(10.0)) ** 2
        #     * fac1
        #     * fac2
        #     * Dprimeoverc1
        #     * Dprimeoverc2
        #     * Cvfac
        # )
        # xiV = Dprimeoverc1 * Dprimeoverc2 * Cvfac * cspeed**2

    return Cv, dtheta, xiV, rsep


def angsep(ra1, dec1, ra2, dec2):
    idegree = 1
    degree2rad = np.pi / 180.0
    rad2min = 180.0 * 60.0 / np.pi
    amin2rad = 1.0 / rad2min

    if idegree == 1:
        alpha1 = ra1 * degree2rad
        theta1 = (90.0 - dec1) * degree2rad
        alpha2 = ra2 * degree2rad
        theta2 = (90.0 - dec2) * degree2rad
    else:
        alpha1 = ra1 * amin2rad
        theta1 = (90.0 - dec1) * amin2rad
        alpha2 = ra2 * amin2rad
        theta2 = (90.0 - dec2) * amin2rad

    cosSep = np.sin(theta1) * np.sin(theta2) * np.cos(alpha1 - alpha2) + np.cos(
        theta1
    ) * np.cos(theta2)

    if np.any(cosSep > 1.0):
        print("warning cosSep greater than 1", cosSep[np.where(cosSep > 1.0)])
        cosSep = np.where(cosSep > 1.0, 1.0, cosSep)

    dtheta = np.arccos(cosSep)
    dthetaEucl = np.sqrt((alpha1 - alpha2) ** 2 + (theta1 - theta2) ** 2)

    return dtheta


# This outputs fac, Dprimeoverc, chi for an input redshift z.
# fac = (1 - (a/a')*c/chi)  where a' = da/deta, eta=conformal time
# Dprimeoverc = D'/c in [Mpc]^{-1}, where D is the growth factor
# chi = radial comoving distance in Mpc (no factor of h).
def cosmocompute(z, cosmo, power):
    dfact = 1.001

    chi = chiRQ(z, cosmo)

    hz = HzzQ(z, cosmo)

    Dgrow = Tgrowthfac(z, cosmo, power)
    Dgrow2 = Tgrowthfac(z * dfact, cosmo, power)

    dDgrowdz = (Dgrow2 - Dgrow) / z / (dfact - 1.0)

    Dprime = -hz * dDgrowdz
    Dprimeoverc = Dprime / cspeed

    # print('testing', z, Dgrow, dDgrowdz, Dprime)
    fac = 1.0 - cspeed * (1.0 + z) / chi / hz

    return fac, Dprimeoverc, chi


def cvintegrate1(chi1, chi2, dtheta, cosmo, power, Nellmax=200):
    """This computes Cvfac, given inputs chi1, chi2, dtheta
    Cvfac is \int_0^\infty dk (dk/2\pi^2) P(k, z=0)
    \sum_0^\infty (2\ell + 1) j'_ell (kchi1) j'_ell (kchi2) P_ell (cos(theta))
        Nellmax controls the maximum ell that is summed to in method one.
        Roughly, one should choose Nellmax to be at least 1/dtheta.
    """
    import scipy.special as sp

    if 1.0 / dtheta > Nellmax:
        print("warning")
        print("Nellmax might be too small")
        print(dtheta, Nellmax)

    Cvfac = 0.0
    for il in range(Nellmax):
        Cvfacpower = cvfacpowercompute(il, chi1, chi2, cosmo, power)
        x = np.cos(dtheta)
        legenfac = sp.lpmv(0, il, x)
        Cvfac += (2.0 * il + 1.0) * Cvfacpower * legenfac

    return Cvfac


def cvfacpowercompute(il, chi1, chi2, cosmo, power):
    """This returns Cvfacpower given il, chi1, chi2
    Cvfacpower = \int_0^\infty (dk/2pi^2) P(k, z=0) j'_il (kchi1) j'_il (kchi2)"""
    import scipy.special as sp

    if il == 0:
        ileff = 1
    else:
        ileff = il

    akeffmax = ileff / chi1
    akeffmaxA = ileff / chi2
    if akeffmaxA > akeffmax:
        akeffmax = akeffmaxA
    if akeffmax < power.akeffmaxmax:
        akeffmax = power.akeffmaxmax

    akplotmax = akeffmax * power.aklargeangfact
    if akplotmax > power.akplotmaxP:
        akplotmax = power.akplotmaxP

    akplotmin = power.akplotminP

    dlkp = np.log(akplotmax / akplotmin) / (power.nplot - 1)

    Cvfacpower = 0.0

    for int in range(1, power.nplot):
        ak = akplotmin * np.exp(dlkp * int)

        if power.ipvelLINEAR == 1:
            puse = poutLINEARnoh(ak, 0.0, cosmo, power)
        else:
            puse = poutNONLINEARnoh(ak, 0.0, cosmo, power)

    x = ak * chi1
    sjp1 = sp.spherical_jn(il, x)
    x = ak * chi2
    sjp2 = sp.spherical_jn(il, x)

    Cvfacpower += ak * puse * sjp1 * sjp2

    Cvfacpower = Cvfacpower / (2.0 / np.pi / np.pi) * dlkp

    return Cvfacpower


def Cvintegrate0(cosmo, power):
    """This outputs Cvfac
    Cvfac = (1/3) \int (dk/2 pi^2) P(k,z=0), which comes from
    = \int (d^3 k /(2\pi^3)) (1/k^2) (cos(theta))^2 P(k,z=0)"""

    import scipy.integrate as sp

    dlkp = np.log(power.akplotmaxP / power.akplotminP) / (power.nplot - 1)
    xstart = power.akplotminP * np.exp(dlkp)
    xend = power.akplotmaxP / np.exp(dlkp)

    tol = 1.0e-10

    y = sp.quad(derive0, xstart, xend, args=(cosmo, power))[0]

    Cvfac = y / (2.0 / np.pi / np.pi / 3.0)

    return Cvfac


def derive0(x, cosmo, power):

    if power.ipvelLINEAR == 1:
        puse = poutLINEARnoh(x, 0.0, cosmo, power)
    else:
        puse = poutNONLINEARnoh(x, 0.0, cosmo, power)

    dydx = puse

    return dydx


def angsep2(dtheta, chi1, chi2, cosmo):
    """This outputs angr1, angr2 and rsept given dtheta,chi1 and chi2.
    where dtheta=angular separation between SNe 1 and 2 (in radians).
    chi1=comoving radial distance to SN 1 (in Mpc, no h factors).
    chi2=same to SN 2
    cos(angr1) = \hat x_1 \cdot \hat r, where \hat r is unit separation vector
    cos(angr2) = \hat x_2 \cdot \hat r
    rsep  = comoving separation between SNe 1 and 2.
    We will assume flat universe here. For nonflat unvierse, needs
    to modify code."""
    rseptol = 1.0e-5
    omegakItol = 1.0e-8
    aminus1tol = 1.0e-8

    if abs(cosmo.omegakI) > omegakItol:
        print("Need to modify code for nonflat univ")
        raise SystemExit

    chimin = np.where(chi2 < chi1, chi2, chi1)

    rsep2 = chi1**2 + chi2**2 - 2.0 * chi1 * chi2 * np.cos(dtheta)
    rsep = np.where(rsep2 < (rseptol * chimin) ** 2, 0.0, np.sqrt(rsep2))
    cosangr1_init = np.where(
        rsep2 < (rseptol * chimin) ** 2,
        0.0,  # Avoid NaN
        (chi1**2 + rsep2 - chi2**2) / (2.0 * chi1 * rsep),
    )

    cosangr1 = np.where(
        rsep2 < (rseptol * chimin) ** 2,
        0.0,  # zero separation, just set angle to pi/2
        np.where(
            abs(cosangr1_init + 1.0) < aminus1tol,
            -1.0,  # cos(angr1) = -1
            np.where(
                abs(cosangr1_init - 1.0) < aminus1tol,
                1.0,  # cos(angr1) = 1
                cosangr1_init,  # angle passes all criteria so leave as is
            ),
        ),
    )
    angr1 = np.pi - np.arccos(cosangr1)

    cosangr2_init = np.where(
        rsep2 < (rseptol * chimin) ** 2,
        0.0,  # Avoid NaN
        (chi2**2 + rsep2 - chi1**2) / (2.0 * chi2 * rsep),
    )
    cosangr2 = np.where(
        rsep2 < (rseptol * chimin) ** 2,
        0.0,  # zero separation, just set angle to pi/2
        np.where(
            abs(cosangr2_init + 1.0) < aminus1tol,
            -1.0,  # cos(angr2) = -1
            np.where(
                abs(cosangr2_init - 1.0) < aminus1tol,
                1.0,  # cos(angr2) = 1
                cosangr2_init,  # angle passes all criteria so leave as is
            ),
        ),
    )

    angr2 = np.arccos(cosangr2)
    # print(rsep, chi1, chi2, cosangr1_init, angr1, cosangr2_init, angr2)

    # chimin = chi1
    # if chi2 < chimin:
    #     chimin = chi2

    # if np.any(chimin <= 0.0):
    #     print("angsep2 fail for zero chi", chi1, chi2)
    #     raise SystemExit

    # rsep2 = chi1**2 + chi2**2 - 2.0 * chi1 * chi2 * np.cos(dtheta)

    # if rsep2 < (rseptol * chimin) ** 2:
    #     rsep = 0.0
    #     angr1 = np.pi / 2.0
    #     angr2 = np.pi / 2.0
    #     return angr1, angr2, rsep
    # else:
    #     rsep = np.sqrt(rsep2)

    # angr1 = (chi1**2 + rsep2 - chi2**2) / (2.0 * chi1 * rsep)
    # angr2 = (chi2**2 + rsep2 - chi1**2) / (2.0 * chi2 * rsep)

    # if abs(angr1 + 1.0) < aminus1tol:
    #     angr1 = 0.0
    # elif abs(angr1 - 1.0) < aminus1tol:
    #     angr1 = np.pi
    # else:
    #     angr1 = np.pi - np.arccos(angr1)

    # if abs(angr2 + 1.0) < aminus1tol:
    #     angr2 = np.pi
    # elif abs(angr2 - 1.0) < aminus1tol:
    #     angr2 = 0.0
    # else:
    #     angr2 = np.arccos(angr2)

    return angr1, angr2, rsep


def initpairVV(cosmo, power):
    """This outputs a table of Pifunc(r) and Sigfunc(r), defined in Eq. 10 of Hui & Frieman
    (but without the factors of D').
    Pifunc(r) = \int (dk/2pi^2) P(k,z=0) [ j_0(kr) - 2j_1(kr)/(kr) ]
    Sigfunc(r) = \int (dk/2pi^2) P(k,z=0) [j_1 (kr)/kr]

    Useful to know: j_0 (x) = 1 in the x --> 0 limit.
    j_1 (x) / x = 1/3 in the x --> limit.
    i.e. j_0 = (sin x)/x, j_1 = (sin x)/x^2 - (cos x)/x"""
    import scipy.special as sp

    if ireadPiSigtable == 0:
        # This means we need to compute the table anew.
        dlnr = np.log(power.rPSmax / power.rPSmin) / (power.nPStab - 1)

        for i in range(power.nPStab):
            power.rtab[i] = power.rPSmin * np.exp(dlnr * i)

            akeffmax = 1.0 / power.rtab[i]
            if akeffmax < power.akeffmaxmax:
                akeffmax = power.akeffmaxmax

            akplotmax = akeffmax * power.aklargeangfact2
            if akplotmax > power.akplotmaxP:
                akplotmax = power.akplotmaxP

            akplotmin = power.akplotminP

            dlkp = np.log(akplotmax / akplotmin) / (power.nplot - 1)

            sumPi = 0.0
            sumSig = 0.0

            for ii in range(1, power.nplot):
                ak = akplotmin * np.exp(dlkp * ii)

                if power.ipvelLINEAR == 1:
                    puse = poutLINEARnoh(ak, 0.0, cosmo, power)
                else:
                    puse = poutNONLINEARnoh(ak, 0.0, cosmo, power)

                x = ak * power.rtab[i]
                sj0 = sp.spherical_jn(0, x)
                sj1 = sp.spherical_jn(1, x)

                sumPi += ak * puse * (sj0 - 2.0 * sj1 / x)
                sumSig += ak * puse * sj1 / x

            power.Pifunctab[i] = sumPi / (2.0 * np.pi * np.pi) * dlkp
            power.Sigfunctab[i] = sumSig / (2.0 * np.pi * np.pi) * dlkp

    # This means we can just read Pifunc, Sigfunc, ar from an existing table
    elif ireadPiSigtable == 1:
        with open("PiSigtab", "r") as f:
            for i in range(power.nPStab):
                power.Pifunctab[i], power.Sigfunctab[i], power.rtab[i] = map(
                    float, f.readline().split()
                )

    if ireadPiSigtable == 0 and iwritePiSigtable == 1:
        with open("PiSigtab", "w") as f:
            for i in range(power.nPStab):
                f.write(f"{power.Pifunctab[i]} {power.Sigfunctab[i]} {power.rtab[i]}\n")

    return


# ------------------------------------------------------------------
#
def PiSig(power, ar):
    """This outputs Pifunc, Sigfunc given input ar (in Mpc, no h's).
    Pifunc and Sigfunc are defined in initpairVV, and are in Mpc^2."""
    if np.any(ar > power.rPSmax):
        print("need to increase rPSmax", np.max(ar), power.rPSmax)
        raise ValueError("rPSmax too small")

    # Will treat any such ar as if ar~rPSmin.

    # if ar <= power.rPSmin:
    #     Pifunc = power.Pifunctab[0]
    #     Sigfunc = power.Sigfunctab[0]
    #     return Pifunc, Sigfunc

    # itry = (
    #     int(
    #         np.log(ar / power.rPSmin)
    #         / (np.log(power.rPSmax / power.rPSmin) / (len(power.Pifunctab) - 1))
    #     )
    #     - 3
    # )

    # if itry < 1:
    #     itry = 1

    # iOK = 0
    # for i in range(itry, len(power.Pifunctab) - 1):
    #     if power.rtab[i] < ar <= power.rtab[i + 1]:
    #         wL = np.log(ar) - np.log(power.rtab[i])
    #         wR = np.log(power.rtab[i + 1]) - np.log(ar)
    #         wT = np.log(power.rtab[i + 1] / power.rtab[i])
    #         Pifunc = power.Pifunctab[i] * wR + power.Pifunctab[i + 1] * wL
    #         Pifunc /= wT
    #         Sigfunc = power.Sigfunctab[i] * wR + power.Sigfunctab[i + 1] * wL
    #         Sigfunc /= wT
    #         iOK = 1
    #         return Pifunc, Sigfunc
    #     elif power.rtab[0] < ar <= power.rtab[1]:
    #         print(ar)
    #         wL = np.log(ar) - np.log(power.rtab[0])
    #         wR = np.log(power.rtab[1]) - np.log(ar)
    #         wT = np.log(power.rtab[1] / power.rtab[0])
    #         Pifunc = power.Pifunctab[0] * wR + power.Pifunctab[1] * wL
    #         Pifunc /= wT
    #         Sigfunc = power.Sigfunctab[0] * wR + power.Sigfunctab[1] * wL
    #         Sigfunc /= wT
    #         iOK = 1
    #         return Pifunc, Sigfunc

    # if iOK == 0:
    #     print("cannot locate ar in table")
    #     print(ar, power.rtab[0], power.rtab[-1])
    #     raise ValueError("ar not found in rtab")

    Pifunc = np.interp(ar, power.rtab, power.Pifunctab)
    Sigfunc = np.interp(ar, power.rtab, power.Sigfunctab)

    return Pifunc, Sigfunc


# ---------------------------------------------------
# An explanatory note on the quantity computed in this code, Cv
# the magnitude covariance from peculiar motion, can be found
# in notes.pdf. This code also outputs xiV, the velocity
# two-point function in (km/s)^2, defined in notes.pdf.

if __name__ == "__main__":

    cosmo_fid = cosmo_params("LCDM")
    power_fid = power_params("Original")

    # This controls which method to use for computing Cv.
    # imethod=1 is the observer-centric method.
    # imethod=2 is the separation-centric method.
    # method 2 is faster.

    imethod = 2

    # (1) Parameter: ioperatemode, defined in the driver program pairV.

    # If ioperatemode=1, then the code is run in interative mode:
    # given RA,DEC,z for 2 SNe, the code outputs Cv
    # (i.e. magnitude covariance due to v)
    # and dtheta*180/pi (i.e. angular separation in degrees), and xiV.

    # If ioperatemode=2, then the code is run in non-interative mode:
    # the inputs are in table.input and table.input2, where
    # table.input has 2 columns: SN name, and survey name
    # table.input2 has 3 columns: RA, DEC and z.
    # The output is table.output, which consists of
    # Cv (i.e. magnitude covariance due to v), angular separation in degree,
    # SN1_name, RA1, DEC1, z1, SN2_name, RA2, DEC2, z2, 1_label, 2_label, xiV.

    # If ioperatemode=3, then the code is run to output a table
    # of Cv and xiV
    # as a function of angular separation, for a given input redshift.
    # This is useful for making a plot of Cv and xiV.
    ioperatemode = 2

    if ioperatemode == 1:
        RA1 = float(input("input RA for SN 1: "))
        DEC1 = float(input("input DEC for SN 1: "))
        z1 = float(input("input z for SN 1: "))
        RA2 = float(input("input RA for SN 2: "))
        DEC2 = float(input("input DEC for SN 2: "))
        z2 = float(input("input z for SN 2: "))
        ipair = 1
        Cv, dtheta, xiV, rsep = pairVV(
            RA1, DEC1, z1, RA2, DEC2, z2, ipair, cosmo_fid, power_fid, imethod
        )
        print("Cv and dtheta are", Cv, dtheta * 180.0 / np.pi)
        print("r-separation is ", rsep)
        print("xiV is", xiV)

    elif ioperatemode == 2:
        start = time.time()
        data = pd.read_csv(sys.argv[1], sep=" ")
        print(data.describe())
        print("Total number of SNe", len(data))
        triang = int(len(data) * (len(data) + 1) / 2)
        init = True
        RA1_out = []
        DEC1_out = []
        z1_out = []
        RA2_out = []
        DEC2_out = []
        z2_out = []
        Cv_out = []
        dtheta_out = []
        rsep_out = []
        xiV_out = []
        CID1_out = []
        CID2_out = []
        ID1_out = []
        ID2_out = []
        colspec = {
            "COV": "float64",
            "ANGSEP": "float64",
            "CID1": "object",
            "RA1": "object",
            "DEC1": "object",
            "z1": "object",
            "CID2": "object",
            "RA2": "object",
            "DEC2": "object",
            "z2": "object",
            "ID1": "int64",
            "ID2": "int64",
            "SEP": "float64",
            "CORR": "float64",
        }

        for i in range(len(data)):
            RA1_out = [*RA1_out, *[data["RA"][i]] * (i + 1)]
            DEC1_out = [*DEC1_out, *[data["DEC"][i]] * (i + 1)]
            z1_out = [*z1_out, *[data["z"][i]] * (i + 1)]
            CID1_out = [*CID1_out, *[data["CID"][i]] * (i + 1)]
            RA2_out = [*RA2_out, *data["RA"][: i + 1]]
            DEC2_out = [*DEC2_out, *data["DEC"][: i + 1]]
            z2_out = [*z2_out, *data["z"][: i + 1]]
            CID2_out = [*CID2_out, *data["CID"][: i + 1]]
            ID1_out = [*ID1_out, *[i + 1] * (i + 1)]
            ID2_out = [*ID2_out, *np.arange(i + 1) + 1]

            (
                Cv,
                dtheta,
                xiV,
                rsep,
            ) = pairVV(
                data["RA"][i],
                data["DEC"][i],
                data["z"][i],
                data["RA"][: i + 1],
                data["DEC"][: i + 1],
                data["z"][: i + 1],
                init,
                cosmo_fid,
                power_fid,
                imethod,
            )
            Cv_out = [*Cv_out, *Cv]
            dtheta_out = [*dtheta_out, *dtheta]
            xiV_out = [*xiV_out, *xiV]
            rsep_out = [*rsep_out, *rsep]

            init = False

        df_out = pd.DataFrame(
            data=np.array(
                [
                    Cv_out,
                    dtheta_out,
                    CID1_out,
                    RA1_out,
                    DEC1_out,
                    z1_out,
                    CID2_out,
                    RA2_out,
                    DEC2_out,
                    z2_out,
                    ID1_out,
                    ID2_out,
                    rsep_out,
                    xiV_out,
                ]
            ).T,
            columns=colspec.keys(),
        )
        df_out = df_out.astype(colspec)
        print(df_out.dtypes)
        df_out.to_csv(
            f"./{sys.argv[1][:sys.argv[1].find('.')]}_output.dat",
            index=False,
            float_format="%.4e",
            sep=" ",
        )
        print(time.time() - start)

    elif ioperatemode == 3:
        ifix = int(input("fixed redshift input 1 fixed angle input 2: "))

        if ifix == 1:
            zinput = float(input("input redshift: "))
            itotal = int(input("input number of separations: "))
            print("Reminder: if do not want veldisp at zero separation be set by hand")
            print("should put veldisp to 0 in snlens.inc")
            RA1 = 0.0
            dRA = (180.0 - RA1) / float(itotal - 1)

        elif ifix == 2:
            angleinput = float(input("input angular separation in degrees: "))
            itotal = int(input("input number of redshift separations: "))
            zinput = float(input("input lower redshift: "))
            zinputmax = float(input("input maximum redshift: "))
            dzout = (zinputmax - zinput) / float(itotal - 1)

        with open("Cv.output", "w") as f3:
            f3.write("COV DIST CID1 RA1 DEC1 z1 CID2 RA2 DEC2 z2 ID1 ID2 CORR\n")
            ipair = 0
            for i in range(itotal):
                if ifix == 1:
                    z1 = zinput
                    z2 = zinput
                    DEC1 = 0.0
                    DEC2 = DEC1
                    RA1 = 0.0
                    RA2 = RA1 + dRA * (i)
                elif ifix == 2:
                    z1 = zinput
                    z2 = z1 + dzout * (i)
                    DEC1 = 0.0
                    DEC2 = DEC1
                    RA1 = 0.0
                    RA2 = RA1 + angleinput

                ipair += 1
                Cv, dtheta, xiV, rsep = pairVV(
                    RA1, DEC1, z1, RA2, DEC2, z2, ipair, cosmo_fid, power_fid, imethod
                )

                f3.write(f"{Cv} {dtheta * 180.0 / np.pi} {z2} {xiV}\n")

            f3.close()