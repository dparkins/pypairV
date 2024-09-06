import numpy as np
import scipy.integrate as sp
from init import cosmo_params, power_params
from universal import Tgrowthfac
from growthq import fgQQ, growthini

# Constants
twopi3 = (2 * np.pi) ** 3

def pout_noh(akinput, z, cosmo, power):
    # This gives the power spectrum in unit of (Mpc)^3 <-- no h's 
    if power.ipvelLINEAR == 1:
        pout_noh = poutLINEARnoh(akinput, z,cosmo,power)
    else:
        pout_noh = poutNONLINEARnoh(akinput, z,cosmo,power)

    return pout_noh

def poutLINEAR(akinput, z, hI, twopi3):
    # this gives the linear power spectrum in unit of (Mpc/h)^3
    # where Δ = k^3 poutLINEAR / (2pi)^3
    # and where akinput is in (h/Mpc)

    ak = akinput * hI
    poutLINEAR = p(ak, z) * hI**3 * twopi3

    return poutLINEAR


def poutNONLINEAR(akinput, z, hI, twopi3):
    # this gives the nonlinear power spectrum in unit of (Mpc/h)^3
    # where Δ = k^3 poutNONLINEAR / (2pi)^3
    # and where akinput is in h/Mpc

    ak = akinput * hI
    poutNONLINEAR = pNL(ak, z) * hI**3 * twopi3

    return poutNONLINEAR


def poutLINEARnoh(ak, z, cosmo, power):
    # This gives the linear power spectrum in unit of (Mpc)^3 <-- no h's !
    # where Δ = k^3 poutLINEARnoh / (2pi)^3
    # and where ak is in 1/Mpc <-- no h !

    poutLINEARnoh = p(ak, z, cosmo, power) * twopi3

    return poutLINEARnoh


def poutNONLINEARnoh(ak, z):
    # This gives the nonlinear power spectrum in unit of (Mpc)^3 <-- no h's !
    # where Δ = k^3 poutNONLINEARnoh / (2pi)^3
    # and where ak is in 1/Mpc <-- no h !
    global pNL, pNLbig, ipNLinibigtable, twopi3  # Assuming these are defined elsewhere

    if ipNLinibigtable == 0:
        poutNONLINEARnoh = pNL(ak, z) * twopi3
    else:
        poutNONLINEARnoh = pNLbig(ak, z) * twopi3

    return poutNONLINEARnoh


def pini(cosmo, power):
    # normalize the power spectrum

    h00 = 100.0 * cosmo.hI
    akmax = h00 / 8.0
    power.pnorm = 1.0
    sig0 = 4 * np.pi * sp.quad(dsig8, 0.0, akmax, args=(cosmo, power), epsabs=1.0e-7)[0]
    power.pnorm = power.sigma8**2 / sig0
    print("pnorm is", power.pnorm)
    sig0 = 4 * np.pi * sp.quad(dsig8, 0.0, akmax, args=(cosmo, power), epsabs=1.0e-7)[0]
    print("sigma8 is", np.sqrt(sig0))
    return


def dsig8(ak, cosmo, power):
    # This function calculates the integrand for the normalization of the
    # power spectrum with Delta = 1 at r = 8 Mpc/h.
    h0 = 100.0 * cosmo.hI

    if ak <= 0.0:
        return 0.0

    # Window function for spherical tophat of radius 8 Mpc/h.
    x = ak * 800.0 / h0
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3)
    return ak**2 * p(ak, 0.0, cosmo, power) * w**2


def pNLini(z, cosmo, power):
    # initialize pNL

    # ak here is in Mpc^-1
    # pNL here is in Mpc^3
    # the pNL as well as p here is roman's p,
    # which is the same as Ed's p i.e. 4pi^3 P_roman
    # gives the right Delta

    power.akrpass = np.zeros(power.nplot)
    power.prpass = np.zeros(power.nplot)
    if power.iJMW == 1:
        # use Jain, Mo and White
        if cosmo.omegamI != 1.0:
            print("JMW cannot be used for omegam ne 1")
            raise SystemExit

        dlkp = np.log(power.akplotmaxP / power.akplotminP) / (power.nplot - 1)

        for i in range(power.nplot):
            ak = power.akplotminP * np.exp(np.float64(i) * dlkp)

            aknext = ak + power.dakf * ak
            aneff1 = (np.log(p(aknext, z)) - np.log(p(ak, z))) / (
                np.log(aknext) - np.log(ak)
            )
            aknext = ak + power.dakf * ak / 2.0
            aneff2 = (np.log(p(aknext, z)) - np.log(p(ak, z))) / (
                np.log(aknext) - np.log(ak)
            )

            if abs(aneff2 - aneff1) > power.antol:
                print(aneff2, aneff1, ak, aknext)
                print("should decrease dakf")
                raise SystemExit

            aneff = aneff2

            Bnn = ((3.0 + aneff) / 3.0) ** (1.3)

            DeltaL = p(ak, z) * 4.0 * np.pi * ak**3
            xparm = DeltaL / Bnn
            DeltaE = Bnn * phiJMW(xparm)
            akeff = ak * (1.0 + DeltaE) ** (1.0 / 3.0)
            PowerE = DeltaE / (4.0 * np.pi * (akeff**3))

            # akrpass[i] = akeff
            # prpass[i] = PowerE

            power.akrpass[i] = akeff
            power.prpass[i] = PowerE

    elif power.iPD == 1:
        # use Peacock and Dodds

        dlkp = np.log(power.akplotmaxP / power.akplotminP) / (power.nplot - 1)

        for i in range(power.nplot):
            ak = power.akplotminP * np.exp(np.float64(i) * dlkp)
            akslope = ak / 2.0
            aknext = akslope + power.dakf * akslope
            aneff1 = (np.log(p(aknext, z)) - np.log(p(akslope, z))) / (
                np.log(aknext) - np.log(akslope)
            )
            aknext = akslope + power.dakf * akslope / 2.0
            aneff2 = (np.log(p(aknext, z)) - np.log(p(akslope, z))) / (
                np.log(aknext) - np.log(akslope)
            )

            if abs(aneff2 - aneff1) > power.antol:
                print(aneff2, aneff1, ak, aknext)
                print("should decrease dakf")
                raise SystemExit

            aneff = aneff2

            DeltaL = p(ak, z) * 4.0 * np.pi * ak**3
            DeltaE = phiPD(DeltaL, aneff, cosmo)
            akeff = ak * (1.0 + DeltaE) ** (1.0 / 3.0)
            PowerE = DeltaE / (4.0 * np.pi * (akeff**3))

            # instead of ak, I stored ln(ak)
            power.akrpass[i] = np.log(akeff)

            # instead of storing power, I store ln (P).
            power.prpass[i] = np.log(PowerE)

    elif power.iSmith == 1:
        # this uses the halo fitting formula in Smith, Peacock et al.
        # first, need to search for aksigma, aneff, and Ceff, which are
        # all defined at sigma^2 (R) = 1, which is Gaussian smoothed.

        aksigma, aneff, Ceff = findksigma(z, cosmo, power)
        print("findaksigma", z, 1.0 / aksigma, aneff, Ceff)

        asmith = (
            1.4861
            + 1.8369 * aneff
            + 1.6762 * aneff**2
            + 0.7940 * aneff**3
            + 0.1670 * aneff**4
            - 0.6206 * Ceff
        )
        asmith = 10.0**asmith

        bsmith = 0.9463 + 0.9466 * aneff + 0.3084 * aneff**2 - 0.9400 * Ceff
        bsmith = 10.0**bsmith

        csmith = -0.2807 + 0.6669 * aneff + 0.3214 * aneff**2 - 0.0793 * Ceff
        csmith = 10.0**csmith

        gammasmith = 0.8649 + 0.2989 * aneff + 0.1631 * Ceff

        alphasmith = 1.3884 + 0.3700 * aneff - 0.1452 * aneff**2

        betasmith = 0.8291 + 0.9854 * aneff + 0.3401 * aneff**2

        amusmith = -3.5442 + 0.1908 * aneff
        amusmith = 10.0**amusmith

        anusmith = 0.9589 + 1.2857 * aneff
        anusmith = 10.0**anusmith

        if abs(cosmo.omegakI) < 1.0e-10:
            f1smith = cosmo.omegamI ** (-0.0307)
            f2smith = cosmo.omegamI ** (-0.0585)
            f3smith = cosmo.omegamI**0.0743
        else:
            print("need more work")
            raise SystemExit

        dlkp = np.log(power.akplotmaxP / power.akplotminP) / (power.nplot - 1)

        for i in range(power.nplot):
            ak = power.akplotminP * np.exp(np.float64(i) * dlkp)

            DeltaL = p(ak, z, cosmo, power) * 4.0 * np.pi * ak**3

            ay = ak / aksigma

            DeltaQ = (
                DeltaL
                * ((1.0 + DeltaL) ** betasmith / (1.0 + alphasmith * DeltaL))
                * np.exp(-(ay / 4.0 + (ay) ** 2 / 8.0))
            )

            DeltaHp = (
                asmith
                * (ay ** (3.0 * f1smith))
                / (
                    1.0
                    + bsmith * ay**f2smith
                    + (csmith * f3smith * ay) ** (3.0 - gammasmith)
                )
            )

            DeltaH = DeltaHp / (1.0 + amusmith / ay + anusmith / ay**2)

            DeltaE = DeltaQ + DeltaH
            PowerE = DeltaE / 4.0 / np.pi / (ak**3)
            power.akrpass[i] = np.log(ak)
            power.prpass[i] = np.log(PowerE)

    return


def p(ak, z, cosmo, power):
    #  p evaluates the linear power spectrum at wavenumber ak for
    #  expansion factor a=1.
    #  N.B. p is the 3-D spectral density and has units of 1/(ak*ak*ak).
    #  N.B. ak has units of 1/Mpc, _not_ h/Mpc.

    #  THE P is ED's P by
    #  P_ED = P_usual / (twopi)^3
    # i.e. 4pi k^3 P_ED is the right Delta
    # where 4pi k^3 / (twopi^3) P_usual is the right Delta

    #if ak <= 0.0:
    #    return 0.0

    if power.iGammaI == 1:
        omegahh = power.gammaeff * cosmo.hI
    else:
        omegahh = cosmo.omegamI * cosmo.hI * cosmo.hI

    q = ak / omegahh

    if power.iBBKS == 1:
        a1 = 2.34 * q
        a2 = 3.89 * q
        a3 = 16.1 * q
        a4 = 5.46 * q
        a5 = 6.71 * q
    elif power.iBBKS == 2:
        a1 = 2.205 * q
        a2 = 4.05 * q
        a3 = 18.3 * q
        a4 = 8.725 * q
        a5 = 8.0 * q
    elif power.iBBKS == 3:
        aEBW = 6.4 * q
        bEBW = 3.0 * q
        cEBW = 1.7 * q
        anuEBW = 1.13

    if power.iBBKS in [1, 2]:
        t = 1.0 + a2 + a3**2 + a4**3 + a5**4
        t = np.log(1.0 + a1) / a1 / np.sqrt(np.sqrt(t))
    elif power.iBBKS == 3:
        t = 1.0 + (aEBW + bEBW**1.5 + cEBW**2) ** anuEBW
        t = 1.0 / (t ** (1.0 / anuEBW))
    elif power.iBBKS == 4:
        t = transferEH(ak, power)

    if power.iBBKS == 2:
        if cosmo.omeganuI != 0:
            if cosmo.omegamI == 1.0:
                power.pnorm = (
                    2689.0
                    * (1.0 / 600.0 / cosmo.hI) ** (1.0 - power.an)
                    * (power.qrms / 18.0) ** 2
                    / (cosmo.hI**4 * (1.0 / (1.0 + power.ToS)))
                )
            else:
                print("only omegam eq 1 allowed for CHDM")
                raise ValueError("Invalid omegam value for CHDM")

    p = power.pnorm * ak ** (power.an - 4.0)

    if power.iBBKS == 2 and cosmo.omeganu != 0.0:
        d1 = 0.004321
        d2 = 2.217e-6
        d3 = 11.63
        d4 = 3.317
        qmdm = ak / cosmo.omeganu / cosmo.h / cosmo.h * np.sqrt(1.0 + z)
        mdmfac = (1.0 + d1 * qmdm ** (d4 / 2.0) + d2 * qmdm**d4) / (
            1.0 + d3 * (ak / cosmo.omeganu / cosmo.h / cosmo.h) ** d4
        )
        mdmfac = mdmfac ** (cosmo.omeganu**1.05)
        p *= mdmfac

    p *= t * t
    tpois = ak**2
    p *= tpois * tpois

    if cosmo.omegamI == 1.0:
        p /= (1.0 + z) ** 2
    else:
        p *= Tgrowthfac(z, cosmo, power) ** 2

    return p


def pNL(ak, z, cosmo, power):
    # There is actually no z dependence,
    # the z is determined when pNLini is called in pNLskew
    dktol = 1.0e-4
    akoriginal = ak
    ak = np.log(ak)

    pr = np.zeros(power.nplot)
    akr = np.zeros(power.nplot)

    for i in range(power.nplot):
        akr[i] = power.akrpass[i]
        pr[i] = power.prpass[i]

    iok = 0
    pNL = 0.0

    for i in range(power.nplot - 1):
        if ak >= akr[i] and ak <= akr[i + 1]:
            if (akr[i + 1] - akr[i]) > dktol:
                wL = (ak - akr[i]) / (akr[i + 1] - akr[i])
                wR = (akr[i + 1] - ak) / (akr[i + 1] - akr[i])
            else:
                wL = 0.0
                wR = 1.0
            pNL = pr[i] * wR + pr[i + 1] * wL
            iok = 1
            break

    if iok == 0:
        print("ak outside range of power table")
        print("ak asked for", ak)
        print("largest table ak", akr[power.nplot - 1])
        print("smallest table ak", akr[0])
        if ak < akr[0]:
            pNL = pr[0]
        elif ak > akr[power.nplot - 1]:
            pNL = pr[power.nplot - 1]

    pNL = np.exp(pNL)
    ak = akoriginal

    return pNL


def pNLinibig(z, izint, zkrpass, cosmo, power):
    # this is based on pNLini, the only difference is that
    # I save up all the z's at the same time.

    # initialize pNL

    # ak here is in Mpc^-1
    # pNL here is in Mpc^3
    # the pNL as well as p here is roman's p,
    # which is the same as Ed's p i.e. 4pi^3 P_roman
    # gives the right Delta

    zkrpass[izint] = z

    if power.iJMW == 1:
        # Use Jain Mo and White
        if cosmo.omegamI != 1.0:
            print("JMW cannot be used for omegam ne 1")
            raise SystemExit

        dlkp = np.log(power.akplotmaxP / power.akplotminP) / (power.nplot - 1)

        for i in range(power.nplot):
            ak = power.akplotminP * np.exp(np.float64(i) * dlkp)

            aknext = ak + power.dakf * ak
            aneff1 = (np.log(p(aknext, z)) - np.log(p(ak, z))) / (
                np.log(aknext) - np.log(ak)
            )
            aknext = ak + power.dakf * ak / 2.0
            aneff2 = (np.log(p(aknext, z)) - np.log(p(ak, z))) / (
                np.log(aknext) - np.log(ak)
            )

            if abs(aneff2 - aneff1) > power.antol:
                print(aneff2, aneff1, ak, aknext)
                print("should decrease dakf")
                raise SystemExit

            aneff = aneff2

            Bnn = ((3.0 + aneff) / 3.0) ** (1.3)

            DeltaL = p(ak, z) * 4.0 * np.pi * ak**3
            xparm = DeltaL / Bnn
            DeltaE = Bnn * phiJMW(xparm)
            akeff = ak * (1.0 + DeltaE) ** (1.0 / 3.0)
            PowerE = DeltaE / (4.0 * np.pi * (akeff**3))

            akrpass[i - 1][izint] = akeff
            prpass[i - 1][izint] = PowerE

    elif iPD == 1:
        # Use Peacock and Dodds
        dlkp = np.log(power.akplotmaxP / power.akplotminP) / (power.nplot - 1)

        for i in range(1, power.nplot + 1):
            ak = power.akplotminP * np.exp((i - 1) * dlkp)
            akslope = ak / 2.0
            aknext = akslope + dakf * akslope
            aneff1 = (np.log(p(aknext, z)) - np.log(p(akslope, z))) / (
                np.log(aknext) - np.log(akslope)
            )
            aknext = akslope + dakf * akslope / 2.0
            aneff2 = (np.log(p(aknext, z)) - np.log(p(akslope, z))) / (
                np.log(aknext) - np.log(akslope)
            )

            if abs(aneff2 - aneff1) > power.antol:
                print(aneff2, aneff1, ak, aknext)
                print("should decrease dakf")
                raise SystemExit

            aneff = aneff2

            DeltaL = p(ak, z) * 4.0 * np.pi * ak**3
            DeltaE = phiPD(DeltaL, aneff, omegam, omegav, omegak, omegaQ)
            akeff = ak * (1.0 + DeltaE) ** (1.0 / 3.0)
            PowerE = DeltaE / (4.0 * np.pi * (akeff**3))

            akrpass[i - 1][izint] = np.log(akeff)
            prpass[i - 1][izint] = np.log(PowerE)

    elif iSmith == 1:
        print("Smith et al power spectrum")
        print("not implemented for this option")
        print("Please set ipNLinibigtable to 0")

    return


def findksigma(z, cosmo, power):
    aksigma = 0.0
    aneff = 0.0
    Ceff = 0.0
    aneff2 = 0.0
    sig2G = 0.0
    sig2Gone = 0.0
    sig2G2 = 0.0
    aR = 1.0
    itry = 0
    itrymax = 1000
    fact = 1.01
    tolsig2G = 1.0e-4

    # find aksigma by bisection
    sig2G = sigmaGausscomp(aR, z, cosmo, power)

    itry = 0

    if sig2G > 1.0:
        aRL = aR
        # need to locate upper limit to aR
        while True:
            aR *= 2.0
            sig2G = sigmaGausscomp(aR, z, cosmo, power)
            if sig2G > 1.0 and itry < itrymax:
                itry += 1
            else:
                aRU = aR
                break
    else:
        aRU = aR
        # need to locate lower limit to aR
        while True:
            aR /= 2.0
            sig2G = sigmaGausscomp(aR, z, cosmo, power)
            if sig2G <= 1.0 and itry < itrymax:
                itry += 1
            else:
                aRL = aR
                break

    if itry >= itrymax:
        print("cannot bracket")
        print(itry)
        return

    # good. we have bracketed the range.
    itry = 0

    while True:
        aR = (aRL + aRU) / 2.0
        sig2G = sigmaGausscomp(aR, z, cosmo, power)

        if abs(sig2G - 1.0) < tolsig2G:
            aRone = aR
            sig2Gone = sig2G
            break

        if sig2G > 1.0 and itry < itrymax:
            aRL = aR
            itry += 1
        elif sig2G <= 1.0 and itry < itrymax:
            aRU = aR
            itry += 1

        if itry >= itrymax:
            print("too many its")
            return

    # aRone found, now compute aneff and Ceff
    aksigma = 1.0 / aRone
    aR = aRone * fact

    sig2G = sigmaGausscomp(aR, z, cosmo, power)

    aneff = -3.0 - (np.log(sig2G / sig2Gone)) / (np.log(aR / aRone))

    aR2 = aRone * fact * fact

    sig2G2 = sigmaGausscomp(aR2, z, cosmo, power)

    aneff2 = -3.0 - (np.log(sig2G2 / sig2G)) / (np.log(aR2 / aR))

    Ceff = (aneff2 - aneff) / (np.log(aR / aRone))

    return aksigma, aneff, Ceff


def sigmaGausscomp(aR, z, cosmo, power):
    aRpass = aR
    zpass = z

    neq = 1
    y = np.zeros(neq)
    c = np.zeros(24)
    work = np.zeros((neq, 9))
    tol = 1.0e-12
    xstart = power.akplotminP
    xend = 50.0 / aR

    ier = 0
    ind = 1
    y, err = sp.quad(derivsig2G, xstart, xend, args=(aR, z, cosmo, power))

    if ind < 0 or ier > 0:
        print(f"dverk error, ind, ier= {ind}, {ier}")
        print(f"whats {aR}, {z}, {y[0]}")

    sig2G = y * 4 * np.pi

    return sig2G


def derivsig2G(k, aRpass, z, cosmo, power):

    dydx = p(k, z, cosmo, power) * k * k * np.exp(-((k * aRpass) ** 2))

    return dydx


def transferEH(ak, power):
    """this implements Eisenstein and Hu's transfer function - the one with CDM and baryons, but no neutrinos i.e. astro-ph/9709112"""
    argtol = 1.0e-8

    qEH = ak / 13.41 / power.keqEH
    fEH = 1.0 / (1.0 + (ak * power.sEH / 5.4) ** 4)
    TcEH = fEH * T0tildeEH(ak, 1.0, power.betacEH, qEH) + (1.0 - fEH) * T0tildeEH(
        ak, power.alphacEH, power.betacEH, qEH
    )

    stildeEH = power.sEH / (1.0 + (power.betanodeEH / ak / power.sEH) ** 3) ** (
        1.0 / 3.0
    )

    arg = ak * stildeEH

    #f arg < argtol:
    #    j0tmp = 1.0
    #else:
    j0tmp = np.sin(arg) / arg

    TbEH = (
        T0tildeEH(ak, 1.0, 1.0, qEH) / (1.0 + (ak * power.sEH / 5.2) ** 2)
        + power.alphabEH
        / (1.0 + (power.betabEH / ak / power.sEH) ** 3)
        * np.exp(-((ak / power.ksilkEH) ** 1.4))
    ) * j0tmp

    transferEH = power.fracBEH * TbEH + power.fracCEH * TcEH

    return transferEH


def T0tildeEH(ak, a, b, qEH):
    """"""
    e = np.exp(1.0)
    CEH = (14.2 / a) + (386.0 / (1.0 + 69.9 * qEH**1.08))
    T0tildeEH = np.log(e + 1.8 * b * qEH) / (
        np.log(e + 1.8 * b * qEH) + CEH * qEH * qEH
    )

    return T0tildeEH


def initializeEH(cosmo, power):
    """"""

    Omegabh2EH = cosmo.omegabI * cosmo.hI**2
    OmegacEH = cosmo.omegamI - cosmo.omegabI
    Omega0h2EH = cosmo.omegamI * cosmo.hI**2

    power.fracBEH = cosmo.omegabI / cosmo.omegamI
    power.fracCEH = OmegacEH / cosmo.omegamI

    b1EH = 0.313 / (Omega0h2EH) ** 0.419 * (1.0 + 0.607 * Omega0h2EH**0.674)
    b2EH = 0.238 * Omega0h2EH**0.223
    zdEH = (
        1291.0
        * (Omega0h2EH) ** 0.251
        / (1.0 + 0.659 * (Omega0h2EH) ** 0.828)
        * (1.0 + b1EH * (Omegabh2EH) ** b2EH)
    )
    zeqEH = 2.5e4 * Omega0h2EH / cosmo.Theta27**4

    RdEH = 31.5 * Omegabh2EH / cosmo.Theta27**4 / (zdEH / 1.0e3)
    ReqEH = 31.5 * Omegabh2EH / cosmo.Theta27**4 / (zeqEH / 1.0e3)
    power.keqEH = 7.46e-2 * Omega0h2EH / cosmo.Theta27**2

    power.sEH = (
        2.0
        / 3.0
        / power.keqEH
        * (6.0 / ReqEH) ** 0.5
        * np.log(((1.0 + RdEH) ** 0.5 + (RdEH + ReqEH) ** 0.5) / (1.0 + ReqEH**0.5))
    )

    a1EH = (46.9 * Omega0h2EH) ** 0.67 * (1.0 + (32.1 * Omega0h2EH) ** (-0.532))

    a2EH = (12.0 * Omega0h2EH) ** 0.424 * (1.0 + (45.0 * Omega0h2EH) ** (-0.582))
    power.alphacEH = 1.0 / (a1EH ** (power.fracBEH) * a2EH ** ((power.fracBEH) ** 3))

    b1EH = 0.944 / (1.0 + (458.0 * Omega0h2EH) ** (-0.708))
    b2EH = 1.0 / (0.395 * Omega0h2EH) ** 0.0266

    power.betacEH = 1.0 / (1.0 + b1EH * ((power.fracCEH) ** b2EH - 1.0))

    power.alphabEH = (
        2.07
        * power.keqEH
        * power.sEH
        * (1.0 + RdEH) ** (-0.75)
        * GEH((1.0 + zeqEH) / (1.0 + zdEH))
    )

    power.ksilkEH = (
        1.6
        * (Omegabh2EH) ** 0.52
        * (Omega0h2EH) ** 0.73
        * (1.0 + (10.4 * Omega0h2EH) ** (-0.95))
    )

    power.betabEH = (
        0.5
        + (power.fracBEH)
        + (3.0 - 2.0 * power.fracBEH) * ((17.2 * Omega0h2EH) ** 2 + 1) ** 0.5
    )

    power.betanodeEH = 8.41 * (Omega0h2EH) ** 0.435

    return


def GEH(y):
    """"""
    GEH = y * (
        -6.0 * (1.0 + y) ** 0.5
        + (2.0 + 3.0 * y) * np.log(((1.0 + y) ** 0.5 + 1.0) / ((1.0 + y) ** 0.5 - 1.0))
    )
    return GEH


def initializePower(cosmo, power):

    if cosmo.omegaQI != 0.0:
        g = growthini(cosmo, power)

    if power.iBBKS == 4:
        out = initializeEH(cosmo, power)

    g = pini(cosmo, power)

    return


def main():

    cosmo_fid = cosmo_params("LCDM")
    power_fid = power_params("Original")
    # Initialize the power

    initializePower(cosmo_fid, power_fid)

    # Will do the nonlinear power spectrum later
    z = float(input("Input redshift: "))
    akinput = float(input("Input k in h divided by Mpc: "))

    ak = akinput * cosmo_fid.hI

    pNLini(z, cosmo_fid, power_fid)

    poutLINEAR = p(ak, z, cosmo_fid, power_fid) * cosmo_fid.hI**3 * twopi3
    poutNONLINEAR = pNL(ak, z, cosmo_fid, power_fid) * cosmo_fid.hI**3 * twopi3

    print("Output linear power:", poutLINEAR)
    print("Output nonlinear power:", poutNONLINEAR)


if __name__ == "__main__":
    main()
