import numpy as np
from init import cosmo_params, power_params
from powerplot import initializePower, pNLini, poutLINEARnoh,poutNONLINEARnoh
from universal import Tgrowthfac,growthfac,phiPD
from angdiamQp import chiRQ, HzzQ
# This code is written by Lam Hui. When using it, please
# acknowledge the paper
# L. Hui, P. B. Greene, Phys. Rev. D73, 123526 (2006) 
# [arXiv:astro-ph/0512159].

# An explanatory note on the quantity computed in this code, Cv
# the magnitude covariance from peculiar motion, can be found
# in notes.pdf. This code also outputs xiV, the velocity
# two-point function in (km/s)^2, defined in notes.pdf.

# Tips for running this code:

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

# (2) Parameter: imethod, defined in snlens.inc

# If imethod=1, compute Cv and xiV using the observer-centric formula.
# If imethod=2, compute Cv and xiV using the separation-centric formula.
# imethod=2 is faster. Consistency between the two methods is
# a good check of the code.

# (3) Parameter: veldisp, defined in snlens.inc

# This define the velocity dispersion (km/s) at zero angular 
# separation.
# If one does not wish to set it by hand, but instead wants to
# compute it using the same code as for non-zero separations,
# then simply set veldisp=0.
# This option is introduced mostly because the recommended
# procedure is to run the code using linear theory (see below),
# in which case one might worry that the predicted velocity
# dispersion at zero separation is under-estimated.
# In reality, this does not seem to be a big problem.

# (4) Parameter: ipvelLINEAR, defined in snlens.inc

# Choose ipvelLINEAR=1 if wants to use linear theory (recommended),
# otherwise choose ipvelLINEAR=0 to use a nonlinear power spectrum.

# (5) Parameters: ireadPiSigtable and iwritePiSigtable, defined in 
#     snlens.inc

# When running this code for the first time, set ireadPiSigtable=0
# and iwritePiSigtable=0
# When running this code again with exactly the same cosmological
# parameters, but merely with different angular separations and
# redshifts, it would be much faster to set ireadPiSigtable=1
# and iwritePiSigtable=0, which takes advantage of a pre-computed
# table of xiV (which was written out when ireadPiSigtable=0 and
# iwritePiSigtable=1).

# (6) Parameter: zmaxact, defined in snlens.inc

# This sets the maximum redshift of interest. By default it is
# set to 5.0. Change to a higher number if one is interested in
# a higher redshift. This is used in pre-computing a table
# of growth factor.

# Universal parameters

cspeed=2.99792458e5 #speed of light

pi=3.1415926535898
twopi=2.0*pi
twopi2=twopi**2
twopi3=twopi**3
fourpi=2.0*twopi

# *Control parameters*

# If want to set the velocity dispersion at zero separation
# (i.e. i equals j) by hand, 
# then set the following quantity to be non-zero (in km/s).
# Otherwise, set to be 0.0d0.

veldisp=0.0
# veldisp=300.0

# This controls size of the table in growthq (number of steps in z).
na=1000

# This controls what angular separation (in radian) is small enough	
# that we will regard the pair to refer to the same SN.
dthetatol=pi/180./60./60.

# This controls size of table for Pifunc and Sigfunc (used if imethod=2)
nPStab=500
# This controls the minimum and maximum r's in the table, in Mpc (no h's).
# Should choose rPSmin to be fairly small, because I will approximate ar=0 with it.
rPSmin=0.1
rPSmax=1.0
# Set the following to 1 if want to read Pifunc and Sigfunc from an existing table
# (see comments in subroutine initpairVV). Otherwise set to 0.
ireadPiSigtable=0
# Set the following to 1 if want to write out such a table for later use.
iwritePiSigtable=1


#-----------------------------------------------------------------
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
def pairVV(RA1, DEC1, z1, RA2, DEC2, z2,cosmo,power,imethod):
	Cvfacsave = 0.0  # Initialize Cvfacsave
	veldisptol = 1.0e-3
	dztol = 1.0e-8
	if imethod == 1:
		dtheta = angsep(RA1, DEC1, RA2, DEC2)

		if ipair == 1:
			initializePower(cosmo,power)

		fac1, Dprimeoverc1, chi1 = cosmocompute(z1,cosmo)
		fac2, Dprimeoverc2, chi2 = cosmocompute(z2,cosmo)

		if power.ipvelLINEAR != 1:
			if ipair == 1:
				pNLini(0.0)

		if ipair == 1 and veldisp <= veldisptol:
			Cvfac = Cvintegrate0(power)
			Cvfacsave = Cvfac

		if dtheta < dthetatol and abs(z1 - z2) < dztol:
			if veldisp > veldisptol:
				Cv = (5.0 / np.log(10.0))**2 * fac1 * fac2 * (veldisp / cspeed)**2
				xiV = veldisp**2
				return Cv, dtheta, xiV
			else:
				Cvfac = Cvfacsave
	
		else:
			Cvfac = cvintegrate1(chi1, chi2, dtheta,power)

		Cv = (5.0 / np.log(10.0))**2 * fac1 * fac2 * Dprimeoverc1 * Dprimeoverc2 * Cvfac
		xiV = Dprimeoverc1 * Dprimeoverc2 * Cvfac * cspeed**2


	elif imethod == 2:
		dtheta = angsep(RA1, DEC1, RA2, DEC2)

		if ipair == 1:
			out = initializePower(cosmo,power)

		fac1, Dprimeoverc1, chi1 = cosmocompute(z1,cosmo)
		fac2, Dprimeoverc2, chi2 = cosmocompute(z2,cosmo)

		if power.ipvelLINEAR != 1:
			if ipair == 1:
				out = pNLini(0.0)

		if ipair == 1:
			Pifunctab, Sigfunctab, rtab = initpairVV(nPStab,rPSmin,rPSmax,power)

		angr1, angr2, rsep = angsep2(dtheta, chi1, chi2,cosmo)

		if dtheta < dthetatol and np.abs(z1 - z2) < dztol:
			if veldisp > veldisptol:
				Cv = (5.0 / np.log(10.0))**2 * fac1 * fac2 * (veldisp / cspeed)**2
				xiV = veldisp**2
				return Cv, dtheta, xiV



		Pifunc, Sigfunc = PiSig(Pifunctab, Sigfunctab, rtab,rsep,rPSmin,rPSmax)
		Cvfac = np.cos(angr1) * np.cos(angr2) * Pifunc + (np.cos(dtheta) - np.cos(angr1) * np.cos(angr2)) * Sigfunc
		Cv = (5.0 / np.log(10.0))**2 * fac1 * fac2 * Dprimeoverc1 * Dprimeoverc2 * Cvfac
		xiV = Dprimeoverc1 * Dprimeoverc2 * Cvfac * cspeed**2

	return Cv, dtheta, xiV

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

	cosSep = np.sin(theta1) * np.sin(theta2) * np.cos(alpha1 - alpha2) + np.cos(theta1) * np.cos(theta2)

	if cosSep > 1.0:
		print('warning cosSep greater than 1', cosSep)
	cosSep = 1.0

	dtheta = np.arccos(cosSep)
	dthetaEucl = np.sqrt((alpha1 - alpha2) ** 2 + (theta1 - theta2) ** 2)

	return dtheta, dthetaEucl


# This outputs fac, Dprimeoverc, chi for an input redshift z.
# fac = (1 - (a/a')*c/chi)  where a' = da/deta, eta=conformal time
# Dprimeoverc = D'/c in [Mpc]^{-1}, where D is the growth factor
# chi = radial comoving distance in Mpc (no factor of h).
def cosmocompute(z,cosmo):
	dfact = 1.001

	chi = chiRQ(z, cosmo)

	hz = HzzQ(z, cosmo)

	Dgrow = Tgrowthfac(z, cosmo)
	Dgrow2 = Tgrowthfac(z * dfact, cosmo)

	dDgrowdz = (Dgrow2 - Dgrow) / z / (dfact - 1.0)

	Dprime = -hz * dDgrowdz
	Dprimeoverc = Dprime / cspeed

# print('testing', z, Dgrow, dDgrowdz, Dprime)

	fac = 1.0 - cspeed * (1.0 + z) / chi / hz

	return fac, Dprimeoverc, chi




def cvintegrate1(chi1, chi2, dtheta, power, Nellmax=200):
	''' This computes Cvfac, given inputs chi1, chi2, dtheta
	    Cvfac is \int_0^\infty dk (dk/2\pi^2) P(k, z=0) 
	    \sum_0^\infty (2\ell + 1) j'_ell (kchi1) j'_ell (kchi2) P_ell (cos(theta))
		Nellmax controls the maximum ell that is summed to in method one.
		Roughly, one should choose Nellmax to be at least 1/dtheta.
	'''
	import scipy.special as sp

	if 1.0 / dtheta > Nellmax:
		print('warning')
		print('Nellmax might be too small')
		print(dtheta, Nellmax)

	Cvfac = 0.0
	for il in range(Nellmax):
		Cvfacpower = cvfacpowercompute(il, chi1, chi2,power)
		x = np.cos(dtheta)
		legenfac = sp.lpmv(0, il, x)
		Cvfac += (2.0 * il + 1.0) * Cvfacpower * legenfac

	return Cvfac

def cvfacpowercompute(il, chi1, chi2,power):
	'''This returns Cvfacpower given il, chi1, chi2
	Cvfacpower = \int_0^\infty (dk/2pi^2) P(k, z=0) j'_il (kchi1) j'_il (kchi2)'''
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
			puse = poutLINEARnoh(ak, 0.0)
		else:
			puse = poutNONLINEARnoh(ak, 0.0)

	x = ak * chi1
	sj, sjp1 = sp.spherical_jn(il, x)
	x = ak * chi2
	sj, sjp2 = sp.spherical_jn(il, x)

	Cvfacpower += ak * puse * sjp1 * sjp2

	Cvfacpower = Cvfacpower / (2.0 / np.pi / np.pi) * dlkp

	return Cvfacpower


def Cvintegrate0(power):
	'''This outputs Cvfac
	Cvfac = (1/3) \int (dk/2 pi^2) P(k,z=0), which comes from
	= \int (d^3 k /(2\pi^3)) (1/k^2) (cos(theta))^2 P(k,z=0)'''

	import scipy.integrate

	dlkp = np.log(power.akplotmaxP / power.akplotminP) / (power.nplot - 1)
	xstart = power.akplotminP * np.exp(dlkp)
	xend = power.akplotmaxP / np.exp(dlkp)

	tol = 1.0e-10
	ier = 0
	ind = 1
	y = np.zeros(1)
	c = np.zeros(24)
	work = np.zeros((1, 9))

	y = integrate.odeint(derive0, xstart, y, xend, tol, ind, c, 1, work, ier)

	if ind < 0 or ier > 0:
		print(f'dverk error, ind, ier= {ind}, {ier}')

	Cvfac = y[0] / (2.0 / np.pi / np.pi / 3.0)

	return Cvfac



def derive0(n, x, y,power):
	dydx = np.zeros(n)

	if power.ipvelLINEAR == 1:
		puse = poutLINEARnoh(x, 0.0)
	else:
		puse = poutNONLINEARnoh(x, 0.0)

	dydx[0] = puse

	return dydx
 
def angsep2(dtheta, chi1, chi2,cosmo):
	'''This outputs angr1, angr2 and rsept given dtheta,chi1 and chi2.
	where dtheta=angular separation between SNe 1 and 2 (in radians).
	chi1=comoving radial distance to SN 1 (in Mpc, no h factors).
	chi2=same to SN 2
	cos(angr1) = \hat x_1 \cdot \hat r, where \hat r is unit separation vector
	cos(angr2) = \hat x_2 \cdot \hat r
	rsep  = comoving separation between SNe 1 and 2.
	We will assume flat universe here. For nonflat unvierse, needs
	to modify code.'''
	rseptol = 1.0e-5
	omegakItol = 1.0e-8
	aminus1tol = 1.0e-8

	if abs(cosmo.omegak) > omegakItol:
		print('Need to modify code for nonflat univ')
		raise SystemExit

	chimin = chi1
	if chi2 < chimin:
		chimin = chi2

	if chimin <= 0.0:
		print('angsep2 fail for zero chi', chi1, chi2)
		raise SystemExit

	rsep2 = chi1**2 + chi2**2 - 2.0 * chi1 * chi2 * np.cos(dtheta)
	if rsep2 < (rseptol * chimin)**2:
		rsep = 0.0
		angr1 = np.pi / 2.0
		angr2 = np.pi / 2.0
		return angr1, angr2, rsep
	else:
		rsep = np.sqrt(rsep2)

	angr1 = (chi1**2 + rsep2 - chi2**2) / (2.0 * chi1 * rsep)

	if abs(angr1 + 1.0) < aminus1tol:
		angr1 = 0.0
	elif abs(angr1 - 1.0) < aminus1tol:
		angr1 = np.pi
	else:
		angr1 = np.pi - np.arccos(angr1)

	angr2 = (chi2**2 + rsep2 - chi1**2) / (2.0 * chi2 * rsep)

	if abs(angr2 + 1.0) < aminus1tol:
		angr2 = np.pi
	elif abs(angr2 - 1.0) < aminus1tol:
		angr2 = 0.0
	else:
		angr2 = np.arccos(angr2)

	return angr1, angr2, rsep


def initpairVV(nPStab,rPSmin,rPSmax,power):
	'''This outputs a table of Pifunc(r) and Sigfunc(r), defined in Eq. 10 of Hui & Frieman
	(but without the factors of D').
	Pifunc(r) = \int (dk/2pi^2) P(k,z=0) [ j_0(kr) - 2j_1(kr)/(kr) ]
	Sigfunc(r) = \int (dk/2pi^2) P(k,z=0) [j_1 (kr)/kr]

	Useful to know: j_0 (x) = 1 in the x --> 0 limit.
	j_1 (x) / x = 1/3 in the x --> limit.
	i.e. j_0 = (sin x)/x, j_1 = (sin x)/x^2 - (cos x)/x'''
	import scipy.special as sp
	Pifunctab = np.zeros(nPStab)
	Sigfunctab = np.zeros(nPStab)
	rtab = np.zeros(nPStab)

	if ireadPiSigtable == 0:
		# This means we need to compute the table anew.
		dlnr = np.log(rPSmax / rPSmin) / (nPStab - 1)

		for i in range(nPStab):
			rtab[i] = rPSmin * np.exp(dlnr * i)

			akeffmax = 1.0 / rtab[i]
			if akeffmax < power.akeffmaxmax:
				akeffmax = power.akeffmaxmax
			
			akplotmax = akeffmax * aklargeangfact2
			if akplotmax > power.akplotmaxP:
				akplotmax = power.akplotmaxP

			akplotmin = power.akplotminP

			dlkp = np.log(akplotmax / akplotmin) / (nplot - 1)

			sumPi = 0.0
			sumSig = 0.0

			for int in range(1, nplot):
				ak = akplotmin * np.exp(dlkp * int)

				if power.ipvelLINEAR == 1:
					puse = poutLINEARnoh(ak, 0.0)
				else:
					puse = poutNONLINEARnoh(ak, 0.0)

				x = ak * rtab[i]
				sj0, sjp0 = sp.spherical_jn(0, x)
				sj1, sjp1 = sp.spherical_jn(1, x)

				sumPi += ak * puse * (sj0 - 2.0 * sj1 / x)
				sumSig += ak * puse * sj1 / x

			Pifunctab[i] = sumPi / (2.0 * np.pi * np.pi) * dlkp
			Sigfunctab[i] = sumSig / (2.0 * np.pi * np.pi) * dlkp

	# This means we can just read Pifunc, Sigfunc, ar from an existing table
	elif ireadPiSigtable == 1:
		with open('PiSigtab', 'r') as f:
			for i in range(nPStab):
				Pifunctab[i], Sigfunctab[i], rtab[i] = map(float, f.readline().split())

	if ireadPiSigtable == 0 and iwritePiSigtable == 1:
		with open('PiSigtab', 'w') as f:
			for i in range(nPStab):
				f.write(f"{Pifunctab[i]} {Sigfunctab[i]} {rtab[i]}\n")

	return Pifunctab, Sigfunctab, rtab

#------------------------------------------------------------------
# 
def PiSig(Pifunctab, Sigfunctab, rtab, ar, rPSmin, rPSmax):
	"""This outputs Pifunc, Sigfunc given input ar (in Mpc, no h's).
	Pifunc and Sigfunc are defined in initpairVV, and are in Mpc^2."""
	if ar > rPSmax:
		print('need to increase rPSmax', ar, rPSmax)
		raise ValueError("rPSmax too small")

# Will treat any such ar as if ar~rPSmin.

	if ar <= rPSmin:
		Pifunc = Pifunctab[0]
		Sigfunc = Sigfunctab[0]
		return Pifunc, Sigfunc


	itry = int(np.log(ar / rPSmin) / (np.log(rPSmax / rPSmin) / (len(Pifunctab) - 1))) - 3

	if itry < 1:
		itry = 1

	iOK = 0
	for i in range(itry, len(Pifunctab) - 1):
		if rtab[i] < ar <= rtab[i + 1]:
			wL = np.log(ar) - np.log(rtab[i])
			wR = np.log(rtab[i + 1]) - np.log(ar)
			wT = np.log(rtab[i + 1] / rtab[i])
			Pifunc = Pifunctab[i] * wR + Pifunctab[i + 1] * wL
			Pifunc /= wT
			Sigfunc = Sigfunctab[i] * wR + Sigfunctab[i + 1] * wL
			Sigfunc /= wT
			iOK = 1
			return Pifunc, Sigfunc, rtab

	if iOK == 0:
		print('cannot locate ar in table')
		print(ar, rtab[0], rtab[-1])
		raise ValueError("ar not found in rtab")

	return Pifunc, Sigfunc


#---------------------------------------------------
if __name__ == "__main__":

	cosmo_fid = cosmo_params('LCDM')
	power_fid = power_params('Original')
	ioperatemode = 2
	# This controls which method to use for computing Cv.
	# imethod=1 is the observer-centric method.
	# imethod=2 is the separation-centric method.
	# method 2 is faster.
	imethod=2
	if ioperatemode == 1:
		RA1 = float(input('input RA for SN 1: '))
		DEC1 = float(input('input DEC for SN 1: '))
		z1 = float(input('input z for SN 1: '))
		RA2 = float(input('input RA for SN 2: '))
		DEC2 = float(input('input DEC for SN 2: '))
		z2 = float(input('input z for SN 2: '))
		ipair = 1
		Cv, dtheta, xiV = pairVV(RA1, DEC1, z1, RA2, DEC2, z2,cosmo_fid,power_fid,imethod)
		print('Cv and dtheta are', Cv, dtheta * 180.0 / pi)
		print('xiV is', xiV)

	elif ioperatemode == 2:
		SNname = []
		survey = []
		RA = []
		DEC = []
		z = []

		with open('table.input', 'r') as f1, open('table.input2', 'r') as f2:
			lines1 = f1.readlines()
			lines2 = f2.readlines()
			itotal = len(lines1)
			for i in range(0,itotal):
				line1 = lines1[i]
				c = line1.split()
				SNname.append(c[0])
				survey.append(c[1])
				line2 = lines2[i]
				d = line2.split()
				RA.append(float(d[0]))
				DEC.append(float(d[1]))
				z.append(float(d[2]))
				print(SNname[i], survey[i], RA[i], DEC[i], z[i])


		print('Total number of SNe', itotal)

		ipair = 0

		with open('table.output', 'w') as f3:
			for i in range(itotal):
				for j in range(i + 1):
					ipair += 1
					RA1 = RA[i]
					DEC1 = DEC[i]
					z1 = z[i]
					RA2 = RA[j]
					DEC2 = DEC[j]
					z2 = z[j]
					Cv, dtheta, xiV = pairVV(RA1, DEC1, z1, RA2, DEC2, z2,cosmo_fid,power_fid,imethod)

					f3.write(f"{Cv:.8e} {dtheta * 180.0 / pi:.8e} {SNname[i]} {RA[i]} {DEC[i]} {z[i]} {SNname[j]} {RA[j]} {DEC[j]} {z[j]} {i + 1} {j + 1} {xiV:.8e}\n")
					print(Cv, i + 1, j + 1)

	elif ioperatemode == 3:
		ifix = int(input('fixed redshift input 1 fixed angle input 2: '))

		if ifix == 1:
			zinput = float(input('input redshift: '))
			itotal = int(input('input number of separations: '))
			print('Reminder: if do not want veldisp at zero separation be set by hand')
			print('should put veldisp to 0 in snlens.inc')
			RA1 = 0.0
			dRA = (180.0 - RA1) / float(itotal - 1)

		elif ifix == 2:
			angleinput = float(input('input angular separation in degrees: '))
			itotal = int(input('input number of redshift separations: '))
			zinput = float(input('input lower redshift: '))
			zinputmax = float(input('input maximum redshift: '))
			dzout = (zinputmax - zinput) / float(itotal - 1)

	with open('Cv.output', 'w') as f3:
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
			Cv, dtheta, xiV = pairVV(RA1, DEC1, z1, RA2, DEC2, z2,cosmo_fid,power_fid,imethod)

			f3.write(f"{Cv} {dtheta * 180.0 / pi} {z2} {xiV}\n")

		f3.close()

