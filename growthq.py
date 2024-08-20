import numpy as np
import scipy.integrate as sp


def growthini(cosmo,power):
    '''this program computes the growth factor for a quintessence universe
    '''
    # Constants and parameters
    power.num_growth = 1000
    zmin = 0.0
    zmax = power.zmaxact  # Assuming zmaxact is defined elsewhere

    amax = 1.0 / (1.0 + zmin)
    amin = 1.0 / (1.0 + zmax)
    da = (amax - amin) / (power.num_growth - 1)

    # Initialize arrays
    g = np.zeros(power.num_growth)
    power.f_growth = np.zeros(power.num_growth)
    power.z_growth = np.zeros(power.num_growth)

    print(cosmo.omegamI,cosmo.omegakI, cosmo.omegavI, cosmo.omegaQI)
    # Open file for writing
    with open('growthq.dat', 'w') as f:
        for i in range(power.num_growth):
            aend = amin + da * np.float64(i)

            astart = 0.0001

            ln_a_init = np.log(astart)
            ln_a_final = np.log(aend)
            ln_delta, err = sp.quad(growthfacQ,ln_a_init,ln_a_final,args=(cosmo))
            delta = np.exp(ln_delta)
            g[i] = delta
            if i == power.num_growth-1:
                g0= delta

       
        for i in range(power.num_growth):
            power.f_growth[i] = g[power.num_growth - (i+1)]

        for i in range(power.num_growth):
            aa = amax - da * np.float64(i)
            z = 1.0 / aa - 1.0
            g[i] = power.f_growth[i]
            power.f_growth[i] = g[i] / (aa/astart)
            power.z_growth[i] = z
            f.write(f"{z} {g[i] / (aa/astart)} {g[i] / g0}\n")
            # second column is what PD called g(omega)
            # and third column is the growth factor normalized
            # to 1 at z=0



    # End of the program


    return

def deriv(n, x, y,cosmo):
    dydx = np.zeros(n)
    dydx[0] = growthfacQ(x,cosmo) * y[0] / x
    return dydx

def growthfacQ(ln_a,cosmo):
    if abs(cosmo.wQpI) > 1.0e-12:
        print('this code does not work for nonzero wp')
        raise SystemExit

    a = np.exp(ln_a)
    alpha = (3.0 / (5.0 - cosmo.wQI / (1.0 - cosmo.wQI)) + 
             3.0 / 125.0 * (1.0 - cosmo.wQI) * (1.0 - 3.0 * cosmo.wQI / 2.0) / 
             (1.0 - 6.0 * cosmo.wQI / 5.0)**3 * (1.0 - cosmo.omegamI))
    
    return omegamAA(a,cosmo) ** alpha

def omegamAA(a,cosmo):
    return cosmo.omegamI / (cosmo.omegamI + cosmo.omegakI * a + cosmo.omegavI * a**3 + cosmo.omegaQI / a**(3.0 * cosmo.wQI))



def fgQQ(z,power):
    '''interpolate over range of growth values'''
    zatol = 1.0e-10
    iok = 0


    for i in range(power.num_growth):
        if z >= power.z_growth[i] and z <= power.z_growth[i + 1]:
            if (power.z_growth[i + 1] - power.z_growth[i]) > zatol:
                wL = (z - power.z_growth[i]) / (power.z_growth[i + 1] - power.z_growth[i])
                wR = (power.z_growth[i + 1] - z) / (power.z_growth[i + 1] - power.z_growth[i])
            else:
                wL = 0.0
                wR = 1.0
            fgQQ_value = power.f_growth[i] * wR + power.f_growth[i + 1] * wL
            iok = 1
            break

    if abs(z - power.z_growth[power.num_growth - 1]) < 1.0e-5:
        fgQQ_value = power.f_growth[power.num_growth - 1]
        iok = 1

    if iok == 0:
        print('z outside range of power table')
        print('z asked for', z)
        print('largest table z', power.z_growth[na - 1])
        print('smallest table z', power.z_growth[0])
        if z < power.z_growth[0]:
            fgQQ_value = power.f_growth[0]
        elif z > power.z_growth[power.num_growth - 1]:
            fgQQ_value = power.f_growth[power.num_growth - 1]

    return fgQQ_value

