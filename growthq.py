import numpy as np


def growthini(power):
    '''this program computes the growth factor for a quintessence universe
    '''
    # Constants and parameters
    neq = 1
    na = 1000
    zmin = 0.0
    zmax = power.zmaxact  # Assuming zmaxact is defined elsewhere

    amax = 1.0 / (1.0 + zmin)
    amin = 1.0 / (1.0 + zmax)
    da = (amax - amin) / (na - 1)

    # Initialize arrays
    y = np.zeros(neq)
    c = np.zeros(24)
    work = np.zeros((neq, 9))
    tol = 1.0e-12
    g = np.zeros(na)
    fg = np.zeros(na)
    zaa = np.zeros(na)

    # Open file for writing
    with open('growthq.dat', 'w') as f:
        for i in range(1, na + 1):
            aa = amin + da * (i - 1)

            astart = 0.0001
            aend = aa

            ier = 0
            ind = 1
            y[0] = astart

            # Call to dverk function (assumed to be defined elsewhere)
            ind, ier = dverk(neq, deriv, astart, y, aend, tol, ind, c, neq, work)

            if ind < 0 or ier > 0:
                print(f'dverk error, ind, ier= {ind}, {ier}')

            g[i - 1] = y[0]

            if i == na:
                g0 = y[0]

        for i in range(1, na + 1):
            fg[i - 1] = g[na - i]

        for i in range(1, na + 1):
            aa = amax - da * (i - 1)
            z = 1.0 / aa - 1.0
            g[i - 1] = fg[i - 1]
            fg[i - 1] = g[i - 1] / aa
            zaa[i - 1] = z
            f.write(f"{z} {g[i - 1] / aa} {g[i - 1] / g0}\n")

    # End of the program


    return

def fgQQ(z,growth_data):
    '''interpolate over range of growth values'''
    zatol = 1.0e-10
    iok = 0

    na = len(growth_data.zaa)

    for i in range(na - 1):
        if z >= zaa[i] and z <= zaa[i + 1]:
            if (zaa[i + 1] - zaa[i]) > zatol:
                wL = (z - zaa[i]) / (zaa[i + 1] - zaa[i])
                wR = (zaa[i + 1] - z) / (zaa[i + 1] - zaa[i])
            else:
                wL = 0.0
                wR = 1.0
            fgQQ_value = fg[i] * wR + fg[i + 1] * wL
            iok = 1
            break

    if abs(z - zaa[na - 1]) < 1.0e-5:
        fgQQ_value = fg[na - 1]
        iok = 1

    if iok == 0:
        print('z outside range of power table')
        print('z asked for', z)
        print('largest table z', zaa[na - 1])
        print('smallest table z', zaa[0])
        if z < zaa[0]:
            fgQQ_value = fg[0]
        elif z > zaa[na - 1]:
            fgQQ_value = fg[na - 1]

    return fgQQ_value

