import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(r'..')
from AtomFieldInt_V3 import c, eps0, hbar, a0, e, me, kB, amu, au, Rb, Cs, dipole

def heatRate(wl, P, w0, trapfreq, X=Rb):
    """heating rate in units of vibrational quanta per second
    wl - wavelength (m)
    P  - beam power (W)
    w0 - beam waist (m)
    trapfreq - atomic trap frequency (rad/s)
    X  - instance of atom() class from AtomFieldInt_V3"""
    atom = dipole(X.m, (0,1/2.,1,1), [wl, P, w0],
                    X.D0S, X.w0S, X.lwS, X.nljS,
                    nuclear_spin = X.I,
                    symbol=X.X)
    return atom.scatRate() * hbar * (2*np.pi/wl)**2 / 2 / X.m / trapfreq

# try to replicate Zhan group - https://arxiv.org/pdf/1902.04284.pdf
# wa = 2*np.pi * 27e3 # axial trap freq in rad/s
# wr = 2*np.pi * 165e3 # radial trap freq in rad/s
# print('Axial OP heating rate: ', heatRate(780.2e-9, 5e-6, 100e-6, wa, Rb))
# print('Radial OP heating rate: ', heatRate(780.2e-9, 5e-6, 100e-6, wr, Rb))
# print('Axial tweezer heating rate: ', heatRate(852e-9, 2e-3, 0.75e-6, wa, Rb))
# print('Radial tweezer heating rate: ', heatRate(852e-9, 2e-3, 0.75e-6, wr, Rb))
# print('Axial raman heating rate: ', heatRate(2*np.pi*c/(Rb5S.omega0[5]-2*np.pi*50e9), 150e-6, 80e-6, wa, Rb))
# print('Radial raman heating rate: ', heatRate(2*np.pi*c/(Rb5S.omega0[5]-2*np.pi*50e9), 150e-6, 80e-6, wr, Rb))

print('Axial tweezer heating rate: %.3g vibrational quanta / second'%heatRate(814e-9, 2.820e-3, 1.1e-6, 2*np.pi*30e3, Rb))