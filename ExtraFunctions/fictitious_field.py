"""Stefan Spence 04.07.19

estimate the displacement of the trap centre due to the vector 
light shift in a tweezer trap using the formula from 
B. Albrecht et al. PRA 94, 061401(R) (2016)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'Y:\Tweezer\Code\Python 3.5\polarisability')
sys.path.append(r'Z:\Tweezer\Code\Python 3.5\polarisability')
from AtomFieldInt_V3 import dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, kB, amu, Eh, au
# from scipy.constants import physical_constants
# muB = physical_constants['Bohr magneton']

wavelength = 1064e-9     # wavelength in m
power = 5e-3             # beam power in W
waist = 1.2e-6           # beam waist of tweezer in m
ehat = (2**(-0.5), 1j*2**(-0.5), 0) # E field circular polarisation


# create dipole objects for ground state Rb / Cs
bprop = [wavelength, power, waist, ehat]
Rb5S = dipole(Rb.m, (0,1/2.,1,1), bprop,  # Rb ground 5 S 1/2 state
                Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                nuclear_spin = Rb.I,
                symbol=Rb.X)
# Rb5S.gJ = 2.00233113 # Fine structure Lande g-factor (from Steck)

Cs6S = dipole(Cs.m, (0,1/2.,3,3), bprop, # Cs ground 6 S 1/2 state
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X)
# Cs6S.gJ = 2.00254032 # Fine structure Lande g-factor (from Steck)

def get_sep(atom=Rb5S, wl=wavelength):
    """Calculate the displacement of the trap centre due to the fictitious field"""
    aS, aV, aT = atom.polarisability(wl,split=True, HF=True) # vector polarisability for |F,MF>
    return aV/aS * atom.MF/atom.F * wl / 4. / np.pi # displacement of potential min.
