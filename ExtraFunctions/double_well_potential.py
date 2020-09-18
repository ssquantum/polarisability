import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.style.use(r'Z:\Tweezer\People\Stefan\DUdefault.mplstyle')
# use JQC official colour scheme
sys.path.append(r'Z:\Tweezer\People\Vincent\python snippets\plotting_tools')
import default_colours
sys.path.append(r'Z:\Tweezer\Code\Python 3.5\polarisability')
from AtomFieldInt_V3 import (dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au)

waist = 1.16e-6
Cs6S = dipole(Cs.m, (0,1/2.,4,4), [938e-9, 4e-3, waist], 
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X)

sep = np.linspace(0.7, 4, 100)*1e-6    # initial separation of the tweezer traps
xs = np.linspace(-1.5,5, 200)*1e-6    # positions along the beam axis
barrier = np.zeros(len(sep))
minU = np.zeros(len(sep))
for i, s in enumerate((sep)):
    U = (Cs6S.acStarkShift(xs,0,0,mj=0) + Cs6S.acStarkShift(xs-s,0,0,mj=0))/kB*1e3 
    minU[i] = np.min(U)
    U0 = Cs6S.acStarkShift(xs,0,0,mj=0)/kB*1e3         # potential in the 1064 trap
    U1 = Cs6S.acStarkShift(xs-s,0,0,mj=0)/kB*1e3 # potential in the 880 trap
    try:
        i0, i1 = find_peaks(-U)[0]
        barrier[i] = np.max(U[i0:i1+1]) - np.min(U)
    except Exception as e:
        print(e)
    
plt.figure()
plt.plot(sep*1e6, barrier)
plt.xlabel('Separation of trap centres ($\mu$m)')
plt.ylabel('Barrier height (mK)')
plt.show()
plt.figure()
plt.plot(sep/waist, -barrier/np.min(U0), label='Numerical')
plt.plot(sep/waist, minU/np.min(U0)-2*np.exp(-sep**2/2/waist**2), label='Analytical')
plt.legend()
plt.xlabel('Separation of trap centres / beam waist')
plt.ylabel('Barrier height / single trap depth')
plt.show()

plt.figure()
plt.plot(xs*1e6, U, 'k')
plt.plot(xs*1e6, U0, color=default_colours.DUcherry_red, alpha=0.6)
plt.plot(xs*1e6, U1, color=default_colours.DUsea_blue, alpha=0.6)
plt.title('%.2g'%(sep[-1]*1e6))
plt.xlabel(r'Position ($\mu$m)')
plt.ylabel('Trap Depth (mK)')
plt.show()