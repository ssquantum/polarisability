"""Stefan Spence 24.04.19
Compare wavelengths in the range 800-830nm or 880nm 
for an optical tweezer for Rb by looking at the 
ratio of polarisabilities and the implications for
merging with a Cs optical tweezer at 1064nm

Model tweezer traps for Rb and Cs and find the potential each experiences
when they're overlapping. Should fix the separate tweezer trap depths to >1mK.
We also want Rb to experience a deeper trap from its tweezer than from the Cs
tweezer so that there isn't too much heating during merging.
Assume a fixed beam waist of 1 micron for all tweezer traps (ignore the 
change in the diffraction limit with wavelength).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('..')
from AtomFieldInt_V3 import (dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au)

Cswl = 1064e-9   # wavelength of the Cs tweezer trap in m
Rbwl = 880.2e-9  # wavelength of the Rb tweezer trap in m
power = 6e-3     # power of Cs tweezer beam in W
Rbpower = 6e-3   # power of Rb tweezer beam in W 
beamwaist = 1e-6 # beam waist in m
bprop = [Cswl, power, beamwaist] # collect beam properties
    
# For the 1064nm trap:
# mass, (L,J,F,MF), bprop, dipole matrix elements (Cm), resonant frequencies (rad/s),
# linewidths (rad/s), state labels, nuclear spin, atomic symbol.
# groundstate rubidium 5 S 1/2
Rb1064 = dipole(Rb.m, (0,1/2.,1,1), bprop,
                Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                nuclear_spin = Rb.I,
                symbol=Rb.X)
                
# groundstate caesium 6 S 1/2
Cs1064 = dipole(Cs.m, (0,1/2.,4,4), bprop,
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X)

# excited state caesium 6 P 3/2
CsP = dipole(Cs.m, (1,3/2.,5,5), bprop,
                Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                nuclear_spin = Cs.I,
                symbol=Cs.X)
                
# set the power of the traps so that the trap depth experienced by each 
# species in the overlapping trap is the same:
# Rbpower = (Cs1064.polarisability(Cswl,mj=0.5) - Rb1064.polarisability(Cswl, mj=0.5) 
#     )/ (Rb1064.polarisability(Rbwl, mj=0.5) - Cs1064.polarisability(Rbwl, mj=0.5)) * power

# Stability condition 1: 
P1Rb =  abs((-0.6e-3*kB*np.pi*eps0*c*beamwaist**2 + Cs1064.polarisability(Cswl)*power) 
                    / Cs1064.polarisability(Rbwl))
print("""Condition 1: The combined trap depth must be > 0.6mK for Cs.
Rb tweezer power < %.3g mW"""%(P1Rb*1e3))

# Stability condition 2: Rb is 1.5x more strongly attracted to its own tweezer
P2Rb = 1.5 * Rb1064.polarisability(Cswl) * power / Rb1064.polarisability(Rbwl)
print("""Condition 2: Rb is 1.5x more strongly attracted to its own tweezer.
Rb tweezer power > %.3g mW\n"""%(P2Rb*1e3))

# for the 880nm trap:
bprop = [Rbwl, abs(Rbpower), beamwaist]
Rb880 = dipole(Rb.m, (0,1/2.,1,1), bprop, # ground state Rb 5 S 1/2
                Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                nuclear_spin = Rb.I,
                symbol=Rb.X)
                
Cs880 = dipole(Cs.m, (0,1/2.,3,3), bprop, # ground state Cs 6 S 1/2
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X)
                

# in the trap with both tweezers overlapping: 
U0Rb = abs(Rb1064.acStarkShift(0,0,0) + Rb880.acStarkShift(0,0,0))
U0Cs = abs(Cs1064.acStarkShift(0,0,0) + Cs880.acStarkShift(0,0,0))
wrRb = np.sqrt(4*U0Rb / Rb.m / beamwaist**2) /2. /np.pi /1e3  # radial trapping frequency for Rb in kHz
wrCs = np.sqrt(4*U0Cs / Cs.m / beamwaist**2) /2. /np.pi /1e3  # radial trapping frequency for Cs in KHz
print("%.0f beam power: %.3g mW\t\t%.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, Rbpower*1e3))
print("""In the combined %.0fnm and %.0fnm trap:
Rubidium:   trap depth %.3g mK
            radial trapping frequency %.0f kHz 
Caesium:    trap depth %.3g mK
            radial trapping frequency %.0f kHz"""%(Rbwl*1e9, Cswl*1e9, U0Rb/kB*1e3, wrRb, U0Cs/kB*1e3, wrCs))

# with just the Cs tweezer trap:
URb = abs(Rb1064.acStarkShift(0,0,0))
wrRb1064 = np.sqrt(4*URb / Rb.m / beamwaist**2) /2. /np.pi /1e3
UCs = abs(Cs1064.acStarkShift(0,0,0))
wrCs1064 = np.sqrt(4*UCs / Cs.m / beamwaist**2) /2. /np.pi /1e3
print("""\nIn just the %.0fnm trap:
Rubidium:   trap depth %.3g mK
            radial trapping frequency %.0f kHz
Caesium:    trap depth %.3g mK
            radial trapping frequency %.0f kHz"""%(Cswl*1e9, URb/kB*1e3, wrRb1064, UCs/kB*1e3, wrCs1064))


# plot merging traps:
n = 3   # number of time steps in merging to plot
sep = np.linspace(0, 10e-6, n)     # initial separation of the tweezer traps
zs = np.linspace(-2, 10, 200)*1e-6 # positions along the beam axis

for atoms in [[Rb1064, Rb880], [Cs1064, Cs880]]:
    plt.figure(figsize=(6,7.5))
    plt.subplots_adjust(hspace=0.01)
    
    for i in range(n):
        ax = plt.subplot2grid((n,1), (i,0))
        
        U = (atoms[0].acStarkShift(0,0,zs) + atoms[1].acStarkShift(0,0,zs-sep[n-i-1]))/kB*1e3 # combined potential along the beam axis
        U1064 = atoms[0].acStarkShift(0,0,zs)/kB*1e3         # potential in the 1064 trap
        U880 = atoms[1].acStarkShift(0,0,zs-sep[n-i-1])/kB*1e3 # potential in the 880 trap
        plt.plot(zs*1e6, U, 'k')
        plt.plot(zs*1e6, U1064, color='tab:orange', alpha=0.6)
        plt.plot(zs*1e6, U880, color='tab:blue', alpha=0.6)
        plt.plot([0]*2, [min(U),0], color='tab:orange', linewidth=10, label='%.0f'%(Cswl*1e9), alpha=0.4)
        plt.plot([sep[n-i-1]*1e6]*2, [min(U),0], color='tab:blue', linewidth=10, label='%.0f'%(Rbwl*1e9), alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_title("Optical potential experienced by "+atoms[0].X
    +"\n%.0f beam power: %.3g mW   %.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, Rbpower*1e3),
                pad = 25)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
        
    plt.xlabel(r'Position ($\mu$m)')
    ax.set_xticks(sep*1e6)
    plt.ylabel('Trap Depth (mK)')
    ax.yaxis.set_major_locator(AutoLocator())
    
plt.show()