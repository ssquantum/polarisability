"""Stefan Spence 11.06.19
Check the Stark shift of the cooling and repump transitions in Rb87"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13}) # increase font size (default 10)
import sys
sys.path.append(r'Y:\Tweezer\Code\Python 3.5\polarisability')
sys.path.append(r'Z:\Tweezer\Code\Python 3.5\polarisability')
from AtomFieldInt_V3 import (dipole, Rb, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au)

def getMFStarkShifts(wavelength = 1064e-9, # laser wavelength in m
                    power = 0.00906143,    # laser power in W
                    beamwaist = 1e-6,      # beam waist in m
                    plotit = False):       # toggle to plot the results
    """Return the Stark shifts of the MF states for cooling/repump transitions.
    Units are MHz"""
    bprop = [wavelength, power, beamwaist] # collect beam properties
    Fs = [1,2]
    numstates = sum(2*np.array(Fs)+1)*3 # total number of hyperfine transitions
    l1 = [6,12] # index of lines for making legend
    #
    states = np.zeros((numstates, 3)) # F, MF, MFprime
    shifts = np.zeros(numstates)      # differential ac Stark shifts
    i = 0       # index 
    for F in Fs:
        for MF in range(-F, F+1):
            for MFp in range(MF-1, MF+2):
                S = dipole(Rb.m, (0,1/2.,F,MF), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
                P = dipole(Rb.m, (1,3/2.,F+1,MFp), bprop,
                    Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
                #
                states[i] = (F, MF, MFp)
                # units are MHz
                shifts[i] = (S.acStarkShift(0,0,0, bprop[0], HF=True) - P.acStarkShift(0,0,0, bprop[0], HF=True))/h/1e6
                i += 1
    #  
    if plotit:
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        i = 0 
        for F, MF, MFp in states:
            if MF != 0:
                deltaMF = (MFp-MF)*np.sign(MF)
            else:
                deltaMF = (MFp-MF)
            plt.plot(MF, shifts[i], '_', color=colors[F-1], alpha=0.33*(2+deltaMF), markersize=15, linewidth=10)
            i += 1
        plt.xlabel("$M_F$")  
        plt.ylabel("AC Stark Shift (MHz)")
        lines = plt.gca().lines
        plt.legend(lines[l1[0]:l1[1]], ['F='+str(f)+r', $\Delta M_F=$'+str(-dmf) 
                    for f in range(min(Fs),max(Fs)+1) for dmf in range(-1,2)])
        plt.show()
    #
    return states, shifts


def aveShift(states, shifts, weights=None):
    """Average over the MF states. For an unequal population distribution,
    use weights to supply the proportion of the population in each state."""
    Fs, occs = np.unique(states[:,0], return_counts=True) # F states
    if not weights: # probability of being in each MF state
        weights = np.concatenate([np.ones(o)/o for o in occs]) # equal distribution
    weighted_shift = shifts*weights
    aveShifts = [np.sum(weighted_shift[np.where(Fs==F)]) for F in Fs] # average
    return aveShifts

if __name__ == "__main__":
    # calculate the stark shifts for trap depths 0 - 2 mK
    beamwaist = 1.2e-6 # beam waist of tweezer in m
    wavelength = 1064e-9 # wavelength of tweezer beam in m
    # make an object to calculate the groundstate polarisability
    G = dipole(Rb.m, (0,1/2.,1,1), [wavelength, 1e-3, beamwaist],
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
    powers = abs(np.linspace(0,2,200)*1e-3*kB / G.polarisability(wavelength) 
                                            * np.pi * eps0 * c * beamwaist**2)

    shifts = np.zeros((len(powers), 2)) # AC Stark shifts in MHz
    for i in range(len(powers)):
        mfstates, mfshifts = getMFStarkShifts(wavelength, powers[i], beamwaist)
        shifts[i] = aveShift(mfstates, mfshifts)
    
    plt.figure()
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    plt.title('Averaging over the hyperfine transitions of Rb D2 line')
    ax1.plot(powers*1e3, shifts[:,0], label="F=1 $\rightarrow$ F'=2")
    ax1.plot(powers*1e3, shifts[:,1], label="F=2 $\rightarrow$ F'=3")
    plt.legend()
    ax1.set_xlabel('Tweezer Beam Power (mW)')
    ax1.set_ylabel('Differential Stark Shift (MHz)')
    ax2.plot(np.linspace(0,2,200), np.zeros(200))
    ax2.cla()
    ax2.set_xlabel('Tweezer Trap Depth (mK)')
    plt.show()