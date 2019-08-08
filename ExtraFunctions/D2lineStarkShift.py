"""Stefan Spence 05.06.19

working towards comparing predictions of the differential light shift in Rb
similar to:
Nicholas R Hutzler et al 2017 New J. Phys. 19 023007
but they use the D1 line whereas we're more interested in the D2 line.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'Y:\Tweezer\Code\Python 3.5\polarisability')
sys.path.append(r'Z:\Tweezer\Code\Python 3.5\polarisability')
from AtomFieldInt_V3 import dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, kB, amu, Eh, au

def getMFStarkShifts(wavelength = 1064e-9, # laser wavelength in m
                    power = 0.00906143,    # laser power in W
                    beamwaist = 1e-6,      # beam waist in m
                    ATOM = Cs):
    """Return the Stark shifts of the MF states for cooling/repump transitions"""
    bprop = [wavelength, power, beamwaist] # collect beam properties
    if ATOM == Cs: # assign the relevant hyperfine transitions
        Fs = [3,4]
        l1 = [18,24] # index of lines for making legend
    elif ATOM == Rb:
        Fs = [1,2]
        l1 = [6,12] # index of lines for making legend
    
    # print("Stark shift of "+ATOM.X+" S1/2 F = %s, %s -> P3/2 F' = %s, %s for different MF states."%(Fs[0],Fs[0]+1,Fs[1],Fs[1]+1))
    
    plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for F in Fs:
        for MF in range(-F, F+1):
            print(" ----- |F = "+str(F)+", m_F = "+str(MF)+">")
            for MFp in range(MF-1, MF+2):
                S = dipole(ATOM.m, (0,1/2.,F,MF), bprop,
                    ATOM.D0S, ATOM.w0S, ATOM.lwS, ATOM.nljS,
                    nuclear_spin = ATOM.I,
                    symbol=ATOM.X)
                P = dipole(ATOM.m, (1,3/2.,F+1,MFp), bprop,
                    ATOM.D0P3, ATOM.w0P3, ATOM.lwP3, ATOM.nljP3,
                    nuclear_spin = ATOM.I,
                    symbol=ATOM.X)
                shift = (S.acStarkShift(0,0,0, bprop[0], HF=True) - P.acStarkShift(0,0,0, bprop[0], HF=True))/h/1e6
                if MF != 0:
                    deltaMF = (MFp-MF)*np.sign(MF)
                else:
                    deltaMF = (MFp-MF)
                plt.plot(MF, shift, '_', color=colors[F-1], alpha=0.33*(2+deltaMF), markersize=15, linewidth=10)
                print("|F' = "+str(F+1)+", m_F' = "+str(MFp)+"> : %.5g MHz"%shift)
                
    plt.xlabel("$M_F$")  
    plt.ylabel("Differential AC Stark Shift (MHz)")
    lines = plt.gca().lines
    plt.legend(lines[l1[0]:l1[1]], ['F='+str(f)+r', $\Delta M_F=$'+str(-dmf) 
                for f in range(min(Fs),max(Fs)+1) for dmf in range(-1,2)])
    plt.show()

if __name__ == "__main__":
    getMFStarkShifts(wavelength = 812e-9, # laser wavelength in m
                    power = 0.00194652,    # laser power in W
                    beamwaist = 1.2e-6,      # beam waist in m
                    ATOM = Rb)