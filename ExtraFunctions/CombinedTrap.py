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
from scipy.optimize import curve_fit
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('..')
from AtomFieldInt_V3 import (dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au)

Cswl = 980e-9      # wavelength of the Cs tweezer trap in m
Rbwl = 810e-9       # wavelength of the Rb tweezer trap in m
power = 10e-3       # power of Cs tweezer beam in W
Cswaist = 1.2e-6    # beam waist for Cs in m
Rbpower = power*0.17 # power of Rb tweezer beam in W 
Rbwaist = 0.9e-6    # beam waist fir Rb in m

    
# For the 1064nm trap: at Cs wavelength with Cs power and Cs beam waist
# mass, (L,J,F,MF), bprop, dipole matrix elements (Cm), resonant frequencies (rad/s),
# linewidths (rad/s), state labels, nuclear spin, atomic symbol.
# groundstate rubidium 5 S 1/2
Rb1064 = dipole(Rb.m, (0,1/2.,1,1), [Cswl, power, Cswaist], 
                Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                nuclear_spin = Rb.I,
                symbol=Rb.X)
                
# groundstate caesium 6 S 1/2
Cs1064 = dipole(Cs.m, (0,1/2.,4,4), [Cswl, power, Cswaist], 
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X)
                
print("Rb tweezer wavelength: %.0f nm\t\tCs tweezer wavelength: %.0f nm\n"%(Cswl*1e9, Rbwl*1e9))

# set the power of the traps so that the trap depth experienced by each 
# species in the overlapping trap is the same:
# Rbpower = (Cs1064.polarisability(Cswl,mj=0.5) - Rb1064.polarisability(Cswl, mj=0.5) 
#     )/ (Rb1064.polarisability(Rbwl, mj=0.5) - Cs1064.polarisability(Rbwl, mj=0.5)) * power

# Stability condition 1: 
def P1Rb(wlCs, wlRb, U0min=-0.6e-3*kB, Cspower=power):
    """Condition 1: The combined trap depth must be > 0.6mK for Cs."""
    return abs((U0min*np.pi*eps0*c + Cs1064.polarisability(wlCs)*Cspower/Cswaist**2) 
                    * Rbwaist**2 / Cs1064.polarisability(wlRb))

# Stability condition 2: 
factor = 2
def P2Rb(wlCs, wlRb, Cspower=power):
    """Condition 2: Rb is factor times more strongly attracted to its own tweezer"""
    return factor * Rb1064.polarisability(wlCs) * Cspower * Rbwaist**2 / Rb1064.polarisability(wlRb) / Cswaist**2

# Stability condition 3:
def P3Rb(wlCs, wlRb, Cspower=power):
    """Condition 3: Cs is factor times more strongly attracted to its own tweezer"""
    return abs(Cs1064.polarisability(wlCs) * Cspower * Rbwaist**2 / Cs1064.polarisability(wlRb) / Cswaist**2 / factor)


print("""Condition 1: The combined trap depth must be > 0.6mK for Cs.
Power ratio Rb / Cs < %.3g """%(P1Rb(Cswl, Rbwl, Cspower=power) / power))
print("Condition 2: Rb is "+str(factor)+"""x more strongly attracted to its own tweezer.
Power ratio Rb / Cs > %.3g \n"""%(P2Rb(Cswl, Rbwl, power) / power))
print("Condition 3: Cs is "+str(factor)+"""x more strongly attracted to its own tweezer.
Power ratio Rb / Cs < %.3g \n"""%(P3Rb(Cswl, Rbwl, power) / power))

# get the stability conditions as a function of wavelength
wavels = np.linspace(795, 845, 200) * 1e-9 # wavelengths to consider in m
ratio1 = P1Rb(Cswl, wavels, Cspower=power)/power # condition 1
ratio2 = P2Rb(Cswl, wavels, Cspower=power)/power # condition 2
ratio3 = P3Rb(Cswl, wavels, Cspower=power)/power # condition 3
diff1 = abs(ratio2 - ratio1)
diff2 = abs(ratio3 - ratio2)
crossover = wavels[np.argmin(diff1)] # wavelength where ratio1 crosses ratio2
print("Stability conditions 1 and 2 crossover at %.1f nm\n"%(crossover*1e9))
crossover2 = wavels[np.argmin(diff2)] # wavelength where ratio2 crosses ratio3
print("Stability conditions 1 and 2 crossover at %.1f nm\n"%(crossover2*1e9))
both = np.concatenate((ratio1, ratio2))

def plotc12():
    """plot the stability conditions as a function of wavelength for
    conditions 1 and 2"""
    plt.figure()
    plt.title('Threshold Conditions on Power Ratio $P_{Rb}/P_{Cs}$')
    plt.plot(wavels*1e9, ratio1, color='tab:blue', 
        label='Upper Limit from $U_{Cs}$ < -0.6 mK')
    plt.plot(wavels*1e9, ratio2, color='tab:orange',
        label='Lower Limit from $U_{Rb}(\lambda) > %s U_{Rb}$(%.0f nm)'%(factor, Cswl*1e9)) 
    plt.fill_between([crossover*1e9, wavels[-1]*1e9], min(both), max(both), color='tab:red', alpha=0.2)
    plt.fill_between([wavels[0]*1e9, crossover*1e9], min(both), max(both), color='tab:green', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Power Ratio $P_{Rb}/P_{Cs}$')
    plt.xlim(wavels[0]*1e9, wavels[-1]*1e9)
    plt.ylim(min(ratio2), max(ratio1))
    plt.text(812, max(ratio2)*0.25, 'Upper limit at %.0f nm: %.3g'%(Rbwl*1e9, P1Rb(Cswl,Rbwl)/power),
        color='tab:blue', bbox=dict(facecolor='white', edgecolor=None))
    plt.text(812, max(ratio2)*0.21, 'Lower limit at %.0f nm: %.3g'%(Rbwl*1e9, P2Rb(Cswl,Rbwl)/power),
        color='tab:orange', bbox=dict(facecolor='white', edgecolor=None))
    plt.legend()

def plotc23():
    """plot the stability conditions as a function of wavelength for
    conditions 2 and 3"""
    Cswavels = np.linspace(930, 1070, 100)*1e-9 # Cs wavelengths in m
    crosswls = np.zeros(len(Cswavels))
    for i in range(len(Cswavels)):
        r2 = P2Rb(Cswavels[i], wavels, power)/power # condition 2
        r3 = P3Rb(Cswavels[i], wavels, power)/power # condition 3
        d2 = abs(r3 - r2)
        crosswls[i] = wavels[np.argmin(d2)] # crossover wavelength in m

    plt.figure()
    plt.title('Threshold conditions on Rb tweezer wavelength')
    plt.plot(Cswavels*1e9, crosswls*1e9, 'ko')
    plt.xlabel('Cs tweezer wavelength (nm)')
    plt.ylabel('Maximum Rb tweezer wavelength (nm)')

# for the 880nm trap:
# ground state Rb 5 S 1/2
Rb880 = dipole(Rb.m, (0,1/2.,1,1), [Rbwl, abs(Rbpower), Rbwaist], 
                Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                nuclear_spin = Rb.I,
                symbol=Rb.X)

# ground state Cs 6 S 1/2
Cs880 = dipole(Cs.m, (0,1/2.,3,3), [Rbwl, abs(Rbpower), Rbwaist], 
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X)
                

# in the trap with both tweezers overlapping: 
U0Rb = Rb1064.acStarkShift(0,0,0) + Rb880.acStarkShift(0,0,0)
U0Cs = Cs1064.acStarkShift(0,0,0) + Cs880.acStarkShift(0,0,0)
# estimate trapping frequency by fitting quadratic U = m w^2 x^2 /2
xmax = max([Cswaist,Rbwaist]) * 0.3 # only harmonic near the bottom
xpos = np.linspace(-xmax, xmax, 200) # x position in m
UCRb  = Rb1064.acStarkShift(xpos,0,0) + Rb880.acStarkShift(xpos,0,0)  # Rb potential in combined trap
UCCs  = Cs1064.acStarkShift(xpos,0,0) + Cs880.acStarkShift(xpos,0,0)  # Cs potential in combined trap
quad = lambda x, a, c: a*x**2 + c
Rbpopt, Rbpcov = curve_fit(quad, xpos, UCRb, p0=(1,U0Rb), maxfev=80000) # popt: (gradient, offset)
Cspopt, Cspcov = curve_fit(quad, xpos, UCCs, p0=(1,U0Cs), maxfev=80000)
wrRb = np.sqrt(abs(Rbpopt[0])*2 / Rb.m) /2./np.pi / 1e3  # radial trapping frequency for Rb in kHz
wrCs = np.sqrt(abs(Cspopt[0])*2 / Cs.m) /2./np.pi / 1e3  # radial trapping frequency for Cs in KHz
# note: if the Cs potential has a dimple from the repulsive Rb component then it might just fit
# to this part, giving a much lower trapping frequency than it should have.
print("%.0f beam power: %.3g mW\t\t%.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, Rbpower*1e3))
print("""In the combined %.0fnm and %.0fnm trap:
Rubidium:       trap depth %.3g mK
                radial trapping frequency %.0f kHz 
Caesium:        trap depth %.3g mK
                radial trapping frequency %.0f kHz"""%(Rbwl*1e9, Cswl*1e9, U0Rb/kB*1e3, wrRb, U0Cs/kB*1e3, wrCs))

# with just the Cs tweezer trap:
def trap_freq(atom):
    """Get the trapping frequency of a dipole object"""
    return np.sqrt(4*abs(atom.acStarkShift(0,0,0)) / atom.m / atom.field.w0**2)

wrRb1064 = trap_freq(Rb1064) /2. /np.pi /1e3
wrCs1064 = trap_freq(Cs1064) /2. /np.pi /1e3
print("""\nIn just the %.0fnm trap:
Rubidium:       trap depth %.3g mK
                radial trapping frequency %.0f kHz
Caesium:        trap depth %.3g mK
                radial trapping frequency %.0f kHz"""%(Cswl*1e9, abs(Rb1064.acStarkShift(0,0,0)/kB*1e3), 
                wrRb1064, abs(Cs1064.acStarkShift(0,0,0)/kB*1e3), wrCs1064))

def plotmerge(n=3):
    """plot merging traps with n timesteps"""
    sep = np.linspace(0, max([Cswaist,Rbwaist])*2, n)     # initial separation of the tweezer traps
    xs = np.linspace(-max(sep)*0.5, max(sep)*1.5, 200)    # positions along the beam axis

    for atoms in [[Rb1064, Rb880, wrRb], [Cs1064, Cs880, wrCs]]:
        plt.figure(figsize=(6,7.5))
        plt.subplots_adjust(hspace=0.01)
        
        for i in range(n):
            ax = plt.subplot2grid((n,1), (i,0))
            # combined potential along the beam axis:
            U = (atoms[0].acStarkShift(xs,0,0) + atoms[1].acStarkShift(xs-sep[n-i-1],0,0))/kB*1e3 
            U1064 = atoms[0].acStarkShift(xs,0,0)/kB*1e3         # potential in the 1064 trap
            U880 = atoms[1].acStarkShift(xs-sep[n-i-1],0,0)/kB*1e3 # potential in the 880 trap
            plt.plot(xs*1e6, U, 'k')
            plt.plot(xs*1e6, U1064, color='tab:orange', alpha=0.6)
            plt.plot(xs*1e6, U880, color='tab:blue', alpha=0.6)
            allU = np.concatenate((U, U1064, U880))
            plt.plot([0]*2, [min(allU),max(allU)], color='tab:orange', linewidth=10, label='%.0f'%(Cswl*1e9), alpha=0.4)
            plt.plot([sep[n-i-1]*1e6]*2, [min(allU),max(allU)], color='tab:blue', linewidth=10, label='%.0f'%(Rbwl*1e9), alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title("Optical potential experienced by "+atoms[0].X
        +"\n%.0f beam power: %.3g mW   %.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, Rbpower*1e3),
                    pad = 25)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
                ax.text(xs[0]*1e6, 0, '$\omega = %.0f$ kHz'%(trap_freq(atoms[0])/2./np.pi/1e3), 
                                                        bbox=dict(facecolor='white', edgecolor=None))
                ax.text(sep[-1]*1e6,0,'$\omega = %.0f$ kHz'%(trap_freq(atoms[1])/2./np.pi/1e3), 
                                                        bbox=dict(facecolor='white', edgecolor=None))
        
            
        plt.xlabel(r'Position ($\mu$m)')
        ax.set_xticks(sep*1e6)
        plt.ylabel('Trap Depth (mK)')
        ax.text(xs[0]*1e6, 0, '$\omega = %.0f$ kHz'%atoms[2], bbox=dict(facecolor='white', edgecolor=None))
        ax.yaxis.set_major_locator(AutoLocator())
    
def plotcross12():
    """Use conditions 1 and 2 to find the maximum Rb tweezer wavelength as a 
    function of Cs beam power"""
    wavels = np.linspace(800, 830, 100) * 1e-9 # wavelengths to consider in m
    crossovers = []
    pratio     = []
    powers = np.linspace(7,20,50)*1e-3
    for p in powers:
        ratio1 = P1Rb(Cswl, wavels, Cspower=p)/p # condition 1
        ratio2 = P2Rb(Cswl, wavels, Cspower=p)/p # condition 2
        diff = abs(ratio2 - ratio1)
        crossovers.append(wavels[np.argmin(diff)]) # wavelength where ratio1 crosses ratio2
        pratio.append(P1Rb(Cswl, wavels[np.argmin(diff)], Cspower=p)/p) # power ratio at crossover

    fig, ax1 = plt.subplots()
    ax1.plot(powers*1e3, np.array(crossovers)*1e9, 'ko')
    ax1.set_xlabel('Cs beam power (mW)')
    ax1.set_ylabel('Crossover wavelength (nm)')

    ax2 = ax1.twinx()
    ax2.plot(powers*1e3, pratio, 'x', color='tab:red') # power ratios at the crossover wavelength
    ax2.set_ylabel('Power ratio ($P_{Rb}/P_{Cs}$)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')


if __name__ == "__main__":
    # only plot if the user passes a second argument, this acts like verbosity level
    # e.g. python CombinedTrap.py 1
    if np.size(sys.argv) > 1:
        if any([argv == '1' for argv in sys.argv]):
            plotc12() # threshold conditions 1 and 2
        elif any([argv == '2' for argv in sys.argv]):
            plotc23() # Rbwl vs Cswl from threshold conditions 2 and 3
        elif any([argv == '3' for argv in sys.argv]):
            plotmerge() # merging tweezers
        elif any([argv == '4' for argv in sys.argv]):
            plotcross12() # Rbwl vs PCs from threshold conditions 2 and 3
        elif any([argv == 'all' for argv in sys.argv]):
            plotc12()
            plotc23()
            plotmerge()
            plotcross12()
        plt.show()