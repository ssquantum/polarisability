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

afu = 2 * np.pi * 1e3 # convert from angular frequency to kHz

Cswl = 1064e-9      # wavelength of the Cs tweezer trap in m
Rbwl = 807e-9       # wavelength of the Rb tweezer trap in m
power = 18e-3       # power of Cs tweezer beam in W
Cswaist = 1.2e-6    # beam waist for Cs in m
Rbpower = power*0.209 # power of Rb tweezer beam in W 
Rbwaist = 1.2e-6    # beam waist fir Rb in m
minU0 = -0.6e-3*kB  # min acceptable combined trap depth for Cs

    
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
def P1Rb(wlCs, wlRb, U0min=minU0, Cspower=power):
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

print("""Condition 1: The combined trap depth must be > %.2f mK for Cs.
Power ratio Rb / Cs < %.3g """%(minU0/kB*1e3, P1Rb(Cswl, Rbwl, Cspower=power) / power))
print("Condition 2: Rb is "+str(factor)+"""x more strongly attracted to its own tweezer.
Power ratio Rb / Cs > %.3g \n"""%(P2Rb(Cswl, Rbwl, power) / power))
print("Condition 3: Cs is "+str(factor)+"""x more strongly attracted to its own tweezer.
Power ratio Rb / Cs < %.3g \n"""%(P3Rb(Cswl, Rbwl, power) / power))

print("Combine all 3 conditions to fix Cspower and then get limits on Rbpower:\n")
Cspowermin = 2*abs(minU0) * np.pi*eps0*c * Cswaist**2 / Cs1064.polarisability(Cswl)
print("Cs power > %.3g mW"%(Cspowermin*1e3))

def getPRbmin(wlCs, wlRb, U0min=minU0, Cspower=power):
    """Combine conditions 1, 2, and 3 to get the min Rb tweezer power"""
    return factor**2 *Rb1064.polarisability(wlCs)/Rb1064.polarisability(wlRb)/Cs1064.polarisability(wlCs) * abs(U0min)*np.pi*eps0*c*Rbwaist**2

def getPRbmax(wlCs, wlRb, U0min=minU0, Cspower=power):
    """Combine conditions 1, 2, and 3 to get the max Rb tweezer power"""
    return Rbwaist**2 * abs(U0min / Cs1064.polarisability(wlRb)) * np.pi*eps0*c

# get the stability conditions as a function of wavelength
wavels = np.linspace(795, 845, 200) * 1e-9 # wavelengths to consider in m
ratio1 = P1Rb(Cswl, wavels, Cspower=power)/power # condition 1
ratio2 = P2Rb(Cswl, wavels, Cspower=power)/power # condition 2
ratio3 = P3Rb(Cswl, wavels, Cspower=power)/power # condition 3
diff1 = abs(ratio2 - ratio1)
diff2 = abs(ratio3 - ratio2)
crossover = wavels[np.argmin(diff1)] # wavelength where ratio1 crosses ratio2
# print("Stability conditions 1 and 2 crossover at %.1f nm\n"%(crossover*1e9))
crossover2 = wavels[np.argmin(diff2)] # wavelength where ratio2 crosses ratio3
# print("Stability conditions 1 and 2 crossover at %.1f nm\n"%(crossover2*1e9))
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


# get the stability conditions as a function of wavelength
wavels = np.linspace(795, 845, 200) * 1e-9 # wavelengths to consider in m
PRbmin = getPRbmin(Cswl, wavels, Cspower=Cspowermin) # in W
PRbmax = getPRbmax(Cswl, wavels, Cspower=Cspowermin) # in W
diff = abs(PRbmin - PRbmax)
crossover = wavels[np.argmin(diff)] # wavelength where PRbmin crosses PRbmax
print("Combined stability conditions crossover at %.1f nm\n"%(crossover*1e9))
both = np.concatenate((PRbmin, PRbmax))*1e3 # in mW

def plotc123(Cspower=Cspowermin):
    """plot the combined stability conditions as a function of wavelength"""
    plt.figure()
    plt.title('Threshold Conditions on $P_{Rb}$ when $P_{Cs}$ = %.3g mW'%(Cspower*1e3))
    plt.plot(wavels*1e9, PRbmax*1e3, color='tab:blue', label='Maximum power')
    plt.plot(wavels*1e9, PRbmin*1e3, color='tab:orange', label='Minimum power') 
    plt.fill_between([crossover*1e9, wavels[-1]*1e9], min(both), max(both), color='tab:red', alpha=0.2)
    plt.fill_between([wavels[0]*1e9, crossover*1e9], min(both), max(both), color='tab:green', alpha=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Rb tweezer beam power $P_{Rb}$ (mW)')
    plt.xlim(wavels[0]*1e9, wavels[-1]*1e9)
    plt.ylim(min(PRbmin), max(PRbmax)*1.5e3)
    plt.text(crossover*1e9, max(PRbmax)*0.24*1e3, 'Upper limit at %.0f nm: %.3g mW'%(Rbwl*1e9, 
        getPRbmax(Cswl, Rbwl, Cspower=Cspower)*1e3),
        color='tab:blue', bbox=dict(facecolor='white', edgecolor=None))
    plt.text(crossover*1e9, max(PRbmax)*0.1*1e3, 'Lower limit at %.0f nm: %.3g mW'%(Rbwl*1e9, 
        getPRbmin(Cswl, Rbwl, Cspower=Cspower)*1e3),
        color='tab:orange', bbox=dict(facecolor='white', edgecolor=None))
    plt.legend()


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
                
def getLD(atom, wtrap):
    """Return the Lamb-Dicke paramter for a given dipole object in a 
    trap with trapping frequency wtrap
    eta = sqrt(w_recoil / w_trap)"""
    return atom.field.k * np.sqrt(hbar/2./atom.m / wtrap)

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
wrRb = np.sqrt(abs(Rbpopt[0])*2 / Rb.m)            # radial trapping frequency for Rb in rad/s
ldRb = getLD(Rb880, wrRb)                          # Lamb-Dicke parameter 
wrRb /= afu                                        # in kHz
wrCs = np.sqrt(abs(Cspopt[0])*2 / Cs.m)            # radial trapping frequency for Cs in rad/s
ldCs = getLD(Cs1064, wrCs)                         # Lamb-Dicke parameter 
wrCs /= afu                                        # in kHz

def checkfit():
    """Graph showing the fitted quadratic to prove that it's good"""
    xtra = np.linspace(-xmax*5, xmax*5, 200)  # extend the range 
    UCRb  = Rb1064.acStarkShift(xtra,0,0) + Rb880.acStarkShift(xtra,0,0)  # Rb potential in combined trap
    UCCs  = Cs1064.acStarkShift(xtra,0,0) + Cs880.acStarkShift(xtra,0,0)  # Cs potential in combined trap
    plt.figure()
    plt.plot(xtra*1e6, UCRb/kB*1e3, 'tab:blue', label='Rb $\omega_r = %.3g$ kHz'%wrRb)
    plt.plot(xtra*1e6, quad(xtra, *Rbpopt)/kB*1e3, '--', color='tab:blue')
    plt.plot(xtra*1e6, UCCs/kB*1e3, 'tab:orange', label='Cs $\omega_r = %.3g$ kHz'%wrCs)
    plt.plot(xtra*1e6, quad(xtra, *Cspopt)/kB*1e3, '--', color='tab:orange')
    plt.xlabel('Radial Position ($\mu m$)')
    plt.ylabel('Dipole Potential (mK)')
    both = np.concatenate((UCRb/kB*1e3, UCCs/kB*1e3))
    plt.ylim((min(both), max(both)))
    plt.legend()

def axialfit():
    """Graph fitting a quadratic to the potential in the axial direction"""
    zmax = max([Cswaist,Rbwaist])  # only harmonic near the bottom
    zpos = np.linspace(-zmax, zmax, 200) # x position in m
    UCRb  = Rb1064.acStarkShift(0,0,zpos) + Rb880.acStarkShift(0,0,zpos)  # Rb potential in combined trap
    UCCs  = Cs1064.acStarkShift(0,0,zpos) + Cs880.acStarkShift(0,0,zpos)  # Cs potential in combined trap
    Rbpopt, _ = curve_fit(quad, zpos, UCRb, p0=(1,U0Rb), maxfev=80000) # popt: (gradient, offset)
    Cspopt, _ = curve_fit(quad, zpos, UCCs, p0=(1,U0Cs), maxfev=80000)
    wzRb = np.sqrt(abs(Rbpopt[0])*2 / Rb.m) /afu           # axial trapping frequency for Rb in kHz
    wzCs = np.sqrt(abs(Cspopt[0])*2 / Cs.m) /afu           # axial trapping frequency for Cs in kHz
    zxtra = np.linspace(-zmax*5, zmax*5, 200)  # extend the range 
    UCRb  = Rb1064.acStarkShift(0,0,zxtra) + Rb880.acStarkShift(0,0,zxtra)  # Rb potential in combined trap
    UCCs  = Cs1064.acStarkShift(0,0,zxtra) + Cs880.acStarkShift(0,0,zxtra)  # Cs potential in combined trap
    plt.figure()
    plt.plot(zxtra*1e6, UCRb/kB*1e3, 'tab:blue', label=r'Rb $\omega_z=%.3g$ kHz, $\eta=%.2g$'%(wzRb, getLD(Rb880, wzRb*afu)))
    plt.plot(zxtra*1e6, quad(zxtra, *Rbpopt)/kB*1e3, '--', color='tab:blue')
    plt.plot(zxtra*1e6, UCCs/kB*1e3, 'tab:orange', label=r'Cs $\omega_z = %.3g$ kHz, $\eta=%.2g$'%(wzCs, getLD(Cs1064, wzCs*afu)))
    plt.plot(zxtra*1e6, quad(zxtra, *Cspopt)/kB*1e3, '--', color='tab:orange')
    plt.xlabel(r'Axial Position ($\mu m$)')
    plt.ylabel('Dipole Potential (mK)')
    both = np.concatenate((UCRb/kB*1e3, UCCs/kB*1e3))
    plt.ylim((min(both), max(both)))
    plt.legend()


# note: if the Cs potential has a dimple from the repulsive Rb component then it might just fit
# to this part, giving a much lower trapping frequency than it should have.
print("%.0f beam power: %.3g mW\t\t%.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, Rbpower*1e3))
print("""In the combined %.0fnm and %.0fnm trap:
Rubidium:       trap depth %.3g mK
                radial trapping frequency %.0f kHz, Lamb-Dicke parameter %.3g 
Caesium:        trap depth %.3g mK
                radial trapping frequency %.0f kHz, Lamb-Dicke parameter %.3g"""%(
                    Rbwl*1e9, Cswl*1e9, U0Rb/kB*1e3, wrRb, ldRb, U0Cs/kB*1e3, wrCs, ldCs))

# with just the Cs tweezer trap:
def trap_freq(atom):
    """Get the trapping frequency of a dipole object"""
    return np.sqrt(4*abs(atom.acStarkShift(0,0,0)) / atom.m / atom.field.w0**2)

wrRb1064 = trap_freq(Rb1064)  # Rb trapping frequency in 1064nm trap in rad/s
wrCs1064 = trap_freq(Cs1064)  # Cs trapping frequency in 1064nm trap in rad/s
print("""\nIn just the %.0fnm trap:
Rubidium:       trap depth %.3g mK
                radial trapping frequency %.0f kHz, Lamb-Dicke parameter %.3g 
Caesium:        trap depth %.3g mK
                radial trapping frequency %.0f kHz, Lamb-Dicke parameter %.3g """%(
                Cswl*1e9, Rb1064.acStarkShift(0,0,0)/kB*1e3, wrRb1064/afu, getLD(Rb1064, wrRb1064),
                Cs1064.acStarkShift(0,0,0)/kB*1e3, wrCs1064/afu, getLD(Cs1064, wrCs1064)))

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
                ax.text(xs[0]*1e6, 0, '$\omega = %.0f$ kHz'%(trap_freq(atoms[0])/afu), 
                                                        bbox=dict(facecolor='white', edgecolor=None))
                ax.text(sep[-1]*1e6,0,'$\omega = %.0f$ kHz'%(trap_freq(atoms[1])/afu), 
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
        if any([argv == '0' for argv in sys.argv]):
            plotc12() # threshold conditions 1 and 2
        if any([argv == '1' for argv in sys.argv]):
            plotc23() # Rbwl vs Cswl from threshold conditions 2 and 3
        if any([argv == '2' for argv in sys.argv]):
            plotc123() # PRb vs wavelength from all threshold conditions
        if any([argv == '3' for argv in sys.argv]):
            plotmerge() # merging tweezers
        if any([argv == '4' for argv in sys.argv]):
            plotcross12() # Rbwl vs PCs from threshold conditions 2 and 3
        if any([argv == '5' for argv in sys.argv]):
            checkfit() # Check the fit of the quadratic to the combined potential
        if any([argv == '6' for argv in sys.argv]):
            axialfit() # fit to the potential in the  axial direction 
        if any([argv == 'all' for argv in sys.argv]):
            # plotc12()
            # plotc23()
            checkfit()
            axialfit()
            plotc123()
            plotmerge()
            plotcross12()
        plt.show()