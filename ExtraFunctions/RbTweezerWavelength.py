"""Stefan Spence 22.04.19
Compare the scattering rates and polarisabilities of Rb and Cs 
in the range of 800-845nm with 880.2nm and 1064nm in order to 
help decide the wavelength of a species-specific Rb tweezer trap"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('..')
from AtomFieldInt_V3 import (dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au)

wavelength = 880.2e-9     # wavelength in m
wavels = np.linspace(795,850,500)*1e-9 # wavelengths in m to plot
power = 5e-3            # beam power in W
beamwaist = 1e-6        # beam waist in m

# create dipole objects for ground and excited state Rb / Cs
bprop = [wavelength, power, beamwaist]
Rb5S = dipole(Rb.m, (0,1/2.,1,1), bprop,  # Rb ground 5 S 1/2 state
                Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                nuclear_spin = Rb.I,
                symbol=Rb.X)

Rb5P = dipole(Rb.m, (1,3/2.,1,1), bprop, # Rb excited 5 P 3/2 state
                Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                nuclear_spin = Rb.I,
                symbol=Rb.X)
                
Cs6S = dipole(Cs.m, (0,1/2.,3,3), bprop, # Cs ground 6 S 1/2 state
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X)
                
Cs6P = dipole(Cs.m, (1,3/2.,3,3), bprop, # Cs excited 6 P 3/2 state
                Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                nuclear_spin = Cs.I,
                symbol=Cs.X)               


# plot the absolute polarisabilities of Rb and Cs
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 2],'hspace':0}, sharex='all')
ax1.set_title('Polarisability of ground state Rb and Cs')
ax1.plot(wavels*1e9, Rb5S.polarisability(wavels)/au, color='tab:blue', label = 'Rb') # Rb
ax1.plot(wavels*1e9, np.zeros(len(wavels)) + Rb5S.polarisability(1064e-9)/au, 
        '--', color='tab:blue', label = 'Rb at 1064 nm') # compare value at 1064nm
ax1.plot(wavels*1e9, Cs6S.polarisability(wavels)/au, color='tab:orange', label = 'Cs') # Cs
ax1.plot(wavels*1e9, np.zeros(len(wavels)) + Cs6S.polarisability(1064e-9)/au, 
        '--', color='tab:orange', label = 'Cs at 1064 nm') # compare value at 1064nm
ax1.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.1) # show zero crossing   
ax1.set_ylabel('Polarisability ($a_0^3$)')
ax1.set_ylim(-7500, 8000)
ax1.legend()

# plot the ratio of groundstate polarisabliities Rb / Cs
ax2.semilogy(wavels*1e9, abs(Rb5S.polarisability(wavels)/Cs6S.polarisability(wavels)), 
    color='tab:blue') # ratio
ax2.plot(wavels*1e9, np.zeros(len(wavels)) + 
    Rb5S.polarisability(1064e-9)/Cs6S.polarisability(1064e-9), '--', color='tab:orange', 
    label='Ratio at 1064 nm') # compare value at 1064nm
ax2.legend()
ax2.set_ylabel('Ratio of Rb / Cs\nPolarisability')
ax2.set_ylim(0.1, 50)
ax2.set_xlabel('Wavelength (nm)')
ax2.set_xlim(wavels[0]*1e9, wavels[-1]*1e9)
plt.tight_layout()


def get_scattering_rates(Powers, wls, w0):
    """get the trap depths for ground and excited states of Rb and Cs
    then use them to calculate the scattering rates
    Powers: beam power in Watts
    wls: wavelengths in m (same size as Powers)
    w0: beamwaist in m
    """
    trapdepths = []
    for obj in [Cs6S, Cs6P, Rb5S, Rb5P]:
        if np.size(Powers) > 1:
            res = np.zeros(len(Powers))
            for i in range(len(Powers)):
                obj.field.E0 = 2 * np.sqrt(Powers[i] / eps0 / c / np.pi)/w0
                # average mj states (doesn't have any effect on j=1/2 states)
                res[i] = 0.5*(obj.acStarkShift(0,0,0, wls[i], mj=1.5) + 
                        obj.acStarkShift(0,0,0, wls[i], mj=0.5))
            trapdepths.append(res)
        else: # for only one power/wavelength
            obj.field.E0 = 2 * np.sqrt(Powers / eps0 / c / np.pi)/w0
            trapdepths.append(0.5*(obj.acStarkShift(0,0,0, wls, mj=1.5) + 
                        obj.acStarkShift(0,0,0, wls, mj=0.5)))

    I = 2*Powers / np.pi / w0**2 # intensity in W / m^2

    # scattering rate of Cs from the D1 and D2 lines:
    deltaCsD1 = 2*np.pi*c * (1/wls - 1/Cs.rwS[0]) # detuning from D1 (rad/s)
    deltaCsD2 = 2*np.pi*c * (1/wls - 1/Cs.rwS[35]) # detuning from D2 (rad/s)
    IsatCsD1 = 2.4981 *1e-3 *1e4 # saturation intensity for D1 transition, sigma polarised
    IsatCsD2 = 1.1023 *1e-3 *1e4 # saturation intensity for D2 transition, pi polarised
    CsRsc = 0
    for vals in [[Cs.lwS[0], deltaCsD1, IsatCsD1], [Cs.lwS[35], deltaCsD2, IsatCsD2]]:
        CsRsc += vals[0]/2. * I/vals[2] / (1 + 4*(vals[1]/vals[0])**2 + I/vals[2])
    # the lifetime is the trap depth / recoil energy / scattering rate
    # Cstau = 1e-3*kB / (hbar*(2*np.pi/wavels))**2 * 2.*Cs.m / CsRsc 
    # duration in vibrational ground state (s) = 1/Lamb-Dicke^2 /Rsc
    Cst = 4*np.sqrt(Cs.m*abs(trapdepths[0])) / (2*np.pi/wls)**2 /hbar /w0 /CsRsc 

    # scattering rate of Rb from the D1 line:
    deltaRbD1 = 2*np.pi*c * (1/wls - 1/Rb.rwS[0]) # detuning from D1 (rad/s)
    IsatRbD1 = 4.484 *1e-3 *1e4 # saturation intensity for D1 transition, pi polarised
    RbRsc = Rb.lwS[0]/2. * I/IsatRbD1 / (1 + 4*(deltaRbD1/Rb.lwS[0])**2 + I/IsatRbD1) # per second
    # Rbtau = 1e-3*kB / (hbar*(2*np.pi/wavels))**2 * 2.*Rb.m / RbRsc 
    # the lifetime is the trap depth / recoil energy / scattering rate
    # duration in vibrational ground state (s) = 1/Lamb-Dicke^2 /Rsc
    Rbt = 4*np.sqrt(Rb.m*abs(trapdepths[2])) / (2*np.pi/wls)**2 /hbar /w0 /RbRsc 

    return CsRsc, Cst, RbRsc, Rbt

# scattering rates in a 1mK trap for Rb, Cs
# choose powers so that Rb is fixed at 1mK trap depth. Add on wavelengths of 880.2nm and 1064nm
RbPowers = abs(1e-3*kB * np.pi * eps0 * c * beamwaist**2 / Rb5S.polarisability(
        np.array(list(wavels) + [880.2e-9]))) # in Watts
CsRsc, Cst, RbRsc, Rbt = get_scattering_rates(RbPowers, np.array(list(wavels) + [880.2e-9]), beamwaist)

# choose powers so that Cs is fixed at 1mK trap depth to compare at 1064nm
CsPower = abs(1e-3*kB * np.pi * eps0 * c * beamwaist**2 / Cs6S.polarisability(1064e-9)) # in Watts
CsRsc_Cs, Cst_Cs, RbRsc_Cs, Rbt_Cs = get_scattering_rates(CsPower, 1064e-9, beamwaist)

# plot the scattering rate in a Rb 1mK trap for Rb, Cs
fig, (ax3, ax5) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 2],'hspace':0.05}, sharex='all')
ax3.set_title('Scattering rates of ground state Rb and Cs at a fixed trap depth of 1 mK')
for Rsc, Rsc1064, color, symbol in [[RbRsc, RbRsc_Cs, 'tab:blue', 'Rb'], [CsRsc, CsRsc_Cs, 'tab:orange', 'Cs']]:
    ax3.semilogy(wavels*1e9, Rsc[:-1], color=color, label=symbol) 
    ax3.semilogy(wavels*1e9, np.zeros(len(wavels))+Rsc[-1], 
            '--', color=color, label=symbol+' at 880.2 nm') # for 1 mK Rb trap
    ax3.semilogy(wavels*1e9, np.zeros(len(wavels))+Rsc1064, 
            ':', color=color, label=symbol+' at 1064 nm')   # for 1 mK Cs trap
# ax3.plot(wavels*1e9, np.zeros(len(wavels))+100, 'k--', alpha=0.25) # show acceptable region
ax3.set_ylabel('Scattering rate ($s^{-1}$)')
ax3.set_ylim(1, 1e5)

# ax4 = ax3.twinx()
# ax4.plot(wavels*1e9, RbPowers[:-1]*1e3, 'tab:green') # show beam powers used to fix RB 1mK trap depth
# ax4.set_ylabel('Beam power to create \na 1 mK for Rb (mW)', color='tab:green')
# ax4.tick_params(axis='y', labelcolor='tab:green')

ax3.legend()

# plot the ratio of scattering rates Rb / Cs
ax5.semilogy(wavels*1e9, RbRsc[:-1]/CsRsc[:-1])
ax5.set_ylabel('Ratio of Rb / Cs\nscattering rates ')
ax5.set_xlabel('Wavelength (nm)')
ax5.set_xlim(wavels[0]*1e9, wavels[-2]*1e9)
plt.tight_layout()
plt.show()