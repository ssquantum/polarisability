"""Trying to calculate the Raman Rabi frequency between ground state hyperfine sublevels
not confirmed if the result is accurate yet."""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(r'..')
sys.path.append(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\polarisability')
from AtomFieldInt_V3 import c, eps0, hbar, h, a0, e, Rb, Cs, wigner3j, wigner6j, dipole

def singleRabi(dme, E, J, Jp, F, mf, Fp, mfp, I, q):
    return (-1)**(J+I+mf) * dme * np.sqrt((2*Fp+1)*(2*F+1)*(2*J+1)) *wigner6j(J, Jp, 1, Fp, F, I) *wigner3j(Fp, 1, F, mfp, -q, -mf) *E/hbar

def diffLightShift(atom, P, w0, J, F, mf, q, omega):
    """atom -- instance of AtomFieldInt_V3.atom()
    P -- power in W
    w0 -- beam waist in m
    J -- ground state orbital angular momentum number
    F -- hyperfine state 
    mf -- projection of hyperfine state 
    q -- circulation polarisation (0 for pi, +/-1 for sigma)
    omega -- detuning of the Raman beams from the one-photon transition
    """
    DE = 0
    E = np.sqrt(4*P / np.pi / eps0 / c / w0**2)
    for i in np.where(atom.nljS[:,0] == np.min(atom.nljS[:,0]))[0]: # range(len(atom.nljS)):  - should include all transitions but then it overestimates
        Jp = atom.nljS[i][2] 
        for Fp in range(int(abs(atom.I-Jp)), int(atom.I+Jp)+1):
            for mfp in range(-Fp, Fp+1):
                d = 0 # zero field hyperfine splitting
                if Fp == 0 and atom == Rb:
                    d = -302e6*2*np.pi
                elif Fp == 1 and atom == Rb:
                    d = -230e6*2*np.pi
                elif Fp == 2 and atom == Rb:
                    d = -73e6*2*np.pi
                elif Fp == 3 and atom == Rb:
                    d = 194e6*2*np.pi
                if Fp == 2 and atom == Cs:
                    d = -334e6*2*np.pi
                elif Fp == 3 and atom == Cs:
                    d = -188e6*2*np.pi
                elif Fp == 4 and atom == Cs:
                    d = 13e6*2*np.pi
                elif Fp == 5 and atom == Cs:
                    d = 264e6*2*np.pi
                # note: here we're using Steck's definition of the reduced DME
                # differs by sqrt(2) from e.g. Arora 2007's definition
                # print(Jp, Fp, mfp, singleRabi(1, hbar, J, Jp, F, mf, Fp, mfp, atom.I, q))
                DE += singleRabi(atom.D0S[i]/2**0.5, E, J, Jp, F, mf, Fp, mfp, atom.I, q)**2 / 4 / (atom.w0S[i] + d - omega)
    return DE

def RabiFreq(atom, P, w0, omega, J, Fa, mfa, Fb, mfb, qa, qb):
    """atom -- instance of AtomFieldInt_V3.atom()
    P -- power in W
    w0 -- beam waist in m
    omega -- Raman beam frequency in rad/s
    J -- ground state orbital angular momentum number
    Fa -- hyperfine state a
    mfa -- projection of hyperfine state a
    Fb -- hyperfine state b
    mfb -- projection of hyperfine state b
    q -- circulation polarisation (0 for pi, +/-1 for sigma)
    """
    Rabi = 0
    E = np.sqrt(4*P / np.pi / eps0 / c / w0**2)
    terms = [[E]] # for debugging
    for i in np.where(atom.nljS[:,0] == np.min(atom.nljS[:,0]))[0]: # range(len(atom.nljS)):  - should include all transitions but then it overestimates
        Jp = atom.nljS[i][2] 
        Delta = atom.w0S[i] - omega
        terms.append([Jp, atom.D0S[i]/2**0.5/e/a0, Delta/2/np.pi/1e9])
        for Fp in range(int(abs(atom.I-Jp)), int(atom.I+Jp)+1):
            for mfp in range(-Fp, Fp+1):
                # note: here we're using Steck's definition of the reduced DME
                # differs by sqrt(2) from e.g. Arora 2007's definition
                contribution = singleRabi(atom.D0S[i]/2**0.5, E, J, Jp, Fa, mfa, Fp, mfp, atom.I, qa) \
                    * singleRabi(atom.D0S[i]/2**0.5, E, J, Jp, Fb, mfb, Fp, mfp, atom.I, qb) /2 /Delta
                terms.append([Fp, mfp, singleRabi(1, hbar, J, Jp, Fa, mfa, Fp, mfp, atom.I, qa),
                    singleRabi(1, hbar, J, Jp, Fb, mfb, Fp, mfp, atom.I, qb), contribution])
                Rabi += contribution
    for i in range(len(terms)):
        if len(terms[i])==5: terms[i][-1] /= Rabi # fractional contribution
    terms = [[np.around(x, 3) for x in y] for y in terms]
    return (Rabi, terms)

power = 117.3e-6 # in W
waist = 80e-6 # in m
detun = 2*np.pi*(c/780.241209686e-9 - 50e9) # 30GHz off D2 line 780.241209686
print('Raman Rabi frequency: %.3g kHz'%(RabiFreq(Rb, power, waist, detun, 0.5, 1,1, 2,2, 1,0)[0] / 2/np.pi / 1e3)) 
# RabiFreq(Cs, 200e-6, 100e-6, 2*np.pi*(c/8.52347065e-7-30e9), 0.5, 3,3, 4,4, 1,0)[0]/2/np.pi/1e3
# for vals in RabiFreq(Rb, power, waist, detun, 0.5, 1,0, 2,0, 1,1)[1]:
#     print(*vals)

# compare trapping frequencies and Rabi frequencies between Rb/Cs
def wz(atom):
    """Axial trapping freq"""
    return np.sqrt(2*np.abs(atom.acStarkShift(0,0,0)/atom.m/atom.field.zR**2))

def wr(atom):
    """Radial trap freq"""
    return np.sqrt(4*np.abs(atom.acStarkShift(0,0,0)/atom.m/atom.field.w0**2))

# rb5s = dipole(Rb.m, (0,1/2.,1,1), [814e-9, 1.46e-3, 1e-6],
#                     Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
#                     nuclear_spin = Rb.I,
#                     symbol=Rb.X)

# print(rb5s.acStarkShift(0,0,0)/h/1e6/20.7)
# cs6s = dipole(Cs.m, (0,1/2.,3,3), [940e-9, 5.02e-3, 1.1e-6],
#                     Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
#                     nuclear_spin = Cs.I,
#                     symbol=Cs.X)

# print(cs6s.acStarkShift(0,0,0)/h/1e6/20.7)

# print((wz(rb5s) / wz(cs6s))**2)

# # # calculate differential stark shifts for Raman transition
power = 830e-6
print('Shift from RB2: %.3g kHz'%((diffLightShift(Cs, power, 110e-6, 0.5, 4, 4, 0, Cs.w0S[35]-(30-4.022)*2e9*np.pi) - diffLightShift(Cs, power, 110e-6, 0.5, 3, 3, 0, Cs.w0S[35]-(30+5.172)*2e9*np.pi))/2/np.pi/1e3))
power = 450e-6
print('Shift from RB1: %.3g kHz'%((diffLightShift(Cs, power, 110e-6, 0.5, 4, 4, 1, Cs.w0S[35]-(30-4.022)*2e9*np.pi) - diffLightShift(Cs, power, 110e-6, 0.5, 3, 3, 1, Cs.w0S[35]-(30+5.172)*2e9*np.pi))/2/np.pi/1e3))
# X = Cs
# F, mF, Fp, mFp = 3, 3, 4, 4
# wl =  8.52347065e-7 # wavelength of D2 line in m  852.347065e-9
# P1 = 200e-6 # in W
# P2 = 200e-6 # in W
# waist = 110e-6 # in m
# # lower hyperfine level of ground state
# atom = dipole(X.m, (0,1/2.,F,mF), [wl, P1, waist],
#                     X.D0S, X.w0S, X.lwS, X.nljS,
#                     nuclear_spin = X.I,
#                     symbol=X.X)
# # upper hyperfine level of ground state
# atom2 = dipole(X.m, (0,1/2.,Fp,mFp), [wl, P2, waist],
#                     X.D0S, X.w0S, X.lwS, X.nljS,
#                     nuclear_spin = X.I,
#                     symbol=X.X)

# for f in [30]:
#     detun = 2*np.pi*(c/wl - f*1e9)
#     detun2 = 2*np.pi*(c/wl - (f+9.192631770)*1e9)
#     print(f)
#     print('Raman Rabi frequency: %.3g kHz'%(RabiFreq(X, np.sqrt(P1*P2), waist, detun, 0.5, F,mF, Fp,mFp, 1,0)[0] / 2/np.pi / 1e3)) 
#     print('Scattering rate: %.3g /s'%(atom.scatRate(2*np.pi*c/detun)))
#     shift = atom.acStarkShift(0,0,0,2*np.pi*c/detun, HF=True)/h/1e3
#     print('AC Stark shift: %.3g kHz'%shift)
#     print('Differential Stark shift: %.3g kHz'%((diffLightShift(X, P1, waist, 0.5, Fp, mFp, 1, detun)+ diffLightShift(X, P2, waist, 0.5, Fp, mFp, 0, detun2)
#         - diffLightShift(X, P1, waist, 0.5, F, mF, 1, detun) - diffLightShift(X, P2, waist, 0.5, F, mF, 0, detun2))/2/np.pi/1e3))

# lamb-dicke parameter 
# momentum kick from perpendicular beams R1 + R2
# n = np.sqrt(hbar / 2 / Rb.m / 2/np.pi/20e3) * 2*np.pi/780e-9

# checking calculations against https://steck.us/alkalidata/rubidium87numbers.1.6.pdf
# sigmaplus = np.array([6, 24/5, 24/5, 24, 8, 4, 20, 40, 120, 12, 8, 8, 12, 30, 10, 5, 3, 2])**0.5
# pi = np.array([6, 24/5, 0, 24/5, 8, 6, 8, 40, 30, 40, 6, 24, 0, 24, 6, 6, 15/4, 10/3, 15/4, 6])**0.5
# i=0
# for F in [1,2]:
#     for Fp in range(F-1, F+2):       
#             for mf in range(-min(F,Fp), min(F,Fp)+1):
#                     print(F, mf, Fp, singleRabi(1,hbar,.5,1.5, F,mf,Fp,mf,1.5,0)*pi[i])
#                     i+=1

# Fung's thesis: 85Rb => I=5/2, |2,0> -> |3,0>, P1 = 55uW, P2 = 250uW, w0 = 80um, q1 = q2 = 1, detun = 30GHz from D2 line
# print('Fung Raman Rabi: %.3g kHz'%(4*np.sqrt(55e-6*250e-6)/eps0/c/np.pi/80e-6**2 * (4.227*e*a0)**2 *(np.sqrt(7/36/63) + np.sqrt(4*5/45/36)) /2 /2/np.pi/30e9 / hbar**2 /2/np.pi/1e3))
# print('Alternative formula: %.3g kHz'%((6.07e6)**2*(4*55*250)**0.5*1e-6/4/16.7/30e9/np.pi/(80e-6)**2 /1e3))


#### https://arxiv.org/pdf/1605.05230.pdf Fig 2 ####
# def R1(atom, J,F,mf,q,omega):
#     DE = 0
#     for i in np.where(atom.nljS[:,0] == np.min(atom.nljS[:,0]))[0]: # range(len(atom.nljS)):  - should include all transitions but then it overestimates
#         Jp = atom.nljS[i][2] 
#         for Fp in range(int(abs(atom.I-Jp)), int(atom.I+Jp)+1):
#             for mfp in range(-Fp, Fp+1):
#                 # note: here we're using Steck's definition of the reduced DME
#                 # differs by sqrt(2) from e.g. Arora 2007's definition
#                 d = 0
#                 if Fp == 0:
#                     d = -302e6*2*np.pi
#                 elif Fp == 1:
#                     d = -230e6*2*np.pi
#                 elif Fp == 2:
#                     d = -73e6*2*np.pi
#                 elif Fp == 3:
#                     d = 194e6*2*np.pi
#                 # print(Jp, Fp, mfp, singleRabi(1, hbar, J, Jp, F, mf, Fp, mfp, atom.I, q), singleRabi(atom.D0S[i], hbar, J, Jp, F, mf, Fp, mfp, atom.I, q)**2 / 4 / (atom.w0S[i] - omega))
#                 DE += singleRabi(atom.D0S[i], hbar, J, Jp, F, mf, Fp, mfp, atom.I, q)**2 / 4 / (atom.w0S[i]+d - omega)
#     return DE

# D = np.linspace(0.1,3.5,100); plt.plot(-D, -(R1(Rb,0.5,2,1,1,Rb.w0S[5]-(D)*2e9*np.pi) + R1(Rb,0.5,2,1,-1,Rb.w0S[5]-(D)*2e9*np.pi) - R1(Rb,0.5,1,-1,1,Rb.w0S[5]-(D+6.834)*2e9*np.pi) - R1(Rb,0.5,1,-1,-1,Rb.w0S[5]-(D+6.834)*2e9*np.pi))/ (R1(Rb,0.5,2,1,1,Rb.w0S[5]-(D-6.834)*2e9*np.pi) + R1(Rb,0.5,2,1,-1,Rb.w0S[5]-(D-6.834)*2e9*np.pi) - R1(Rb,0.5,1,-1,1,Rb.w0S[5]-(D)*2e9*np.pi) - R1(Rb,0.5,1,-1,-1,Rb.w0S[5]-(D)*2e9*np.pi)) ); plt.show()
# D = np.linspace(0.1,30,100); plt.plot(-D, (1/D-1/(D+6.834))/(1/(D-6.834)-1/D)); plt.show()
# D = np.linspace(-10,10,100); plt.plot(-D, (6.834-D)/(D+6.834)); plt.show()