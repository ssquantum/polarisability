"""Stefan Spence 13.11.18
17.01.18 -- Run to launch UI to calculate stark shifts in Rb/Cs

Simulation of atoms in an optical tweezer.
1) Formulate the equations for Gaussian beam propagation.
2) Look at the dipole interaction as a function of laser wavelength and 
spatial position
3) calculate the polarisability for a given state at a given wavelength

14.11.18 add in dipole potential
calculate the polarisability and compare the 2-level model to including other transitions

19.11.18 extend polarisability function
now it allows several laser wavelengths and several resonant transitions 
added Boltzmann's constant to global variables to convert Joules to Kelvin

20.11.18
make the polarisability function work for multiple or individual wavelengths
correct the denominator in the polarisability function from Delta^2 - Gamma^2
to Delta^2 + Gamma^2

13.12.18
Allow the dipole.polarisability() to take wavelength as an argument

18.12.18
The previous polarisability results did not agree with literature. Since the
stark shift is the priority, load the polarisability from provided data.
Also must include vector and tensor polarizabilities
see F. L. Kien et al, Eur. Phys. J. D, 67, 92 (2013)

21.12.18
Some papers include the Stark shift for hyperfine states (Kien 2013), whereas
others use just the fine structure (see B. Arora et al, Phys. Rev. A 76, 052509 
(2007))
So we will incorporate functions for both of them.

02.01.19
Use arc (see https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/ ) 
to get the data for dipole matrix elements and transition properties
(note that arc gets all its Rb, Cs literature values from Safronova papers:
Safronova et al, PRA 60, 6 (1999)
Safronova et al, PRA 69, 022509 (2004)

07.01.19
Add in functions to calculate the polarisability
 - when the hyperfine transitions are important (not finished - needs dipole
 matrix elements for hyperfine transitions): polarisability()
 - when hyperfine transitions can be ignored: polarisabilityJ()
Arora 2007 does include hyperfine splittings in a separate equations, so make
one acStarkShift() function where hyperfine interactions can be toggled

08.01.19
Remove the duplicate starkshift/polarisability functions

14.01.19
Give state labels (n,l,j) to the transition data

15.01.19
Correct the polarisability formula (had w - w0 instead of w0 - w)
Note: the arc data doesn't have linewidths for transitions
Since they're usually quite small this usually doesn't make too much of a 
difference [Archived this version]

16.01.19
Remove functions for loading polarisability data from other papers
Store transition data in dat files so that importing arc is unnecessary

17.01.19
explicitly state that the denominator in fractions are floats, otherwise there
is integer division 

23.01.19 
write a function to match the figures for polarisability from Arora 2007
Correct a factor of 1/2 in the polarisability formula to match Arora 2007
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from math import factorial 

# global constants:
c    = 2.99792458e8  # speed of light in m/s
eps0 = 8.85419e-12   # permittivity of free space in m^-3 kg^-1 s^4 A^2
h    = 6.6260700e-34 # Planck's constant in m^2 kg / s
hbar = 1.0545718e-34 # reduced Planck's constant in m^2 kg / s
a0 = 5.29177e-11     # Bohr radius in m
e = 1.6021766208e-19 # magnitude of the charge on an electron in C
me = 9.10938356e-31  # mass of an electron in kg
kB = 1.38064852e-23  # Boltzmann's constant in m^2 kg s^-2 K^-1
amu = 1.6605390e-27  # atomic mass unit in kg
Eh = me * e**4 /(4. *np.pi *eps0 *hbar)**2  # the Hartree energy
au = e**2 * a0**2 / Eh # atomic unit for polarisability
# note that atomic unit au = 4 pi eps0 a0^3

def tric(a, b, c):
    """Return the value of the triangle coefficient. Used for Wigner 6j symbol"""
    return (factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) 
                    )/ float(factorial(a + b + c + 1))
                
def wigner3j(j1, j2, j3, m1, m2, m3):
    """Return the value of the Wigner 3j symbol. m quantum numbers must be 
    within -j,..,j, and satisfy m1+m2=m3, the js must satisfy the triangle
    inequality, and sum(js) must be integer."""
    
    # conditions for the value to be non-zero:
    if abs(m1) > abs(j1) or abs(m2) > abs(j2) or abs(m3) > abs(j3):
        return 0
        
    elif m1 + m2 != -m3 or abs(j1 - j2) > j3 or j1 + j2 < j3 or (j1+j2+j3)%1!=0:
        return 0    
    
    facts = np.sqrt(tric(j1,j2,j3)) # factor of sqrt factorials in front of sum
    for j, m in [[j1,m1], [j2,m2], [j3,m3]]:
        facts *= np.sqrt(factorial(j + m) * factorial(j - m))
        
    tsum = 0
    for t in range(int(j1 + j2 + j3)):
        try:
            tsum += (-1)**t /float(factorial(t)) /float(factorial(j3-j2+t+m1)
                )/float(factorial(j3-j1+t-m2)) /float(factorial(j1+j2-j3-t)
                ) /float(factorial(j1-t-m1)) /float(factorial(j2-t-m2))
        except ValueError:
            # sum is only over positive factorials
            tsum += 0
            
    return (-1)**(j1 - j2 - m3) * facts * tsum

def wigner6j(j1, j2, j3, J1, J2, J3):
    """Return the value of the Wigner 6j symbol. Triads must satisfy the 
    triangle inequalities and sum to an integer, otherwise the wigner 6j is 0."""
    
    tripls = 1      # factor of triangle coefficients in front of the sum
    for v in [[j1,j2,j3],[j1,J2,J3],[J1,j2,J3],[J1,J2,j3]]:
        if v[2] < abs(v[1]-v[0]) or v[2] > v[1] + v[0] or sum(v)%1 != 0:
            return 0   # must satisfy triangle inequality and sum to an integer
            
        else:
            tripls *= np.sqrt(tric(*v))
    
    tsum = 0
    for t in range(int(round(j1+j2++j3+J1+J2+J3+1))):
        try:
            tsum += (-1)**t * factorial(t+1) /float(factorial(t-j1-j2-j3)
            )/float(factorial(t-j1-J2-J3)) /float(factorial(t-J1-j2-J3)
            )/ float(factorial(t-J1-J2-j3)) /float(factorial(j1+j2+J1+J2-t)
            ) /float(factorial(j2+j3+J2+J3-t)) / float(factorial(j1+j3+J1+J3-t))
        except ValueError:
            # sum is only over positive factorials
            tsum += 0
    
    return tripls * tsum
    

#####################
    
    
class atom:
    """Properties of an atom: 
    
    The transitions follow the order:
    S1/2 -> nP1/2, nP3/2
    P1/2 -> nS1/2. nD3/2
    P3/2 -> nS1/2, nD3/2, nD5/2
    
    D0: Dipole matrix elements (C m)
    rw: resonant wavelength (m) of transitions 
    w0: resonant frequency (rad/s) of transitions 
    lw: natural linewidth (rad/s) of transitions 
    """
    def __init__(self, mass, nuclear_spin, symbol, S1_2DME, P1_2DME, P3_2DME,
                S1_2RW, P1_2RW, P3_2RW, S1_2LW, P1_2LW, P3_2LW,
                S1_nlj, P1_nlj, P3_nlj):
        self.D0S = S1_2DME  # dipole matrix elements from S1/2 state
        self.D0P1 = P1_2DME # dipole matrix elements from P1/2 state
        self.D0P3 = P3_2DME # dipole matrix elements from P3/2 state
        self.nljS = S1_nlj # (n,l,j) quantum numbers for transitions
        self.nljP1 = P1_nlj
        self.nljP3 = P3_nlj
        self.rwS = S1_2RW   # resonant wavelengths from S1/2 state (m)
        self.rwP1 = P1_2RW  # resonant wavelengths from P1/2 state (m)
        self.rwP3 = P3_2RW  # resonant wavelengths from P3/2 state (m)
        self.w0S = 2*np.pi*c / S1_2RW # resonant frequency (rad/s)
        self.w0P1 = 2*np.pi*c / P1_2RW
        self.w0P3 = 2*np.pi*c / P3_2RW
        self.lwS = S1_2LW   # natural linewidth from S1/2 (rad/s)
        self.lwP1 = P1_2LW  # natural linewidth from P1/2 (rad/s)
        self.lwP3 = P3_2LW  # natural linewidth from P3/2 (rad/s)
        self.m  = mass
        self.I  = nuclear_spin
        self.X  = symbol
        

######### atomic properties for Cs-133:  ##########
# file contains columns: n, l, j, dipole matrix element, wavelength, linewidth
# for the 6S1/2 state:
S1_2 = np.loadtxt(r'.\TransitionData\CsS1_2.dat', delimiter=',', skiprows=1)
        
# for the 6P1/2 state:
P1_2 = np.loadtxt(r'.\TransitionData\CsP1_2.dat', delimiter=',', skiprows=1)

# for the 6P3/2 state:
P3_2 = np.loadtxt(r'.\TransitionData\CsP3_2.dat', delimiter=',', skiprows=1)


Cs = atom( S1_2DME = S1_2[:,3], P1_2DME = P1_2[:,3], P3_2DME = P3_2[:,3], # matrix elements
    S1_2RW = S1_2[:,4], P1_2RW = P1_2[:,4], P3_2RW = P3_2[:,4], # resonant wavelengths
    S1_2LW = S1_2[:,5], P1_2LW = P1_2[:,5], P3_2LW = P3_2[:,5], # natural linewidths
    S1_nlj = S1_2[:,:3], P1_nlj = P1_2[:,:3], P3_nlj = P3_2[:,:3], # final state of transition
    mass = 133*amu,        # mass in kg
    nuclear_spin = 7/2.,   # intrinsic angular momentum quantum number of the nucleus
    symbol = 'Cs')


######### atomic properties for Rb-87:  ###########
# file contains columns: n, l, j, dipole matrix element, wavelength, linewidth
# for the 6S1/2 state:
S1_2 = np.loadtxt(r'.\TransitionData\RbS1_2.dat', delimiter=',', skiprows=1)
        
# for the 6P1/2 state:
P1_2 = np.loadtxt(r'.\TransitionData\RbP1_2.dat', delimiter=',', skiprows=1)

# for the 6P3/2 state:
P3_2 = np.loadtxt(r'.\TransitionData\RbP3_2.dat', delimiter=',', skiprows=1)

Rb = atom( S1_2DME = S1_2[:,3], P1_2DME = P1_2[:,3], P3_2DME = P3_2[:,3], # matrix elements
    S1_2RW = S1_2[:,4], P1_2RW = P1_2[:,4], P3_2RW = P3_2[:,4], # resonant wavelengths
    S1_2LW = S1_2[:,5], P1_2LW = P1_2[:,5], P3_2LW = P3_2[:,5], # natural linewidths
    S1_nlj = S1_2[:,:3], P1_nlj = P1_2[:,:3], P3_nlj = P3_2[:,:3], # final state of transition
    mass = 87*amu,        # mass in kg
    nuclear_spin = 3/2.,   # intrinsic angular momentum quantum number of the nucleus
    symbol = 'Rb')


#######################


class Gauss:
    """Properties and associated equations of a Gaussian beam"""
    def __init__(self, wavelength, power, beam_waist, polarization=(0,0,1)):
        self.lam = wavelength    # wavelength of the laser light (in metres)
        self.P   = power         # total power of the beam (in Watts)
        self.w0  = beam_waist    # the beam waist defines the laser mode (in metres)
        self.ehat= polarization  # the direction of polarization (assume linear)
        # note: we will mostly ignore polarization since the induced dipole 
        # moment will be proportional to the direction of the field
        
        # assume that the beam waist is positioned at z0 = 0
        
        # from these properties we can deduce:
        self.zR = np.pi * beam_waist**2 / wavelength # the Rayleigh range
        self.E0 = 2/beam_waist * np.sqrt(power / eps0 / c / np.pi) # field amplitude at the origin
        self.k  = 2 * np.pi / wavelength             # the wave vector
        
    def amplitude(self, x, y, z):
        """Calculate the amplitude of the Gaussian beam at a given position
        note that this function will not work if several coordinates are 1D arrays
        instead, loop over the other coordinates so that there is only ever one
        coordinate as an array."""
        rhosq = x**2 + y**2                     # radial coordinate squared    
        q     = z - 1.j * self.zR               # complex beam parameter
        
        # Gaussian beam equation (see Optics f2f Eqn 11.7)
        return self.zR /1.j /q * self.E0 * np.exp(1j * self.k * z) * np.exp(
                                                1j * self.k * rhosq / 2. / q)
        

#######################
        
        
class dipole:
    """Properties and equations of the dipole interaction between atom and field"""
    def __init__(self, mass, spin_state, field_properties,
                    dipole_matrix_elements, resonant_frequencies, decay_rates,
                    transition_labels, nuclear_spin=7/2.,
                    symbol="Cs"):
        self.m                          = mass * amu           # mass of the atom in kg
        self.L, self.J, self.F, self.MF = spin_state           # spin quantum numbers L, J, F, M_F
        self.I                          = nuclear_spin         # nuclear spin quantum number I
        self.field = Gauss(*field_properties)                  # combines all properties of the field
        self.X = symbol
        
        self.states = transition_labels                 # (n,l,j) quantum numbers for transitions
        self.omega0 = np.array(resonant_frequencies)    # resonant frequencies (rad/s)
        self.gam   = np.array(decay_rates)              # spontaneous decay rate (s)
        self.D0s   = np.array(dipole_matrix_elements)   # D0 = -e <a|r|b> for displacement r along the polarization direction
        self.omegas = np.array(2*np.pi*c/self.field.lam)# laser frequencies (rad/s)
        
            
    def acStarkShift(self, x, y, z, wavel=[], mj=None, HF=False):
        """Return the potential from the dipole interaction 
        U = -<d>E = -1/2 Re[alpha] E^2
        Then taking the time average of the cos^2(wt) AC field term we get 
        U = -1/4 Re[alpha] E^2"""
        return -self.polarisability(wavel, mj, HF, split=False) /4. *np.abs( 
                            self.field.amplitude(x,y,z) )**2
    
            
    def polarisability(self, wavel=[], mj=None, HF=False, split=False):
        """Return the polarisability as given Arora 2007 (also see Cooper 2018,
        Mitroy 2010, Kein 2013) assuming that J and mj are good quantum 
        numbers when hyperfine splitting can be neglected, or that F and mf are
        good quantum numbers. Assumes linear polarisation so that the vector
        polarisability is zero."""
        if np.size(wavel) != 0:            
            omegas = np.array(2*np.pi*c/wavel)# laser frequencies (rad/s)
        else:
            omegas = self.omegas
        
        # initiate arrays for results
        empty = np.zeros(np.size(omegas))
        aSvals, aVvals, aTvals = empty.copy(), empty.copy(), empty.copy()
        
        for ii in range(np.size(omegas)):
            aS, aV, aT = 0, 0, 0
            
            # loop over final states
            for i in range(len(self.states)):                    
                if np.size(omegas) > 1:
                    Ep = hbar*(self.omega0[i] + omegas[ii] + 1j*self.gam[i])
                    Em = hbar*(self.omega0[i] - omegas[ii] - 1j*self.gam[i])
                
                else:
                    Ep = hbar*(self.omega0[i] + omegas + 1j*self.gam[i])
                    Em = hbar*(self.omega0[i] - omegas - 1j*self.gam[i])
                    
                
                aS += 1/3. /(2.*self.J + 1.) *self. D0s[i]**2 * (1/Ep + 1/Em)
                
                aV += 0.5*(-1)**(self.J + 2 + self.states[i][2]) * np.sqrt(6*self.J
                    /(self.J + 1) /(2*self.J + 1)) * self.D0s[i]**2 * wigner6j(
                    1, 1, 1, self.J, self.states[i][2], self.J) * (1/Em - 1/Ep)
                
                aT += 2*np.sqrt(5 * self.J * (2*self.J - 1) / 6. /(self.J + 1) /
                    (2*self.J + 1) / (2*self.J + 3)) * (-1)**(self.J + 
                    self.states[i][2]) * wigner6j(self.J, 1, self.states[i][2], 
                    1, self.J, 2) * self.D0s[i]**2 * (1/Ep + 1/Em)
                    
                i += 1
      
            aSvals[ii] = aS.real  # scalar polarisability
            aVvals[ii] = aV.real  # vector polarisability
            aTvals[ii] = aT.real  # tensor polarisability
        
        if split:
            return (aSvals, aVvals, aTvals)
        
        # combine polarisabilities
        # NB: currently ignoring vector polarisability as in Arora 2007
        if self.J > 0.5:
            if HF: # hyperfine splitting is significant
                return aSvals + np.sqrt((self.J + 1)*(2*self.J + 1)*(2*self.J + 
                    3)/self.J /(2*self.J - 1)) *(-1)**(self.I + self.J - self.MF
                    ) * (2*self.F + 1) * wigner6j(self.F, 2, self.F, self.J, 
                    self.I, self.J) * wigner3j(self.F, 2, self.F, self.MF, 0, 
                    -self.MF) * aTvals
                    
            else: # hyperfine splitting is ignored
                return aSvals + aTvals * (3*mj**2 - self.J*(self.J + 1)
                ) / self.J / (2*self.J - 1)
        else:
            # there is no tensor polarisability for the J=1/2 state
            return aSvals
                            
        
#############################

##### example functions:#####

#############################
   
def getMagicWavelengths(deltaE, E, wavelengths):
    """Find the magic wavelengths where the energy difference is zero.
    Define this where the fractional difference |deltaE/E| < 0.05 and the 
    difference deltaE changes sign"""

    magicWavelengths = []
    magicindexes = np.where(abs(deltaE/E)<0.05)[0]
    
    for mi in magicindexes:
        if np.sign(deltaE[mi]) == -np.sign(deltaE[mi+1]):
            magicWavelengths.append(wavelengths[mi])
    
    return magicWavelengths
     
def plotStarkShifts():
    """Use loaded polarisabilities to find the ac Stark Shifts in Rb, Cs"""
    # typical optical tweezer parameters:
    wavelength = 880e-9                 # laser wavelength in nm
    beamwaist = 1e-6                    # beam waist in m
    power = 20e-3                       # power in Watts 
    bprop = [wavelength, power, beamwaist] # collect beam properties
    
    Rb5S = dipole(Rb.m, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
    
    Rb5P = dipole(Rb.m, (1,3/2.,1,1), bprop,
                    Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
                    
    Cs6S = dipole(Cs.m, (0,1/2.,3,3), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
                    
    Cs6P = dipole(Cs.m, (1,3/2.,3,3), bprop,
                    Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
    
    # need a small spacing to resolve the magic wavelengths - so it will run slow
    # to resolve magic wavelengths, take about 10,000 points.
    wavels = np.linspace(700e-9, 1100e-9, 500) 
    
    # ac Stark Shift in Joules:
    dE6S = Cs6S.acStarkShift(0,0,0,wavels, mj=0.5, HF=False)
    dE6P = Cs6P.acStarkShift(0,0,0,wavels, mj=1.5, HF=False)
    dif6P = dE6P - dE6S
    
    magic6P = getMagicWavelengths(dif6P, dE6P, wavels)
    
    plt.figure()
    plt.title("AC Stark Shift in $^{133}$Cs")
    plt.plot(wavels*1e9, dE6S/h*1e-6, 'b--', label='Ground S$_{1/2}$')
    plt.plot(wavels*1e9, dE6P/h*1e-6, 'r-.', label='Excited P$_{3/2}$')
    plt.plot(wavels*1e9, (dif6P)/h*1e-6, 'k', label='Difference')
    plt.plot([magic6P[0]*1e9]*2, [min(dif6P/h/1e6),max(dif6P/h/1e6)], 'm:',
                label = 'Magic Wavelength')
    plt.legend()
    for mw in magic6P[1:]:
        plt.plot([mw*1e9]*2, [min(dif6P/h/1e6),max(dif6P/h/1e6)], 'm:')
    plt.ylabel("Stark Shift (MHz)")
    plt.xlabel("Wavelength (nm)")
    plt.xlim(wavels[0]*1e9, wavels[-1]*1e9)
    plt.ylim(-2200,2200)
    plt.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.25) # show zero crossing
    plt.show()
    print("Magic wavelengths at:\n", magic6P)
    
    
    # ac Stark Shift in Joules:
    dE5S = Rb5S.acStarkShift(0,0,0,wavels, mj=0.5, HF=False)
    dE5P = Rb5P.acStarkShift(0,0,0,wavels, mj=1.5, HF=False)
    dif5P = dE5P - dE5S

    plt.figure()
    plt.title("AC Stark Shift in $^{87}$Rb")
    plt.plot(wavels*1e9, dE5S/h*1e-6, 'b--', label='Ground S$_{1/2}$')
    plt.plot(wavels*1e9, dE5P/h*1e-6, 'r-.', label='Excited P$_{3/2}$')
    plt.plot(wavels*1e9, (dif5P)/h*1e-6, 'k', label='Difference')
    plt.legend()
    plt.ylabel("Stark Shift (MHz)")
    plt.xlabel("Wavelength (nm)")
    plt.ylim(-500,500)
    plt.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.25) # show zero crossing
    plt.show()
    
def compareArora():
    """Plot Fig 5 and Fig 7 in Arora et al 2007 to show that the polarisabilities 
    of Rb and Cs without hyperfine levels are correct"""
    # beam properties: wavelength, power, beam waist
    bprop = [1064e-9, 35e-3, 1.05e-6]
    
    # for some reason to match these graphs the dipole matrix elements have to 
    # be modified
    # factor = np.sqrt(2)
    # Cs.D0P3[37] = Cs.D0P3[37] * factor
    # Cs.D0P3[74] = Cs.D0P3[74] * factor
    for ATOM in [Rb, Cs]:
        S = dipole(ATOM.m, (0,1/2.,1,1), bprop,
                        ATOM.D0S, ATOM.w0S, ATOM.lwS, ATOM.nljS,
                        nuclear_spin = ATOM.I,
                        symbol=ATOM.X)
        
        P3 = dipole(ATOM.m, (1,3/2.,3,3), bprop,
                        ATOM.D0P3, ATOM.w0P3, ATOM.lwP3, ATOM.nljP3,
                        nuclear_spin = ATOM.I,
                        symbol=ATOM.X)
                        
        if ATOM == Rb:
            wavel = np.linspace(780, 800, 200)*1e-9
            Ylim = (-8000, 8000)
        elif ATOM == Cs:
            wavel = np.linspace(925, 1000, 200)*1e-9
            Ylim = (-1000, 5000)
        plt.figure()
        plt.title("Polarisability of "+ATOM.X)
        plt.plot(wavel*1e9, S.polarisability(wavel)/au, 'r', label='s')
        plt.plot(wavel*1e9, P3.polarisability(wavel,mj=0.5)/au, 'g--', label='p$_{3/2}$, mj=1/2')
        plt.plot(wavel*1e9, P3.polarisability(wavel,mj=1.5)/au, 'm:', label='p$_{3/2}$, mj=3/2')
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Polarisability (a.u.)")
        plt.ylim(Ylim)
        plt.xlim(wavel[0]*1e9, wavel[-1]*1e9)
        plt.show()
    

def getStarkShift(obj):
    """Print the ac Stark Shift for all of the hyperfine levels in a particular
    fine structure state of the atom in dipoleObject"""
    Lterms = ['S', 'P', 'D', 'F', 'G'] # labels for angular momentum states
    
    # show some important parameters
    outstring = obj.X + " " + Lterms[obj.L] + str(int(obj.J*2)
            ) + "/2 ac Stark Shift at %.0f nm for E field %.2g V/m:\n"%(
            obj.field.lam*1e9, obj.field.E0)
            
    outstring += "\nIf hyperfine splitting is insignificant:\n"
    for MJ in np.arange(1, 2*obj.J+1, 2).astype(int): # NB: this is 2*MJ
        outstring += "MJ = "+str(MJ)+"/2 : %.5g MHz\n"%(
            obj.acStarkShift(0,0,0, obj.field.lam, mj=MJ/2., HF=False)/h/1e6)
            
        
    outstring += "\nIf hyperfine splitting is significant:\n"
    for F in range(int(abs(obj.I - obj.J)), int(obj.I + obj.J+1)):
        if F > 0:
            obj.F, obj.MF = F, F
            outstring += "|"+str(F)+","+str(F)+">  : %.5g MHz\n"%(
                    obj.acStarkShift(0,0,0, obj.field.lam, HF=True)/h/1e6)
                    
    return outstring
                                                    
def runGUI():
    """A UI to get the stark shift from a user supplied state"""
    import tkinter
    from tkinter import messagebox
    
    root = tkinter.Tk()  # main window
    frames, labels, entries = [], [], [] # the user must enter variables
    labeltext = ['Wavelength (m): ', 'Beam waist (m): ', 'Beam power (W): ', 
            'Atom (Rb/Cs): ', 'Orbital angular momentum L: ', 
            'Total angular momentum J: ']
    entrystrings = [tkinter.StringVar() for i in range(len(labeltext))]
    default = ['880e-9', '1e-6', '20e-3', 'Rb', '0', '0.5']
    
    
    for i in range(len(labeltext)):
        frames.append(tkinter.Frame(root)) # frame for placing entries
        frames[-1].pack(side=tkinter.TOP)  # position in descending order
        labels.append(tkinter.Label(frames[-1], text=labeltext[i]))
        labels[-1].pack(side=tkinter.LEFT) # position label on left
        entries.append(tkinter.Entry(frames[-1], textvariable=entrystrings[i]))
        entrystrings[i].set(default[i])    # set default string text
        entries[-1].pack(side=tkinter.RIGHT)# text entry on right
    
    
    def showResult():
        wavelength = float(entrystrings[0].get())  # laser wavelength in nm
        beamwaist = float(entrystrings[1].get())   # beam waist in m
        power = float(entrystrings[2].get())       # power in Watts 
        bprop = [wavelength, power, beamwaist] # collect beam properties
        atomSymbol = entrystrings[3].get()         # choose Rb or Cs
        L = int(entrystrings[4].get())             # orbital angular momentum
        J = float(entrystrings[5].get())             # total angular momentum
        
        # choose element
        if atomSymbol == "Rb":
            atomObj = Rb
            F = 1
        elif atomSymbol == "Cs":
            atomObj = Cs
            F = 3
        else:
            messagebox.showinfo("Error", "You must choose Rb or Cs")
            return 0
        
        # get transition data for the given state
        if L == 0:
            D0, w0, lw, nlj = atomObj.D0S, atomObj.w0S, atomObj.lwS, atomObj.nljS
        elif L == 1 and J == 0.5:
            D0, w0, lw, nlj = atomObj.D0P1, atomObj.w0P1, atomObj.lwP1, atomObj.nljP1
        elif L == 1 and J == 1.5:
            D0, w0, lw, nlj = atomObj.D0P3, atomObj.w0P3, atomObj.lwP3, atomObj.nljP3
        
        # construct the instance of the dipole class
        dipoleObj = dipole(atomObj.m, (L,J,F,F), bprop,
                D0, w0, lw, nlj,
                nuclear_spin = atomObj.I,
                symbol=atomObj.X)
        
        messagebox.showinfo("Calculation Result", getStarkShift(dipoleObj))
        
    resultButton = tkinter.Button(root, text="Calculate Stark Shifts", 
                                    command=showResult)
    resultButton.pack(side = tkinter.BOTTOM)
    
    root.mainloop()

if __name__ == "__main__":
    print("compare Fung Yin 2016 thesis stark shifts for 85 Rb:")
    # beam properties: wavelength, power, beam waist
    bprop = [1064e-9, 35e-3, 1.05e-6]
    
    # for the 5S 1/2 ground state:
    nlj5S = [[5,1,0.5], [5,1,1.5]]              # final states
    # dipole matrix elements (Cm) - note different definition of DME
    # (J|d|J) = sqrt(2J + 1) <J|d|J>
    dme5S = np.array([2.53574447e-29, 3.57770876e-29]) * np.sqrt(2)
    rwl5S = np.array([794.98, 780.24])*1e-9         # resonant wavelengths (m)
    w05S = 2*np.pi*c / rwl5S             # resonant wavelengths (rad/s)
    
    # for the 5P 1/2 excited state:
    nlj5P1 = [[4,2,1.5], [5,2,1.5], [6,0,0.5], [6,2,1.5], [7,0,0.5], [8,0,0.5]]
    dme5P1 = np.array([3.40587727e-29, 1.81107703e-29, 1.75783958e-29, 
            6.73795221e-30, 6.35609943e-30, 2.12837967e-30,]) * 2
    rwl5P1 = np.array([1475.86, 762.1, 1323.69, 620.8, 728.2, 607.24])*1e-9
    w05P1 = 2* np.pi *c / rwl5P1
    
    # for the 5P 3/2 excited state:   
    nlj5P3 = [[4,2,1.5], [4,2,2.5], [5,2,1.5], [5,2,2.5], [6,0,0.5], [6,2,1.5],
            [6,2,2.5], [7,0,0.5], [8,0,0.5]]
    dme5P3 = np.array([1.53622915e-29, 4.61519230e-29, 2.79821372e-30, 8.39047079e-30,
       2.56124969e-29, 2.37486842e-30, 7.03562364e-30, 5.68330890e-30,
       2.96816442e-30]) * 2
    rwl5P3 = np.array([1529.26, 1529.37, 776.16, 775.97, 1367.67, 630.1, 
                630.02, 741.02, 616.13]) * 1e-9
    w05P3 = 2* np.pi *c / rwl5P3
           
    # construct 85 Rb 5S 1/2:     
    Rb5S = dipole(85*amu, [0,0.5,2,2], bprop, dme5S, w05S, np.zeros(len(dme5S)),
                    nlj5S, 5/2., "Rb")
                    
    # construct 85 Rb 5P 1/2:
    Rb5P1 = dipole(85*amu, [1,0.5,2,2], bprop, dme5P1, w05P1, np.zeros(len(dme5P1)),
                    nlj5P1, 5/2., "Rb")

    # construct 85 Rb 5P 3/2:                    
    Rb5P3 = dipole(85*amu, [1,1.5,2,2], bprop, dme5P3, w05P3, np.zeros(len(dme5P3)),
                    nlj5P3, 5/2., "Rb")
    
    print(getStarkShift(Rb5S))
    
    print(getStarkShift(Rb5P1))
    
    print(getStarkShift(Rb5P3))
    
    
    # compare to previous results:
    Rb5S = dipole(85*amu, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = 5/2.,
                    symbol=Rb.X)
    Rb5P1 = dipole(85*amu, (1,1/2.,1,1), bprop,
                    Rb.D0P1, Rb.w0P1, Rb.lwP1, Rb.nljP1,
                    nuclear_spin = 5/2.,
                    symbol=Rb.X)
    Rb5P3 = dipole(85*amu, (1,3/2.,3,3), bprop,
                    Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                    nuclear_spin = 5/2.,
                    symbol=Rb.X)
    print("----------Previous Results:-----------")
    print(getStarkShift(Rb5S))
    print(getStarkShift(Rb5P1))
    print(getStarkShift(Rb5P3))