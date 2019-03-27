"""Stefan Spence 13.11.18
17.01.18 -- Run to launch UI to calculate stark shifts in Rb/Cs

Version 3: the calculations are in agreement with Arora 2007 when the hyperfine
splitting can be ignored. However, this version assumes that when hyperfine 
splitting is relevant, it is much larger than the stark shift, so that the
stark shift hamiltonian is diagonal in the hyperfine basis (no state mixing).
This assumption doesn't hold for the excited states.

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

29.01.19
When the hyperfine boolean is True, use the formula from Kien 2013

04.02.19
use Arora 2007 for hyperfine 

20.03.19
Also print the polarisability components in getStarkShift()

27.03.19
When looking at excited states with several possible mj values, average
over the possible mj values.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from math import factorial 
from matplotlib.ticker import AutoLocator

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
        # average intensity of sinusoidal wave gives the factor of 2
        self.E0 = 2 * np.sqrt(power / eps0 / c / np.pi)/beam_waist  # field amplitude at the origin
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
        """wavel: wavelength (m) - default is self.field.lam
        mj: used when hyperfine splitting is negligible.
        HF: Boolean - include hyperfine structure
        split: Boolean - False gives total polarisability, True splits into
        scalar, vector, and tensor.
        Return the polarisability as given Arora 2007 (also see Cooper 2018,
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
                    /(self.J + 1.) /(2*self.J + 1.)) * self.D0s[i]**2 * wigner6j(
                    1, 1, 1, self.J, self.states[i][2], self.J) * (1/Em - 1/Ep)
                    
                aT += 2*np.sqrt(5 * self.J * (2*self.J - 1) / 6. /(self.J + 1) /
                        (2*self.J + 1) / (2*self.J + 3)) * (-1)**(self.J + 
                        self.states[i][2]) * wigner6j(self.J, 1, self.states[i][2], 
                        1, self.J, 2) * self.D0s[i]**2 * (1/Ep + 1/Em)
                    
      
            aSvals[ii] = aS.real  # scalar polarisability
            aVvals[ii] = aV.real  # vector polarisability
            aTvals[ii] = aT.real  # tensor polarisability

        # combine polarisabilities
        
        if self.J > 0.5:
            if HF:  # hyperfine splitting is significant
                # from Kien 2013: when stark shift << hfs splitting so there isn't mixing of F levels
                # combine eq 16 and 18 to get the a_nJF in terms of the a_nJ
                # also assume stark shift << Zeeman splitting so we can use |F,MF> states.
                aVvals *= -(-1)**(self.J + self.I + self.F) * np.sqrt(self.F * (2*self.F + 1)
                    *(self.J + 1) *(2*self.J + 1) /self.J /(self.F + 1)) *wigner6j(self.F, 1, self.F, 
                    self.J, self.I, self.J)
                
                # from Arora 2007
                aTvals *= (-1)**(self.I + self.J - self.MF) * (2*self.F + 1
                    ) * np.sqrt((self.J + 1) *(2*self.J + 1) *(2*self.J + 3)
                    /self.J /(2*self.J - 1.)) * wigner3j(self.F, 2, self.F, 
                    self.MF, 0, -self.MF) * wigner6j(self.F, 2, self.F,
                    self.J, self.I, self.J)  
                
                if split:
                    return (aSvals, aVvals, aTvals)
                else:        
                    return aSvals + aTvals
                
            else: # hyperfine splitting is ignored
                # NB: currently ignoring vector polarisability as in Arora 2007
                if split:
                    return (aSvals, aVvals, aTvals)
                else:
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
     
def plotStarkShifts(wavelength = 880e-9,             # laser wavelength in nm
                    beamwaist = 1e-6,                # beam waist in m
                    power = 20e-3):                  # power in Watts
    """Find the ac Stark Shifts in Rb, Cs"""
    # typical optical tweezer parameters:
    bprop = [wavelength, power, beamwaist] # collect beam properties
    
    # mass, (L,J,F,MF), bprop, dipole matrix elements (Cm), resonant frequencies (rad/s),
    # linewidths (rad/s), state labels, nuclear spin, atomic symbol.
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
    wavels = np.linspace(800e-9, 950e-9, 500) 
    
    # ac Stark Shift in Joules:
    dE6S = Cs6S.acStarkShift(0,0,0,wavels, mj=0.5, HF=False)
    # average over mj states
    dE6P = 0.5*(Cs6P.acStarkShift(0,0,0,wavels, mj=1.5, HF=False) + 
        Cs6P.acStarkShift(0,0,0,wavels, mj=0.5, HF=False))
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
    # average over mj states
    dE5P = 0.5*(Rb5P.acStarkShift(0,0,0,wavels, mj=1.5, HF=False) + 
            Rb5P.acStarkShift(0,0,0,wavels, mj=0.5, HF=False))
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
    """Plot Fig. 5 - 8 in Arora et al 2007 to show that the polarisabilities 
    of Rb and Cs without hyperfine levels are correct"""
    # beam properties: wavelength, power, beam waist
    # intensity set to 1e10 MW/cm^2
    bprop = [1064e-9, np.pi*0.5e-2, 1e-6]
    
    for ATOM in [Rb, Cs]:
        if ATOM == Rb:
            wavel1 = np.linspace(780, 800, 200)*1e-9
            Ylim1 = (-8000, 8000)
            wavel2 = np.linspace(787,794, 200)*1e-9 
            Ylim2 = (-1000, 1000)
            FS, FP = 1, 3
        elif ATOM == Cs:
            wavel1 = np.linspace(925, 1000, 200)*1e-9
            Ylim1 = (-1000, 5000)
            wavel2 = np.linspace(927, 945, 200)*1e-9
            Ylim2 = (-100, 100)
            FS, FP = 3, 5
            
        S = dipole(ATOM.m, (0,1/2.,FS,FS), bprop,
                        ATOM.D0S, ATOM.w0S, ATOM.lwS, ATOM.nljS,
                        nuclear_spin = ATOM.I,
                        symbol=ATOM.X)
        
        P3 = dipole(ATOM.m, (1,3/2.,FP,FP), bprop,
                        ATOM.D0P3, ATOM.w0P3, ATOM.lwP3, ATOM.nljP3,
                        nuclear_spin = ATOM.I,
                        symbol=ATOM.X)
                        
        # compare polarisability of excited states
        plt.figure()
        plt.title("Polarisability of "+ATOM.X)
        plt.plot(wavel1*1e9, S.polarisability(wavel1)/au, 'r', label='s')
        plt.plot(wavel1*1e9, P3.polarisability(wavel1,mj=0.5)/au, 'g--', label='p$_{3/2}$, mj=1/2')
        plt.plot(wavel1*1e9, P3.polarisability(wavel1,mj=1.5)/au, 'm:', label='p$_{3/2}$, mj=3/2')
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Polarisability (a.u.)")
        plt.ylim(Ylim1)
        plt.xlim(wavel1[0]*1e9, wavel1[-1]*1e9)
        
        # calculate stark shifts between F, MF states
        mfLS = ['r', 'g--', 'm:', 'c-.', 'k-.', 'y']  # line styles
        plt.figure()
        plt.title("AC Stark Shifts for transitions from P$_{3/2}$ m$_F$ to \nthe groundstate in "+ATOM.X)
        dES = S.acStarkShift(0,0,0, wavel2, HF=True)   # ground state stark shift
        for MF in range(FP+1):
            P3.MF = MF
            dEPMF = P3.acStarkShift(0,0,0, wavel2, HF=True) # excited MF state stark shift
            plt.plot(wavel2*1e9, (dEPMF - dES)/h/1e6, mfLS[MF], label=r'm$_F$ = $\pm$'+str(MF))
        xlims = [wavel2[0]*1e9, wavel2[-1]*1e9]
        plt.plot(xlims, [0,0], 'k:', alpha=0.4)  #  show where zero is
        plt.ylim(Ylim2)
        plt.xlim(xlims)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Stark Shift (MHz)")
        plt.legend()
        
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
        a = np.zeros(4)  # make sure the results are always a size 3 array
        alpha = obj.polarisability(wavel=obj.field.lam, mj=MJ/2., HF=False, split=True)
        atot  = obj.polarisability(wavel=obj.field.lam, mj=MJ/2., HF=False, split=False)
        a[:np.size(alpha)] = alpha
        outstring += "polarisability components (scalar, vector, tensor) : \n(%.4g, %.4g, %.4g) a.u.\n"%(
            a[0]/au, a[1]/au, a[2]/au)
        outstring += "combined polarisability : %.4g a.u\n"%(atot/au)
            
        
    outstring += "\nIf hyperfine splitting is significant:\n"
    for F in range(int(abs(obj.I - obj.J)), int(obj.I + obj.J+1)):
        mfAveShift = 0
        for MF in range(-F, F+1):
            obj.F, obj.MF = F, MF
            mfAveShift += obj.acStarkShift(0,0,0, obj.field.lam, HF=True)/h/1e6
        outstring += "F = "+str(F)+ ", ave. mF  : %.5g MHz.\n"%(mfAveShift/(2.*F+1.))
        obj.F, obj.MF = F, F
        outstring += "|"+str(F)+","+str(F)+">  : %.5g MHz\n"%(
                obj.acStarkShift(0,0,0, obj.field.lam, HF=True)/h/1e6)

        a = np.zeros(4)  # make sure the results are always a size 3 array
        alpha = obj.polarisability(wavel=obj.field.lam, HF=True, split=True)
        atot  = obj.polarisability(wavel=obj.field.lam, HF=True, split=False)
        a[:np.size(alpha)] = alpha
        outstring += "polarisability components (scalar, vector, tensor) : \n(%.4g, %.4g, %.4g) a.u.\n"%(
            a[0]/au, a[1]/au, a[2]/au)
        outstring += "combined polarisability : %.4g a.u\n\n"%(atot/au)
                    
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
    
    
def combinedTrap(Cswl = 1064e-9, # wavelength of the Cs tweezer trap in m
                Rbwl = 880e-9, # wavelength of the Rb tweezer trap in m
                power = 6e-3, # power in W
                beamwaist = 1e-6): # beam waist in m
    """Model tweezer traps for Rb and Cs and find the potential each experiences
    when they're overlapping"""
    bprop = [Cswl, power, beamwaist] # collect beam properties
    
    # For the 1064nm trap:
    # mass, (L,J,F,MF), bprop, dipole matrix elements (Cm), resonant frequencies (rad/s),
    # linewidths (rad/s), state labels, nuclear spin, atomic symbol.
    # groundstate rubidium
    Rb1064 = dipole(Rb.m, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
                    
    # groundstate caesium
    Cs1064 = dipole(Cs.m, (0,1/2.,4,4), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
                    
    CsP = dipole(Cs.m, (1,3/2.,5,5), bprop,
                    Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
                    
    # set the power of the traps so that the trap depth experienced by each 
    # species in the overlapping trap is the same:
    P880 = (Cs1064.polarisability(Cswl,mj=0.5) - Rb1064.polarisability(Cswl, mj=0.5)) / (Rb1064.polarisability(Rbwl, mj=0.5) - Cs1064.polarisability(Rbwl, mj=0.5)) * power
    
    # for the 880nm trap:
    bprop = [Rbwl, abs(P880), beamwaist]
    Rb880 = dipole(Rb.m, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
                    
    Cs880 = dipole(Cs.m, (0,1/2.,3,3), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
                    
    
    # in the trap with both tweezers overlapping: 
    U0 = abs(Rb1064.acStarkShift(0,0,0) + Rb880.acStarkShift(0,0,0))
    wrRb = np.sqrt(4*U0 / Rb.m / beamwaist**2) /2. /np.pi /1e3
    wrCs = np.sqrt(4*U0 / Cs.m / beamwaist**2) /2. /np.pi /1e3
    print("%.0f beam power: %.3g mW\t\t%.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, P880*1e3))
    print("""In the combined %.0fnm and %.0fnm trap with a depth of %.3g mK the radial trapping frequencies are: 
Rubidium: %.0f kHz \nCaesium: %.0f kHz"""%(Rbwl*1e9, Cswl*1e9, U0/kB*1e3, wrRb, wrCs))
    
    # with just the Cs tweezer trap:
    URb =abs(Rb1064.acStarkShift(0,0,0))
    wrRb1064 = np.sqrt(4*URb / Rb.m / beamwaist**2) /2. /np.pi /1e3
    UCs = abs(Cs1064.acStarkShift(0,0,0))
    wrCs1064 = np.sqrt(4*UCs / Cs.m / beamwaist**2) /2. /np.pi /1e3
    print("""\nIn just the %.0fnm trap:
    Rubidium has trap depth %.3g mK
                 radial trapping frequency %.0f kHz
    Caesium has trap depth %.3g mK
                radial trapping frequency %.0f kHz"""%(Cswl*1e9, URb/kB*1e3, wrRb1064, UCs/kB*1e3, wrCs1064))
    
    print(getStarkShift(Cs1064))
    print(getStarkShift(CsP))
    
    # plot merging traps:
    n = 5   # number of time steps in merging to plot
    sep = np.linspace(0, 10e-6, n)     # initial separation of the tweezer traps
    zs = np.linspace(-2, 10, 200)*1e-6 # positions along the beam axis
    
    for atoms in [[Rb1064, Rb880], [Cs1064, Cs880]]:
        plt.figure()
        plt.subplots_adjust(hspace=0.01)
        for i in range(n):
            ax = plt.subplot2grid((n,1), (i,0))
            if i == 0:
                ax.set_title("Optical potential experienced by "+atoms[0].X
                +"\n%.0f beam power: %.3g mW   %.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, P880*1e3))
            
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
            
        plt.xlabel(r'Position ($\mu$m)')
        ax.set_xticks(sep*1e6)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Trap Depth (mK)')
        ax.yaxis.set_major_locator(AutoLocator())
        
    plt.show()
            
            
def getMFStarkShifts():
    """Return the Stark shifts of the MF states for Cs cooling/repump transitions"""
    
    print("stark shift of Cs 6S1/2 -> 6P3/2 for different MF states at 1064nm for beam power 6 mW, beam waist 1 micron, giving trap depth 1 mK")
    bprop = [1064e-9, 6e-3, 1e-6]      # wavelength, beam power, beam waist
    
    plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for F in [3, 4]:
        for MF in range(-F, F+1):
            print(" ----- |F = "+str(F)+", m_F = "+str(MF)+">")
            for MFp in range(MF-1, MF+2):
                S = dipole(Cs.m, (0,1/2.,F,MF), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
                P = dipole(Cs.m, (1,3/2.,F+1,MFp), bprop,
                    Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
                shift = (S.acStarkShift(0,0,0, bprop[0], HF=True) - P.acStarkShift(0,0,0, bprop[0], HF=True))/h/1e6
                if MF != 0:
                    deltaMF = (MFp-MF)*np.sign(MF)
                else:
                    deltaMF = (MFp-MF)
                plt.plot(MF, shift, '_', color=colors[F-3], alpha=0.33*(2+deltaMF), markersize=15, linewidth=10)
                print("|F' = "+str(F+1)+", m_F' = "+str(MFp)+"> : %.5g MHz"%shift)
                
    plt.xlabel("$M_F$")  
    plt.ylabel("AC Stark Shift (MHz)")
    lines = plt.gca().lines
    plt.legend(lines[18:24], ['F='+str(f)+r', $\Delta M_F=$'+str(-dmf) for f in range(3,5) for dmf in range(-1,2)])
    plt.show()

def compareKien():
    """compare Kien 2013 Fig 4,5"""
    bprop =[880e-9,20e-3,1e-6]
    Cs880 = dipole(Cs.m, (0,1/2.,3,3), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
                    
    CsP = dipole(Cs.m, (1,3/2.,3,3), bprop,
                    Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)               
    
    wls = [np.linspace(680, 690, 200)*1e-9, np.linspace(930, 940, 200)*1e-9]
    ylims = [(-1200, 300), (-3000, 6000)]
    for ii in range(2):
        plt.figure()
        plt.title("Cs Polarisabilities. Red: 6S$_{1/2}$, Blue: 6P$_{3/2}$.\nscalar: solid, vector: dashed, tensor: dotted")
        a1 = Cs880.polarisability(wls[ii],mj=0.5,split=True)
        a2 = 0.5*(np.array(CsP.polarisability(wls[ii],mj=1.5, split=True))+
            np.array(CsP.polarisability(wls[ii],mj=0.5, split=True)))
        ls = ['-', '--', ':']
        for i in range(3):
            plt.plot(wls[ii]*1e9, a1[i]/au, 'r', linestyle=ls[i], label="Cs")
            plt.plot(wls[ii]*1e9, a2[i]/au, 'b', linestyle=ls[i], label="$P_{3/2}$")
        #plt.legend()
        plt.ylim(ylims[ii])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Polarisablity (a.u.)")
        plt.show()
    
        
if __name__ == "__main__":
    wavelength = 880e-9 # wavelength in m
    power = 5e-3 # beam power in W
    beamwaist = 1e-6 # beam waist in m
    bprop = [wavelength, power, beamwaist]
    
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
    
    trap_depth = Rb5S.acStarkShift(0,0,0,wavelength)/kB * 1e3 # in mK at 880nm
    print("Rb 5S at %.3g nm: \t %.3g mK"%(wavelength*1e9, trap_depth))
    print("Cs 6S at 1064 nm: \t %.3g mK"%(Cs6S.acStarkShift(0,0,0,1064e-9)/kB*1e3))
    wavels = np.linspace(750,950,500)*1e-9 # wavelengths in m
    
    for STATES in [[Rb5S, Rb5P],[Cs6S, Cs6P]]:
        plt.figure()
        plt.title("AC Stark Shift in "+STATES[0].X+"\nbeam power %.3g mW, beam waist %.3g $\mu$m"%(power*1e3,beamwaist*1e6))
        plt.plot(wavels*1e9, STATES[0].acStarkShift(0,0,0,wavels)/kB*1e3, 'tab:blue', label='Ground S$_{1/2}$')
        excited_shift = 0.5*(STATES[1].acStarkShift(0,0,0,wavels,mj=0.5) + STATES[1].acStarkShift(0,0,0,wavels,mj=1.5))
        plt.plot(wavels*1e9, excited_shift/kB*1e3, 'r-.', label='Excited P$_{3/2}$')
        plt.legend()
        plt.ylabel("Trap Depth (mK)")
        plt.xlabel("Wavelength (nm)")
        plt.xlim(wavels[0]*1e9, wavels[-1]*1e9)
        plt.ylim(-5,5)
        plt.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.25) # show zero crossing
    plt.show()

    # runGUI()
    # combinedTrap(Rbwl=820e-9)
    # getMFStarkShifts()
    # plotStarkShifts(wlrange=[800,1100])
    