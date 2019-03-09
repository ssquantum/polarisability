"""Stefan Spence 13.11.18
Grad Course: Atomic and Molecular Interactions
Calculate the spectrum of one and two Rydberg atoms in an optical tweezer.
1) Formulate the equations for Gaussian beam propagation.
2) Look at the dipole interaction and the induced dipole moment as a function of
laser wavelength and spatial position
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
Note: the arc data doesn't have linewidths for transitions: they're almost all 0

24.01.19
Some dipole matrix elements loaded from arc are inaccurate, so load them from a 
file instead
Had to correct the formula for the polarisability by a factor of 1/2 in order to
match the dipole matrix elements used in Arora 2007

11.02.19
Add in data from Arora 2007 and 2012 for comparison
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from math import factorial 
# from arc import Caesium, Rubidium87

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
                    )/ factorial(a + b + c + 1)
                
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
            tsum += (-1)**t /factorial(t) /factorial(j3-j2+t+m1) /factorial(j3-
                j1+t-m2) /factorial(j1+j2-j3-t) /factorial(j1-t-m1) /factorial(
                j2-t-m2)
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
            tsum += (-1)**t * factorial(t+1) / factorial(t-j1-j2-j3) /factorial(
            t-j1-J2-J3) / factorial(t-J1-j2-J3) / factorial(t-J1-J2-j3) / (
            factorial(j1+j2+J1+J2-t)) / factorial(j2+j3+J2+J3-t) / factorial(
            j1+j3+J1+J3-t)
        except ValueError:
            # sum is only over positive factorials
            tsum += 0
    
    return tripls * tsum
    
# EDIT 24.01.19 -- some matrix elements from arc are inaccurate
# so load from a file instead
# def getArcValues(n1, l1, j1):
#     """Return the reduced dipole matrix elements, transition wavelengths, and 
#     natural linewidths of all significant transitions from the given state with
#     principal quantum number n1, orbital angular momentum l1, and total angular
#     momentum j1"""
#     # set the atom
#     if n1 == 6:
#         arcAtom = Caesium()
#     elif n1 == 5:
#         arcAtom = Rubidium87()
#         
#     # use dipole selection rules to find accessible final states
#     if l1 == 0 and j1 == 0.5:
#         empty = np.zeros(2*(42-n1))
#         l2j2 = [[1,0.5], [1,1.5]]  # -> P1/2, P3/2
#         
#     elif l1 == 1 and j1 == 0.5:
#         empty = np.zeros( 42-n1 + 42-n1+1 )
#         l2j2 = [[0,0.5], [2,1.5]]  # -> S1/2, D3/2
#         
#     elif l1 == 1 and j1 == 1.5:
#         empty = np.zeros( 42-n1 + 2*(42-n1+1) )
#         l2j2 = [[0,0.5], [2,1.5], [2,2.5]] # -> S1/2, D3/2, D5/2
#         
#             
#     # initiate output arrays
#     matrix_elements, wavelengths, linewidths = empty.copy(), empty.copy(), empty.copy()
#     final_states = []
#     i = 0   # counter for index of arrays
#     
#     
#     # loop over final states
#     lower_bound = n1
#     for l2, j2 in l2j2:
#         if l2 == 2:
#             lower_bound = n1 - 1
#             
#         for n2 in range(lower_bound, 42):
#             matrix_elements[i] = arcAtom.getReducedMatrixElementJ_asymmetric(
#                                 n1, l1, j1, n2, l2, j2) * a0 * e
#             wavelengths[i] = arcAtom.getTransitionWavelength(n1, l1, j1, 
#                                 n2, l2, j2)
#             linewidths[i] = arcAtom.getTransitionRate(n1, l1, j1, n2, l2,
#                                 j2, temperature=0.1)
#             i += 1
#             final_states.append([n2, l2, j2])
#             
#             
#     return matrix_elements, wavelengths, linewidths, final_states

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
        
"""
24.01.19
Some dipole matrix elements loaded from arc are inaccurate, so load them from a 
file instead

######### atomic properties for Cs-133:  ##########
# for the 6S1/2 state:
S1_2DME, S1_2RW, S1_2LW, S1_states = getArcValues(6, 0, 0.5)
        
# for the 6P1/2 state:
P1_2DME, P1_2RW, P1_2LW, P1_states = getArcValues(6, 1, 0.5)

# for the 6P3/2 state:
P3_2DME, P3_2RW, P3_2LW, P3_states = getArcValues(6, 1, 1.5)
# need to insert some linewidth data:
# P3_2LW[74] = 3.6e6 * 2 * np.pi  # cite Tai et al PRA 12 3 (1975)
# P3_2DME[74] = 4.8 * e * a0 # to match the data in Kien 2013...

Cs = atom( S1_2DME = S1_2DME, P1_2DME = P1_2DME, P3_2DME = P3_2DME, # matrix elements
    S1_2RW = S1_2RW, P1_2RW = P1_2RW, P3_2RW = P3_2RW, # resonant wavelengths
    S1_2LW = S1_2LW, P1_2LW = P1_2LW, P3_2LW = P3_2LW, # natural linewidths
    S1_nlj = S1_states, P1_nlj = P1_states, P3_nlj = P3_states, # final state of transition
    mass = 133*amu,        # mass in kg
    nuclear_spin = 7/2.,   # intrinsic angular momentum quantum number of the nucleus
    symbol = 'Cs')


######### atomic properties for Rb-87:  ###########
# for the 6S1/2 state:
S1_2DME, S1_2RW, S1_2LW, S1_states = getArcValues(5, 0, 0.5)
        
# for the 6P1/2 state:
P1_2DME, P1_2RW, P1_2LW, P1_states = getArcValues(5, 1, 0.5)
# correct some dipole matrix elements
# P1_2DME[38] = 1.8110770276274834e-29

# for the 6P3/2 state:
P3_2DME, P3_2RW, P3_2LW, P3_states = getArcValues(5, 1, 1.5)

# to better match Griffin 2006:
# P3_2DME[0] = 3.870680906736364e-29
# P3_2DME[1] = 4.1e-29
# P3_2DME[37] = 1.500667981267195e-29
# P3_2DME[75] = 4.50793878892524e-29

Rb = atom( S1_2DME = S1_2DME, P1_2DME = P1_2DME, P3_2DME = P3_2DME, # matrix elements
    S1_2RW = S1_2RW, P1_2RW = P1_2RW, P3_2RW = P3_2RW, # resonant wavelengths
    S1_2LW = S1_2LW, P1_2LW = P1_2LW, P3_2LW = P3_2LW, # natural linewidths
    S1_nlj = S1_states, P1_nlj = P1_states, P3_nlj = P3_states, # final state of transition
    mass = 87*amu,        # mass in kg
    nuclear_spin = 3/2.,   # intrinsic angular momentum quantum number of the nucleus
    symbol = 'Rb')
"""

# 24.01.19
# Some dipole matrix elements loaded from arc are inaccurate, so load them from a 
# file instead
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
        return self.zR /1.j /q * self.E0 * np.exp(1j * self.k * z) * np.exp(1j * self.k * rhosq / 2. / q)
        

#######################
        
        
class dipole:
    """Properties and equations of the dipole interaction between atom and field"""
    def __init__(self, mass, spin_state, field_properties,
                    dipole_matrix_elements, resonant_frequencies, decay_rates,
                    transition_labels, nuclear_spin=7/2., filedir=None, 
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
        
        
        # initiate scalar, vector, and tensor components of the polarisability:
        # wavelength in nm, polarisability in atomic units
        self.alphaWavelengths, self.alphaS, self.alphaV, self.alphaT = 0, 0, 0, 0
        if symbol == "Cs":
            self.loadCsPolarisability(filedir)
        elif symbol == "Rb":
            self.loadRbPolarisability(filedir)
            
        
    def loadCsPolarisability(self, fdir):
        """Load values for the scalar, vector and tensor components of the
        polarisability, as given by Kien et al 2013"""
        orbital = ['S', 'P', 'D', 'F', 'G', 'H', 'I']
        selfL = orbital[self.L]         # letter for the orbital angular momentum
        selfmltpl = int(2*self.J + 1)   # multiplicity 2J + 1
        
        if fdir != None:
            for filename in os.listdir(fdir):
                state = filename.split("_")[2].split(".")[0]    # state is n L J
                mltpl = int(2*float(state.split("-")[0][-1])/float(state[-1]) + 1) # multiplicity 2J + 1
                                
                if state[1] == selfL and mltpl == selfmltpl:
                    if filename.split("_")[1][-1] == "0":
                        data = np.loadtxt(os.path.join(fdir,filename),
                                    delimiter=',', comments=r'\x0c')
                        self.alphaWavelengths = data[:,0]
                        # scalar polarisability
                        self.alphaS = data[:,1] * au  # convert values from atomic units         
                        
                    elif filename.split("_")[1][-1] == "1":
                        # vector polarisability, convert from atomic units
                        self.alphaV = np.loadtxt(os.path.join(fdir,filename),
                            delimiter=',', comments=r'\x0c')[:,1] * au  
                                    
                    elif filename.split("_")[1][-1] == "2":
                        # tensor polarisability
                        self.alphaT = np.loadtxt(os.path.join(fdir,filename),
                            delimiter=',', comments=r'\x0c')[:,1] * au
                                    
    def loadRbPolarisability(self, filename):
        """Load values for the scalar, vector and tensor components of the
        polarisability, as given by Robert Potvliege. Note that they assumed
        linearly polarized light so there is no vector polarisability."""
        
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        self.alphaWavelengths = data[:,0]
        if self.L == 0:
            self.alphaS = data[:,1] * au  # convert values from atomic units
            # no tensor polarisability for s state
            
        elif self.L == 1:
            self.alphaS = data[:,2] * au  # convert values from atomic units
            self.alphaT = data[:,3] * au
                
    
        
    def findValue(self, wavel=[]):
        """Find the value of the polarisability at given wavelength(s) wavel"""
        numwavel = 1                # number of wavelengths to evaluate at
        if np.size(wavel) == 0:
            wavel = self.field.lam  # default wavelength to evaluate at
            
        if np.size(wavel) > 1:
            numwavel = np.size(wavel)
            wavel = min(wavel)      # the arrays are always in increasing order
            
        ldif = abs(self.alphaWavelengths - wavel)
        index = np.where(ldif == min(ldif))[0][0]
        alphas = np.array([self.alphaS, self.alphaV, self.alphaT])
        alpha = []
        for a in alphas:
            try:
                alpha.append(a[index:index + numwavel])
            except TypeError:
                alpha.append(a)
                
        return alpha
        
                
    def acStarkShift(self, x, y, z, wavel=[], mj=None, HF=False):
        """Find the AC Stark Shift as given in Arora 2007 from loaded 
        polarisability data assuming that J and 
        M_J are good quantum numbers (the Stark shift is small compared to
        the splitting and hyperfine splitting can be ignored) at a wavelength of 
        wavel in a Gaussian E field at position (x,y,z). This formula also 
        ignores the polarisation of the incident light. When hyperfine 
        transitions are important, the quantum numbers F and M_F determine the 
        contribution of the tensor polarisability, otherwise J and M_J 
        determine the contribution of the tensor polarisability."""
            
        # get polarisability values at the laser wavelength:
        aS, aV, aT = self.findValue(wavel)
        
        if self.J > 0.5:
            if HF:
                return -(aS + np.sqrt((self.J + 1)*(2*self.J + 1)*(2*self.J + 3)
                    /self.J /(2*self.J - 1)) * (-1)**(self.I + self.J - self.MF
                    ) * (2*self.F + 1) * wigner6j(self.F, 2, self.F, self.J, 
                    self.I, self.J) * wigner3j(self.F, 2, self.F, self.MF, 0, 
                    -self.MF) * aT) /4. * np.abs(self.field.amplitude(x,y,z))**2
            else: 
                return - (aS + aT * (3*mj**2 - self.J*(self.J + 1))/self.J /(2*
                    self.J - 1)) /4. * np.abs(self.field.amplitude(x,y,z))**2
        else:
            # there is no tensor polarisability for the J=1/2 state
            return - aS /4. * np.abs(self.field.amplitude(x,y,z))**2
            
        
    def U(self, x, y, z):
        """Return the potential from the dipole interaction 
        U = -<d>E = -1/2 Re[alpha] E^2
        Then taking the time average of the cos^2(wt) AC field term we get 
        U = -1/4 Re[alpha] E^2"""
        return -self.polarisability() /2. *np.abs( self.field.amplitude(x,y,z) )**2
    
            
    def polarisability(self, wavel=[], mj=None, HF=False):
        """Return the polarisability as given Arora 2007 (also see Cooper 2018,
        Mitroy 2010, Kein 2013) assuming that J and mj are good quantum 
        numbers when hyperfine splitting can be neglected, or that F and mf are
        good quantum numbers. Assumes linear polarisation so that the vector
        polarisability is zero."""
        if np.size(wavel) != 0:            
            omegas = np.array(2*np.pi*c/wavel)# laser frequencies (rad/s)
        else:
            omegas = self.omegas
     
        empty = np.zeros(np.size(omegas))
        aSvals, aVvals, aTvals = empty.copy(), empty.copy(), empty.copy()
        
        for ii in range(len(omegas)):
            
            aS, aV, aT = 0, 0, 0
            
            # loop over final states
            for i in range(len(self.states)):
                    Ep = hbar*(self.omega0[i] + omegas[ii] + 1j*self.gam[i])
                    Em = hbar*(self.omega0[i] - omegas[ii] - 1j*self.gam[i])
                
                    aS += 1/3. /(2*self.J + 1) *self. D0s[i]**2 * (1/Em + 1/Ep)
                    
                    aT += 2*np.sqrt(5 * self.J * (2*self.J - 1) / 6. /(self.J + 1) /
                        (2*self.J + 1) / (2*self.J + 3)) * (-1)**(self.J + 
                        self.states[i][2]) * wigner6j(self.J, 1, self.states[i][2], 
                        1, self.J, 2) * self.D0s[i]**2 * (1/Em + 1/Ep)
                        
                    i += 1
      
            aSvals[ii] = aS.real
            aTvals[ii] = aT.real
        
        
        return (aSvals, aTvals)
        
        # combine polarisabilities
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
     
def plotStarkShifts(E0=1e5):
    """Use loaded polarisabilities to find the ac Stark Shifts in Rb, Cs at a
    peak field strength of E0."""
    # typical optical tweezer parameters:
    wavelength = 880e-9                 # laser wavelength in nm
    omega = 2*np.pi * c / wavelength    # angular frequency in rad/s
    beamwaist = 1e-6                    # beam waist in m
    #power = 1e10*np.pi*1e-6**2          # power in watts for intensity 1MW/cm^2
    # set the electric field strength as 1e5 V/m
    power = (E0*beamwaist/2.)**2 * eps0 * c * np.pi # power in Watts 
    bprop = [wavelength, power, beamwaist] # collect beam properties
    
    # the Polarisability file might be for Rb-85, the paper is unclear. See:
    #  P F Griffin et al (2006) New J. Phys. 8 11
    Rb5S = dipole(Rb.m, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS,
                    nuclear_spin = Rb.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Griffin2006_RbPolarisability',
                    symbol=Rb.X)
    
    Rb5P = dipole(Rb.m, (1,1/2.,1,1), bprop,
                    Rb.D0P1, Rb.w0P1, Rb.lwP1,
                    nuclear_spin = Rb.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Griffin2006_RbPolarisability',
                    symbol=Rb.X)
    
    Cs6S = dipole(Cs.m, (0,1/2.,3,3), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS,
                    nuclear_spin = Cs.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Kien2013_Cs_alpha',
                    symbol=Cs.X)
    Cs6P = dipole(Cs.m, (1,3/2.,3,3), bprop,
                    Cs.D0P3, Cs.w0P3, Cs.lwP3,
                    nuclear_spin = Cs.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Kien2013_Cs_alpha',
                    symbol=Cs.X)
    
                    
    wavels = Cs6P.alphaWavelengths
    # ac Stark Shift in Joules:
    dE6S = Cs6S.acStarkShift(0,0,0,wavels,HF=True)
    dE6P = Cs6P.acStarkShift(0,0,0,wavels,HF=True)
    dif6P = dE6P - dE6S
    
    magic6P = getMagicWavelengths(dif6P, dE6P, wavels)
    
    plt.figure()
    plt.title("AC Stark Shift in $^{133}$Cs")
    plt.plot(wavels, dE6S/h*1e-6, 'b--', label='Ground S$_{1/2}$')
    plt.plot(wavels, dE6P/h*1e-6, 'r-.', label='Excited P$_{3/2}$')
    plt.plot(wavels, (dif6P)/h*1e-6, 'k', label='Difference')
    plt.plot([magic6P[0]]*2, [min(dif6P/h/1e6),max(dif6P/h/1e6)], 'm:',
                label = 'Magic Wavelength')
    _ = magic6P.pop(0)
    plt.legend()
    for mw in magic6P:
        plt.plot([mw]*2, [min(dif6P/h/1e6),max(dif6P/h/1e6)], 'm:')
    plt.ylabel("Stark Shift (MHz)")
    plt.xlabel("Wavelength (nm)")
    plt.xlim((800,1100))
    plt.plot(wavels, np.zeros(len(wavels)), 'k', alpha=0.5)
    plt.show()
    print("Magic wavelengths at:\n", magic6P)
    
    Rbwave = Rb5S.alphaWavelengths
    # ac Stark Shift in Joules:
    dE5S = Rb5S.acStarkShift(0,0,0,Rbwave,HF=True)
    dE5P = Rb5P.acStarkShift(0,0,0,Rbwave,HF=True)
    dif5P = dE5P - dE5S

    plt.figure()
    plt.title("AC Stark Shift in $^{85}$Rb")
    plt.plot(Rbwave, dE5S/h*1e-6, 'b--', label='Ground S$_{1/2}$')
    plt.plot(Rbwave, dE5P/h*1e-6, 'r-.', label='Excited P$_{1/2}$')
    plt.plot(Rbwave, (dif5P)/h*1e-6, 'k', label='Difference')
    plt.legend()
    plt.ylabel("Stark Shift (MHz)")
    plt.xlabel("Wavelength (nm)")
    plt.xlim((800,1100))
    plt.ylim((-dif5P[0]/h*1e-6,dif5P[0]/h*1e-6))
    plt.show()

def printStarkShift(obj):
    """Print the ac Stark Shift for all of the hyperfine levels in a particular
    fine structure state of the atom in dipoleObject"""
    Lterms = ['S', 'P', 'D', 'F', 'G']
    
    print(obj.X + " " + Lterms[obj.L] + str(int(obj.J*2)) +
            "/2 ac Stark Shift at %.0f nm for E field %.2g V/m:"
            %(obj.field.lam*1e9, obj.field.E0))
            
    for F in range(int(abs(obj.I - obj.J)), int(obj.I + obj.J+1)):
        if F > 0:
            obj.F, obj.MF = F, F
            print("|"+str(F)+","+str(F)+">  = %.5g MHz"%(obj.acStarkShift(0,0,
                                            0,obj.field.lam*1e9,HF=True)/h/1e6))
                                                    
def printRbCsStarkShifts(lam=880e-9):
    """Print the ac Stark shifts for Rb and Cs states at wavelength lam"""
    # typical optical tweezer parameters:
    wavelength = lam                    # laser wavelength in m
    beamwaist = 1e-6                    # beam waist in m
    power = 20e-3                       # power in watts
    # set the electric field intensity as 1e10 (V/m)^2
    #power = (1e5*beamwaist/2.)**2 * eps0 * c * np.pi # power in Watts 
    bprop = [wavelength, power, beamwaist] # collect beam properties
    
    # the Polarisability file might be for Rb-85, the paper is unclear. See:
    #  P F Griffin et al (2006) New J. Phys. 8 11
    Rb5 = dipole(Rb.m, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS,
                    nuclear_spin = Rb.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Griffin2006_RbPolarisability',
                    symbol=Rb.X)
                    
    # find the Stark shifts at 880nm for the Rb |F, M_F> states:
    printStarkShift(Rb5)                    # 5S 1/2 state

    Rb5.L, Rb5.J = 1, 3/2.                  # 5P 3/2 state
    Rb5.loadRbPolarisability(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Griffin2006_RbPolarisability')
    printStarkShift(Rb5)
    
    
    Cs6 = dipole(Cs.m, (0,1/2.,3,3), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS,
                    nuclear_spin = Cs.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Kien2013_Cs_alpha',
                    symbol=Cs.X)
                    
    # find the Stark shifts at 880nm for the Cs |F, M_F> states:
    printStarkShift(Cs6)                    # 6S 1/2 state
    E6S1_2 = Cs6.acStarkShift(0,0,0,mj=1/2.)       # stark shift of the 6S 1/2 state
    
    Cs6.L, Cs6.J = 1, 3/2.                  # 6P 3/2 state
    Cs6.loadCsPolarisability(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Kien2013_Cs_alpha')
    printStarkShift(Cs6)                    
    
    # Stark shift between Cs 6S 1/2 and 6P 3/2 in units of kHz / (kV/cm)^2:
    shift1 = (E6S1_2 - Cs6.acStarkShift(0,0,0,mj=1/2.))/h*1e10 / np.abs(Cs6.field.amplitude(0,0,0))**2
    

if __name__ == "__main__":
    # typical optical tweezer parameters:
    wavelength = np.linspace(800e-9, 900e-9, 200)                 # laser wavelength in m
    beamwaist = 1e-6                    # beam waist in m
    power = 20e-3                       # power in watts
    # set the electric field intensity as 1e10 (V/m)^2
    #power = (1e5*beamwaist/2.)**2 * eps0 * c * np.pi # power in Watts 
    bprop = [wavelength, power, beamwaist] # collect beam properties
    
    from scipy.interpolate import interp1d#
    
    def plotwithres(ax, plttitle, ws, data, label1, model, label2):
        """Plot a comparison of our model for the polarisability (from Arora 
        20007) against literature data"""
        
        ax[0].set_title(plttitle)
        ax[0].plot(ws, data, 'o', label=label1, markersize=3)
        ax[0].plot(ws, model, '--', label=label2)
        ax[0].set_ylabel("Polarisability ($a_{0}^{3}$)")
        ax[0].legend()
        
        ax[1].plot(ws, (data-model)/data, color=ax[0].lines[-2].get_color(),
                            markersize=3)
        ax[1].set_ylabel("Weighted \nResiduals\n")
        
        plt.xlabel("Wavelength (nm)")
        plt.xlim((wavelength[0]*1e9,wavelength[-1]*1e9))
        
    
    # load data from Safronova 2006 for ground state polarisability:
    saf_Rb = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Safronova2006_alpha\Safronova2006_Rb5s_alpha.csv', delimiter=",", skiprows=1)
    saf_Cs = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Safronova2006_alpha\Safronova2006_Cs6s_alpha.csv', delimiter=",", skiprows=1)
    
    # load data from Arora 2012 for Rb polarisability:
    ar12_5S = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Arora2012_alpha\Rb5S.csv', delimiter=",", skiprows=1)
    ar12_5P1 = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Arora2012_alpha\Rb5P1_2.csv', delimiter=",", skiprows=1)
    ar12_5P3_1 = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Arora2012_alpha\Rb5P3_2mj1_2.csv', delimiter=",", skiprows=1)
    ar12_5P3_3 = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Arora2012_alpha\Rb5P3_2mj3_2.csv', delimiter=",", skiprows=1)
    
    # load data from Arora 2007 for Cs polarisability:
    ar12_6P1_2 = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Arora2007_alpha\Arora2007_Cs6P1_2.csv', delimiter=",", skiprows=1)
    ar12_6P3_1 = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Arora2007_alpha\Arora2007_Cs6P3_2_mj1_2.csv', delimiter=",", skiprows=1)
    ar12_6P3_3 = np.loadtxt(r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Arora2007_alpha\Arora2007_Cs6P3_2_mj3_2.csv', delimiter=",", skiprows=1)
    
    
    # find the polarisability for the Rb states:
    # P F Griffin et al (2006) New J. Phys. 8 11
    Rb5S = dipole(Rb.m, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = Rb.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Griffin2006_RbPolarisability',
                    symbol=Rb.X)
      
    # 5S 1/2 state
    as1, av1, at1 = Rb5S.findValue(Rb5S.alphaWavelengths) # loaded polarisabilities from Griffin 2006
    as2, at2 = Rb5S.polarisability()                     # calculated polarisabilities
    grif = interp1d(Rb5S.alphaWavelengths, as1/au)       
    
    fig1, ax1 = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1],'hspace':0}, sharex='all')
    plotwithres(ax1, "Scalar Polarisability for the $^{87}$Rb 5S$_{1/2}$ state", 
        wavelength*1e9, grif(wavelength*1e9), "Griffin 2006", as2/au, "Model")
    ax1[0].plot(saf_Rb[:,0], saf_Rb[:,1], 'rx', markersize=10, label="Safronova 2006")
    ax1[0].plot(ar12_5S[:,0], ar12_5S[:,1], 'gx', markersize=10, label="Arora 2012")
    ax1[0].legend()
    
    ax1[1].plot(saf_Rb[:,0], (saf_Rb[:,1]-Rb5S.polarisability(saf_Rb[:,0]*1e-9)[0]/au)/saf_Rb[:,1], 'rx',
                            markersize=5)
    ax1[1].plot(ar12_5S[:,0], (ar12_5S[:,1]-Rb5S.polarisability(ar12_5S[:,0]*1e-9)[0]/au)/ar12_5S[:,1], 'gx',
                            markersize=5)
    plt.show()
    
    
    # 5P 3/2 state
    Rb5P = dipole(Rb.m, (1,3/2.,1,1), bprop,
                    Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                    nuclear_spin = Rb.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Griffin2006_RbPolarisability',
                    symbol=Rb.X)
    
    as1, av1, at1 = Rb5P.findValue(Rb5P.alphaWavelengths) # loaded polarisabilities from Griffin 2006
    as2, at2 = Rb5P.polarisability()                     # calculated polarisabilities
    grifS = interp1d(Rb5P.alphaWavelengths, as1/au)
    grifT = interp1d(Rb5P.alphaWavelengths, at1/au)
    
    fig2, ax2 = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1],'hspace':0}, sharex='all')
    plotwithres(ax2, "Polarisability for the $^{87}$Rb 5P$_{3/2}$ state", 
        wavelength*1e9, grifS(wavelength*1e9), "Griffin 2006 Scalar", as2/au, "Scalar")
    plotwithres(ax2, "Polarisability for the $^{87}$Rb 5P$_{3/2}$ state", 
        wavelength*1e9, grifT(wavelength*1e9), "Griffin 2006 Tensor", at2/au, "Tensor")
    ax2[1].lines[-1].set_color(ax2[0].lines[-2].get_color())
    plt.show()
    
    
    Cs6S = dipole(Cs.m, (0,1/2.,3,3), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                    nuclear_spin = Cs.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Kien2013_Cs_alpha',
                    symbol=Cs.X)
    
    as1, av1, at1 = Cs6S.findValue(Cs6S.alphaWavelengths) # loaded polarisabilities from Kien 2013
    as2, at2 = Cs6S.polarisability()                     # calculated polarisabilities
    kien = interp1d(Cs6S.alphaWavelengths, as1/au)
    
    fig3, ax3 = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1],'hspace':0}, sharex='all')
    plotwithres(ax3, "Scalar Polarisability for the $^{133}$Cs 6S$_{1/2}$ state", 
        wavelength*1e9, kien(wavelength*1e9), "Kien2013", as2/au, "Model")
    ax3[0].plot(saf_Cs[:,0], saf_Cs[:,1], 'rx', markersize=10, label="Safronova 2006")
    ax3[0].legend()
    ax3[0].set_ylim((-40000, 40000))
    ax3[1].set_ylim((-0.3, 0.3))
    ax3[1].plot(saf_Cs[:,0], (saf_Cs[:,1]-Cs6S.polarisability(saf_Cs[:,0]*1e-9)[0]/au)/saf_Cs[:,1], 'rx',
                            markersize=5)
    plt.show()
    
    
    Cs6P = dipole(Cs.m, (1,3/2.,3,3), bprop,
                    Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                    nuclear_spin = Cs.I,
                    filedir=r'C:\Users\qgtx64\DocumentsCDrive\QSUM\Polarisability\Kien2013_Cs_alpha',
                    symbol=Cs.X)
    
    as1, av1, at1 = Cs6P.findValue(Cs6P.alphaWavelengths) # loaded polarisabilities from Kien 2013
    as2, at2 = Cs6P.polarisability()                     # calculated polarisabilities
    kienS = interp1d(Cs6P.alphaWavelengths, as1/au)
    kienT = interp1d(Cs6P.alphaWavelengths, at1/au)

    fig4, ax4 = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1],'hspace':0}, sharex='all')
    plotwithres(ax4, "Polarisability for the $^{133}$Cs 6P$_{3/2}$ state", 
        wavelength*1e9, kienS(wavelength*1e9), "Kien 2013 Scalar", as2/au, "Scalar")
    plotwithres(ax4, "Polarisability for the $^{133}$Cs 6P$_{3/2}$ state", 
        wavelength*1e9, kienT(wavelength*1e9), "Kien 2013 Tensor", at2/au, "Tensor")
    ax4[0].set_ylim((-40000, 40000))
    ax4[1].set_ylim((-0.5, 0.5))
    plt.show()
