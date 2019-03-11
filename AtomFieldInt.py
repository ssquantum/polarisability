"""Stefan Spence 13.11.18

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
Use equation 10 from Kien 2013 for the hyperfine polarisability (quite slow)

25.02.19
Update transition data files to include natural linewidths.
Add functions to save transition data to file.

28.02.19
diagonalise the hfs + stark shift Hamiltonian as in Arora 2007 assuming that 
light is linearly polarised.

07.03.19
Finished diagonalising the combined Hamiltonian, working on matching up the
eigenvalues with the eigenstates
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
            
    return np.power(-1., j1 - j2 - m3) * facts * tsum

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
    nlj:quantum numbers (n, l, j) for the transitions
    rw: resonant wavelength (m) of transitions 
    w0: resonant frequency (rad/s) of transitions 
    lw: natural linewidth (rad/s) of transitions 
    Ah: magnetic dipole constant (Hz)
    Bh: electric quadrupole constant (Hz)
    """
    def __init__(self, mass, nuclear_spin, symbol, S1_DME, P1_DME, P3_DME,
                S1_RW, P1_RW, P3_RW, S1_LW, P1_LW, P3_LW,
                S1_nlj, P1_nlj, P3_nlj, S1_Ah, P1_Ah, P3_Ah, P3_Bh):
        self.D0S = S1_DME  # dipole matrix elements from S1/2 state
        self.D0P1 = P1_DME # dipole matrix elements from P1/2 state
        self.D0P3 = P3_DME # dipole matrix elements from P3/2 state
        self.nljS = S1_nlj # (n,l,j) quantum numbers for transitions
        self.nljP1 = P1_nlj
        self.nljP3 = P3_nlj
        self.rwS = S1_RW   # resonant wavelengths from S1/2 state (m)
        self.rwP1 = P1_RW  # resonant wavelengths from P1/2 state (m)
        self.rwP3 = P3_RW  # resonant wavelengths from P3/2 state (m)
        self.w0S = 2*np.pi*c / S1_RW # resonant frequency (rad/s)
        self.w0P1 = 2*np.pi*c / P1_RW
        self.w0P3 = 2*np.pi*c / P3_RW
        self.lwS = S1_LW   # natural linewidth from S1/2 (rad/s)
        self.lwP1 = P1_LW  # natural linewidth from P1/2 (rad/s)
        self.lwP3 = P3_LW  # natural linewidth from P3/2 (rad/s)
        self.AhS = S1_Ah   # magnetic dipole constant (Hz)
        self.AhP1 = P1_Ah
        self.AhP3 = P3_Ah
        self.BhP3 = P3_Bh  # electric quadrupole constant (Hz)
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


Cs = atom( S1_DME = S1_2[:,3], P1_DME = P1_2[:,3], P3_DME = P3_2[:,3], # matrix elements
    S1_RW = S1_2[:,4], P1_RW = P1_2[:,4], P3_RW = P3_2[:,4], # resonant wavelengths
    S1_LW = S1_2[:,5], P1_LW = P1_2[:,5], P3_LW = P3_2[:,5], # natural linewidths
    S1_nlj = S1_2[:,:3], P1_nlj = P1_2[:,:3], P3_nlj = P3_2[:,:3], # final state of transition
    S1_Ah = S1_2[0,6], P1_Ah = P1_2[0,6], P3_Ah = P3_2[0,6],   # magnetic dipole constant
    P3_Bh = P3_2[0,7],                                         # electric quadrupole constant
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

Rb = atom( S1_DME = S1_2[:,3], P1_DME = P1_2[:,3], P3_DME = P3_2[:,3], # matrix elements
    S1_RW = S1_2[:,4], P1_RW = P1_2[:,4], P3_RW = P3_2[:,4], # resonant wavelengths
    S1_LW = S1_2[:,5], P1_LW = P1_2[:,5], P3_LW = P3_2[:,5], # natural linewidths
    S1_nlj = S1_2[:,:3], P1_nlj = P1_2[:,:3], P3_nlj = P3_2[:,:3], # final state of transition
    S1_Ah = S1_2[0,6], P1_Ah = P1_2[0,6], P3_Ah = P3_2[0,6],   # magnetic dipole constant
    P3_Bh = P3_2[0,7],                                      # electric quadrupole constant
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
                    symbol="Cs", Ahfs=0, Bhfs=0):
        self.m                          = mass * amu           # mass of the atom in kg
        self.Ahfs                       = Ahfs                 # magnetic dipole constant
        self.Bhfs                       = Bhfs                 # electric quadrupole constant
        self.L, self.J                  = spin_state           # spin quantum numbers L, J
        self.I                          = nuclear_spin         # nuclear spin quantum number I
        self.field = Gauss(*field_properties)                  # combines all properties of the field
        self.X = symbol
        
        self.states = transition_labels                 # (n,l,j) quantum numbers for transitions
        self.omega0 = np.array(resonant_frequencies)    # resonant frequencies (rad/s)
        self.gam   = np.array(decay_rates)              # spontaneous decay rate (s)
        self.D0s   = np.array(dipole_matrix_elements)   # D0 = -e <a|r|b> for displacement r along the polarization direction
        self.omegas = np.array(2*np.pi*c/self.field.lam)# laser frequencies (rad/s)
        
            
    def acStarkShift(self, x, y, z, wavel=[], mj=None):
        """Return the potential from the dipole interaction 
        U = -<d>E = -1/2 Re[alpha] E^2
        Then taking the time average of the cos^2(wt) AC field term we get 
        U = -1/4 Re[alpha] E^2"""
        return -self.polarisability(wavel, mj, split=False) /4. *np.abs( 
                            self.field.amplitude(x,y,z) )**2
    
            
    def polarisability(self, wavel=[], mj=None, split=False):
        """wavel: wavelength (m) - default is self.field.lam
        mj: used when hyperfine splitting is negligible.
        split: Boolean - False gives total polarisability, True splits into
        scalar, vector, and tensor.
        Return the polarisability as given Arora 2007 (also see Cooper 2018,
        Mitroy 2010, Kein 2013) assuming that J and mj are good quantum 
        numbers and hyperfine splitting can be neglected. Assumes linear 
        polarisation so that the vector polarisability is zero."""
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
                aS += 1/3. /(2.*self.J + 1.) *self.D0s[i]**2 * (1/Ep + 1/Em)
                    
                aV += 0.5*(-1.)**(self.J + 2 + self.states[i][2]) * np.sqrt(6*self.J
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
            # NB: currently ignoring vector polarisability as in Arora 2007
            if split:
                return (aSvals, aVvals, aTvals)
            else:
                return aSvals + aTvals * (3*mj**2 - self.J*(self.J + 1)
                    ) / self.J / (2*self.J - 1)
        else:
            # there is no tensor polarisability for the J=1/2 state
            return aSvals
            
                            
    def diagH(self, wavel, x, y, z):
        """Diagonalise the combined Hamiltonian of hyperfine splitting + the ac
        Stark Shift. This gives the eigenenergies and eigenstates in the |F,mF>
        basis at a particular wavelength. Currently assuming linear polarisation
        along the z direction as in Arora 2007."""
        omega = 2*np.pi*c/wavel   # laser frequency in rad/s
        
        # |I-J| <= F <= I+J
        Fs = np.arange(abs(int(self.I - self.J)), int(self.I + self.J + 1))
        num_states = sum(2*Fs + 1)
        
        H = np.zeros((num_states, num_states)) # combined interaction Hamiltonian
        F_labels = np.concatenate([[F]*(2*F+1) for F in Fs])
        MF_labels = np.concatenate([list(range(-F,F+1)) for F in Fs])
        Hhfs = np.zeros(num_states)            # diagonal elements of the hfs Hamiltonian
        
        # state vector: (|F0 -F0>, |F0 -F0+1>, ..., |F0 F0>, |F1 -F1>, ..., |FN FN>)
        for F in Fs:
            for MF in range(-F, F+1):
                # hyperfine interaction is diagonal in F and mF:
                G = F*(F + 1) - self.I*(self.I + 1) - self.J*(self.J + 1)
                if self.J == 0.5:
                    Vhfs = h/2. * self.Ahfs * G # no quadrupole
                else:
                    Vhfs = h/2. * (self.Ahfs * G + self.Bhfs/4. * (3*G*(G + 1)
                    - 4*self.I*(self.I + 1) * self.J*(self.J + 1)) / self.I
                    /(2*self.I - 1.) /self.J /(2*self.J - 1.))
                
                # stark interaction is diagonal in mF
                # since the Hamiltonian is Hermitian, we only need to fill the lower triangle
                i = 2*F - min(Fs) + MF + np.sum(2*np.arange(min(Fs),F)) # horizontal index of Hamiltonian
                Fps = np.arange(min(Fs), F+1) # F'
                
                Hhfs[i] = Vhfs
                
                # not making the rotating wave approximation
                Ep = hbar*(self.omega0 + omega + 1j*self.gam)
                Em = hbar*(self.omega0 - omega - 1j*self.gam)
                
                aS = np.sum(self.D0s**2 /3. /(2.*self.J + 1.) * (1/Ep + 1/Em))
                
                aT = 0
                for ii in range(len(self.D0s)):  # wigner6j function takes scalars
                    aT += (-1)**(self.J + self.states[ii][2] + 1
                            ) * wigner6j(self.J, 1, self.states[ii][2], 1, 
                            self.J, 2) * self.D0s[ii]**2 * (1/Ep[ii] + 1/Em[ii])
                    
                for Fp in Fps:
                    if Fp >= abs(MF):
                        # due to symmetry, only some of the matrix elements need filling
                        j = Fp + MF + np.sum(2*np.arange(min(Fs),Fp) + 1)
                        
                        aT_F = aT * 4*np.sqrt(5/6. * (2*F + 1) * (2*Fp + 1)) * (-1)**(
                            self.J + self.I + F - Fp - MF) * wigner3j(F, 2, 
                            Fp, MF, 0, -MF) * wigner6j(F, 2, Fp, self.J, 
                            self.I, self.J)
                        
                        # print(F, Fp, MF, i, j)
                        # print(wigner3j(F, 2, Fp, MF, 0, -MF))
                        # print(wigner6j(F, 2, Fp, self.J, self.I, self.J))
                        # print()
                        if F == Fp: 
                            # The hyperfine splitting is diagonal in |F,MF>                   
                            H[i,j] = -0.5 * (aS.real + aT_F.real) * np.abs( 
                                        self.field.amplitude(x,y,z) )**2 + Vhfs
                        else: 
                            # state mixing is only from the anisotropic polarisability
                            H[i,j] = -0.5 * aT_F.real * np.abs( self.field.amplitude(x,y,z) )**2
                            
        # could fill the rest of H from symmetry: # H = H + H.T - np.diagflat(np.diag(H))
        # diagonalise the Hamiltonian to find the combined shift
        # since it's hermitian np can diagonalise just from the lower traingle
        eigenvalues, eigenvectors = np.linalg.eigh(H, UPLO='L')
        
        # to get the Stark shift, subtract off the hyperfine shift
        Hac = eigenvalues - Hhfs
        
        # note: the diagonalisation in numpy will likely re-order the eigenvectors
        # assume the eigenvector is that closest to the original eigenvector
        indexes = np.argmax(abs(eigenvectors), axis=1)
        return Hac[indexes], eigenvectors[:,indexes], Hhfs[indexes], F_labels, MF_labels