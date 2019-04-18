"""Stefan Spence 18.04.19
Get the Stark Shift of Rb and Cs ground and excited states
as a function of intensity showing state mixing by the overlap
with original eigenstates."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('..')
from AtomFieldInt import (dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au)

def make_segments(x, y):
    '''
    see https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,label=''):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, label=label)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

#### Rb ####
# beam properties:
wavelength = 810e-9  # laser wavelength in m
power      = 5e-3    # laser power in W
beamwaist  = 1e-6    # tweezer trap beam waist in m
bprop = [wavelength, power, beamwaist]

# create dipole objects for ground and excited states
Rb5S = dipole(Rb.m, (0,1/2.), bprop,
                Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                nuclear_spin = Rb.I,
                symbol=Rb.X,
                Ahfs = Rb.AhS)

Rb5P = dipole(Rb.m, (1,3/2.), bprop,
                Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                nuclear_spin = Rb.I,
                symbol=Rb.X,
                Ahfs = Rb.AhP3,
                Bhfs = Rb.BhP3)

# powers up to 1 mK trap depth
maxPower = abs(1e-3*kB * np.pi * eps0 * c * beamwaist**2 / Rb5S.polarisability(wavelength)) # in W
powers = np.linspace(0, maxPower, 100) # in W

# get Stark Shift
EvalRb5S = np.zeros((len(powers), 8)) # stark shift of Rb ground state
EvecRb5S = np.zeros((len(powers), 8, 8)) # eigenvectors of Rb ground state
EvalRb5P = np.zeros((len(powers), 16)) # stark shift of Rb excited state
EvecRb5P = np.zeros((len(powers), 16, 16)) # eigenvectors of Rb excited state
for i, p in enumerate(powers):
    Rb5S.field.E0 = 2 * np.sqrt(p / eps0 / c / np.pi)/beamwaist
    EvalRb5S[i], EvecRb5S[i], _, F5S, MF5S = Rb5S.diagH(wavelength, 0,0,0)
    Rb5P.field.E0 = 2 * np.sqrt(p / eps0 / c / np.pi)/beamwaist
    EvalRb5P[i], EvecRb5P[i], _, F5P, MF5P = Rb5P.diagH(wavelength, 0,0,0)


cmps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper'] # list of colourmaps
plt.figure()
vmin = min(EvecRb5S[:,0,0])
vmax = max(EvecRb5S[:,0,0])
cl = colorline(powers*1e3, EvalRb5S[:,0]/h/1e6, z=EvecRb5S[:,0,0], label='5S$_{1/2}$')
for i, line in enumerate(EvalRb5P.T):
    cl = colorline(powers*1e3, line/h/1e6, z=EvecRb5P[:,i,i], cmap=cmps[i], label='5P$_{3/2}$ |F=%s, M$_F$=%s>'%(F5P[i], MF5P[i]))
plt.xlabel('Power (mW)')
plt.ylabel('Stark Shift (MHz)')
plt.ylim(-22, 20)
plt.xlim(min(powers*1e3), max(powers*1e3))
plt.legend()
plt.show()

#### Cs ####
# beam properties:
wavelength = 1064e-9  # laser wavelength in m
power      = 5e-3    # laser power in W
beamwaist  = 1e-6    # tweezer trap beam waist in m
bprop = [wavelength, power, beamwaist]

# create dipole objects for ground and excited states
Cs6S = dipole(Cs.m, (0,1/2.), bprop,
                Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                nuclear_spin = Cs.I,
                symbol=Cs.X,
                Ahfs = Cs.AhS)
                
Cs6P = dipole(Cs.m, (1,3/2.), bprop,
                Cs.D0P3, Cs.w0P3, Cs.lwP3, Cs.nljP3,
                nuclear_spin = Cs.I,
                symbol=Cs.X,
                Ahfs = Cs.AhP3,
                Bhfs = Cs.BhP3)