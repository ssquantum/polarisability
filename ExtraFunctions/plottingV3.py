"""Stefan Spence 11.03.19

Collecting together the plotting and data handling functions from 
AtomFieldInt_V3.py 04.02.19
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13}) # increase font size (default 10)
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('..')
sys.path.append(r'Y:\Tweezer\People\Vincent\python snippets\plotting_tools')
from AtomFieldInt_V3 import dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, kB, amu, Eh, au
from default_colours import DUsea_blue, DUcherry_red
from matplotlib.ticker import AutoLocator
   
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
            plt.plot(wavel2*1e9, (dEPMF - dES)/h/1e6, mfLS[MF], label='m$_F$ = $\pm$'+str(MF))
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
            
        
    outstring += "\nIf hyperfine splitting is significant:\n"
    for F in range(int(abs(obj.I - obj.J)), int(obj.I + obj.J+1)):
        mfAveShift = 0
        for MF in range(-F, F+1):
            obj.F, obj.MF = F, MF
            mfAveShift += obj.acStarkShift(0,0,0, obj.field.lam, HF=True)/h/1e6
        outstring += "F = "+str(F)+ ", ave. mF  : %.5g MHz.\t"%(mfAveShift/(2.*F+1.))
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
    
    
def combinedTrap(Cswl = 1064e-9, # wavelength of the Cs tweezer trap in m
                Rbwl = 880e-9, # wavelength of the Rb tweezer trap in m
                power = 20e-3, # power in W
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
            plt.plot(zs*1e6, U1064, color=DUcherry_red, alpha=0.6)
            plt.plot(zs*1e6, U880, color=DUsea_blue, alpha=0.6)
            plt.plot([0]*2, [min(U),0], color=DUcherry_red, linewidth=10, label='%.0f'%(Cswl*1e9), alpha=0.4)
            plt.plot([sep[n-i-1]*1e6]*2, [min(U),0], color=DUsea_blue, linewidth=10, label='%.0f'%(Rbwl*1e9), alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.xlabel('Position ($\mu$m)')
        ax.set_xticks(sep*1e6)
        plt.ylabel('Trap Depth (mK)')
        ax.yaxis.set_major_locator(AutoLocator())
        
    plt.show()

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
    plt.ylabel("AC Stark Shift (MHz)")
    lines = plt.gca().lines
    plt.legend(lines[l1[0]:l1[1]], ['F='+str(f)+r', $\Delta M_F=$'+str(-dmf) 
                for f in range(min(Fs),max(Fs)+1) for dmf in range(-1,2)])
    plt.show()

    
def plotPolarisability():
    """Plot the polarisability of Rb 5S and Cs 6S states highlighting our laser wavelengths"""
    bprop = [1064e-9, 6e-3, 1e-6]      # wavelength, beam power, beam waist
    wavelengths = np.linspace(700, 1100, 500)*1e-9 # in m
    ymax = 5000
    
    # groundstate rubidium
    Rb5S = dipole(Rb.m, (0,1/2.,1,1), bprop,
                    Rb.D0S, Rb.w0S, Rb.lwS, Rb.nljS,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
    alphaRb = Rb5S.polarisability(wavelengths)/au # polarisability in atomic units
                    
    # groundstate caesium
    Cs6S = dipole(Cs.m, (0,1/2.,4,4), bprop,
                    Cs.D0S, Cs.w0S, Cs.lwS, Cs.nljS,
                    nuclear_spin = Cs.I,
                    symbol=Cs.X)
    alphaCs = Cs6S.polarisability(wavelengths)/au # polarisability in atomic units
            
    plt.figure()
    # split up the plotting so as not to have lines at resonances:
    for v in [[alphaRb, DUsea_blue, 'Rb 5S$_{1/2}$'], [alphaCs, DUcherry_red, 'Cs 6S$_{1/2}$']]:
        plus = np.where(v[0] > 0)[0] # where polarisability is positive
        ind1 = plus[np.where(plus > np.arange(len(plus))+plus[0])[0][0]] # second positive region
        plt.plot(wavelengths[:plus[0]-1]*1e9, v[0][:plus[0]-1], color=v[1], label=v[2])
        plt.plot(wavelengths[plus[0]+1:ind1-1]*1e9, v[0][plus[0]+1:ind1-1], color=v[1])
        plt.plot(wavelengths[ind1+1:]*1e9, v[0][ind1+1:], color=v[1])

    # plot dotted lines to show where the resonances are
    plt.plot([780]*2, [-ymax, ymax], '--', color=DUsea_blue)
    plt.plot([795]*2, [-ymax, ymax], '--', color=DUsea_blue)
    plt.plot([852.3]*2, [-ymax, ymax], '--', color=DUcherry_red)
    plt.plot([894.6]*2, [-ymax, ymax], '--', color=DUcherry_red)

    # show zero crossing
    plt.plot([wavelengths[0]*1e9, wavelengths[-1]*1e9], [0,0], 'k--', alpha=0.4)
    # show laser wavelengths
    # plt.fill_between([1060,1070], ymax, -ymax, color=DUcherry_red, alpha=0.3)
    # plt.fill_between([935,945], ymax, -ymax, color=DUcherry_red, alpha=0.3)
    # plt.fill_between([878, 882], ymax, -ymax, color=DUsea_blue, alpha=0.3)
    # plt.fill_between([805,825], ymax, -ymax, color=DUsea_blue, alpha=0.3)
    plt.ylim((-ymax, ymax))
    plt.ylabel('Polarisability ($a_0^3$)')
    plt.xlim((wavelengths[0]*1e9, wavelengths[-1]*1e9))
    plt.xlabel('Wavelength (nm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    wavelength = 1064e-9 # wavelength in m
    power = 6e-3 # beam power in W
    beamwaist = 1e-6 # beam waist in m
    bprop = [wavelength, power, beamwaist]
    
    Rb5P1 = dipole(Rb.m, (1,1/2.,1,1), bprop,
                    Rb.D0P1, Rb.w0P1, Rb.lwP1, Rb.nljP1,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)
    Rb5P3 = dipole(Rb.m, (1,3/2.,1,1), bprop,
                    Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X)               
    
    # print(getStarkShift(Rb5P1))
    print(getStarkShift(Rb5P3))
    print(np.array(Rb5P3.polarisability(795e-9, HF=True, split=True))/au)
    
    # combinedTrap(power=6e-3)
    # getMFStarkShifts()
    plotPolarisability()
                    
    # compare Kien 2013 Fig 4,5:
    # wls = [np.linspace(680, 690, 200)*1e-9, np.linspace(930, 940, 200)*1e-9]
    # ylims = [(-1200, 300), (-3000, 6000)]
    # for ii in range(2):
    #     plt.figure()
    #     plt.title("Cs Polarisabilities. Red: 6S$_{1/2}$, Blue: 6P$_{3/2}$.\nscalar: solid, vector: dashed, tensor: dotted")
    #     a1 = Cs880.polarisability(wls[ii],mj=0.5,split=True)
    #     a2 = CSP.polarisability(wls[ii],mj=1.5, split=True)
    #     ls = ['-', '--', ':']
    #     for i in range(3):
    #         plt.plot(wls[ii]*1e9, a1[i]/au, 'r', linestyle=ls[i], label="Cs")
    #         plt.plot(wls[ii]*1e9, a2[i]/au, 'b', linestyle=ls[i], label="$P_{3/2}$")
    #     #plt.legend()
    #     plt.ylim(ylims[ii])
    #     plt.xlabel("Wavelength (nm)")
    #     plt.ylabel("Polarisablity (a.u.)")
    #     plt.show()