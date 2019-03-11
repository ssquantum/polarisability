"""Stefan Spence 07.03.19"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('..')
from matplotlib.ticker import AutoLocator
from AtomFieldInt import (dipole, Rb, Cs, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au)


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
    """Find the ac Stark Shifts in Rb, Cs assuming hyperfine splitting is negligible"""
    # typical optical tweezer parameters:
    bprop = [wavelength, power, beamwaist] # collect beam properties
    
    # mass, (L,J,F,MF), bprop, dipole matrix elements (Cm), resonant frequencies (rad/s),
    # linewidths (rad/s), state labels, nuclear spin, atomic symbol.
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
    
    # need a small spacing to resolve the magic wavelengths - so it will run slow
    # to resolve magic wavelengths, take about 10,000 points. Hac, eigenvectors, Hhfs, F_labels, MF_labels
    wavels = np.linspace(700e-9, 1100e-9, 2000) 
    
    # ac Stark Shift in Joules:
    dE6S = Cs6S.acStarkShift(0,0,0,wavels, mj=0.5)
    dE6P = Cs6P.acStarkShift(0,0,0,wavels, mj=1.5)
    dif6P = dE6P - dE6S
    
    magic6P = getMagicWavelengths(dif6P, dE6P, wavels)
    
    plt.figure()
    plt.title("AC Stark Shift in $^{133}$Cs")
    plt.plot(wavels*1e9, dE6S/h*1e-6, 'b--', label='Ground State S$_{1/2}$')
    plt.plot(wavels*1e9, dE6P/h*1e-6, 'r-.', label='Excited State P$_{3/2}$')
    # plt.plot(wavels*1e9, (dif6P)/h*1e-6, 'k', label='Difference')
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
    dE5S = Rb5S.acStarkShift(0,0,0,wavels, mj=0.5)
    dE5P = Rb5P.acStarkShift(0,0,0,wavels, mj=1.5)
    dif5P = dE5P - dE5S

    plt.figure()
    plt.title("AC Stark Shift in $^{87}$Rb")
    plt.plot(wavels*1e9, dE5S/h*1e-6, 'b--', label='Ground State S$_{1/2}$')
    plt.plot(wavels*1e9, dE5P/h*1e-6, 'r-.', label='Excited State P$_{3/2}$')
    # plt.plot(wavels*1e9, (dif5P)/h*1e-6, 'k', label='Difference')
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
    numpoints = 100
    for ATOM in [Rb, Cs]:
        if ATOM == Rb:
            wavel1 = np.linspace(780, 800, numpoints)*1e-9
            Ylim1 = (-8000, 8000)
            wavel2 = np.linspace(787,794, numpoints)*1e-9 
            Ylim2 = (-1000, 1000)
            FP=3
        elif ATOM == Cs:
            wavel1 = np.linspace(925, 1000, numpoints)*1e-9
            Ylim1 = (-1000, 5000)
            wavel2 = np.linspace(927, 945, numpoints)*1e-9
            Ylim2 = (-100, 100)
            FP=5
            
        S = dipole(ATOM.m, (0,1/2.), bprop,
                        ATOM.D0S, ATOM.w0S, ATOM.lwS, ATOM.nljS,
                        nuclear_spin = ATOM.I,
                        symbol=ATOM.X)
        
        P3 = dipole(ATOM.m, (1,3/2.), bprop,
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
        plt.title("AC Stark Shifts for transitions from P$_{3/2}$ |F'=3, m$_F'\\rangle$ to \nthe groundstate in "+ATOM.X)
        ES = np.zeros(len(wavel2))
        EP3 = np.zeros((FP+1, len(wavel2)))
        for i in range(len(wavel2)):
            ES[i] = S.diagH(wavel2[i], 0,0,0)[0][0]  # ground state shift is independent of MF
            EP3vals, _, _, Fs, MFs = P3.diagH(wavel2[i], 0,0,0) # get eigenvalues of excited state stark shift operator
            EP3[:,i] = EP3vals[-FP-1:]  # only interested in F' = F
            
        for MF in range(FP+1): # the shift only depends on the magnitude of MF
            plt.plot(wavel2*1e9, (EP3[MF] - ES)/h/1e6, mfLS[MF], label='m$_F$ = $\pm$'+str(MF))
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
    for MJ in np.arange(1, 2*obj.J+1, 2).astype(int): # NB: this MJ is 2*MJ
        U = obj.acStarkShift(0,0,0, obj.field.lam, mj=MJ/2.) # stark shift in J
        outstring += "MJ = "+str(MJ)+"/2 : %.5g MHz  =  %.3g mK\n"%( U/h/1e6, U/kB*1e3)
            
        
    outstring += "\nIf hyperfine splitting is significant:\n"
    starkEns, eigVecs, hfsEns, Fs, MFs = obj.diagH(obj.field.lam, 0,0,0)
    F = min(Fs)
    for i in range(len(Fs)):
        indF = np.where(Fs == F)[0][0]
        print(i, indF)
        if i == indF:
            mfAveShift = np.mean(starkEns[indF:indF+2*F+1])/h/1e6
            outstring += "F = "+str(Fs[i])+ ", ave. mF  : %.5g MHz.\t"%(mfAveShift/(2.*F+1.))
        F = Fs[i]
        outstring += "|"+str(Fs[i])+","+str(MFs[i])+">  : %.5g MHz\n"%(starkEns[i]/h/1e6)
                    
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
        elif atomSymbol == "Cs":
            atomObj = Cs
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
        dipoleObj = dipole(atomObj.m, (L,J), bprop,
                D0, w0, lw, nlj,
                nuclear_spin = atomObj.I,
                symbol=atomObj.X)
        
        messagebox.showinfo("Calculation Result", getStarkShift(dipoleObj))
        
    resultButton = tkinter.Button(root, text="Calculate Stark Shifts", 
                                    command=showResult)
    resultButton.pack(side = tkinter.BOTTOM)
    
    root.mainloop()

def latexATOMDATA():
    """send the atom data to a text file for formatting in a table in latex"""
    with open("AtomData.txt", "w+") as f:
        ls = ["S", "P", "D", "F"]
        for ATOM in [Rb, Cs]:
            for DATA in [[ATOM.nljS, ATOM.rwS, ATOM.D0S, ATOM.lwS, ' S$_{1/2}$ '], [ATOM.nljP1, ATOM.rwP1, ATOM.D0P1, ATOM.lwP1, ' P$_{1/2}$ '],[ATOM.nljP3, ATOM.rwP3, ATOM.D0P3, ATOM.lwP3, ' P$_{3/2}$ ']]:
                f.write("\\multicolumn{4}{|c|}{" + ATOM.X + DATA[4] +"} \\\\ \\hline\n")
                for i in range(len(DATA[1])):
                    f.write("%s"%int(DATA[0][i][0])+ls[int(DATA[0][i][1])]+"$_{%s/2}$"%int(DATA[0][i][2]*2)+" & %.4g & %.3g & %.3g \\\\\n\\hline\n"%(DATA[1][i]*1e9, DATA[2][i]/e/a0, DATA[3][i]/2./np.pi/1e6))
    

def writeTransitionData():
    """updating the atomic transitions data file"""
    for ATOM in [Rb, Cs]:
        for DATA in [[ATOM.nljS, ATOM.rwS, ATOM.D0S, ATOM.lwS, 'S1_2'], [ATOM.nljP1, ATOM.rwP1, ATOM.D0P1, ATOM.lwP1, 'P1_2'],[ATOM.nljP3, ATOM.rwP3, ATOM.D0P3, ATOM.lwP3, 'P3_2']]:
            with open(ATOM.X+DATA[4]+'.dat', 'w+') as f:
                f.write('# n, l, j, Dipole Matrix Element (Cm), Resonant Wavlength (m), Linewidth (rad/s)\n')
                for i in range(len(DATA[1])):
                    if DATA[0][i][0] < 23:
                        f.write("%s,%s,%s,%.16e,%.16e,%.16e\n"%(int(DATA[0][i][0]),int(DATA[0][i][1]),float(DATA[0][i][2]),abs(DATA[2][i]), DATA[1][i], (2*np.pi/abs(DATA[1][i]))**3 * DATA[2][i]**2/3./np.pi/hbar/eps0/(2.*DATA[0][i][2]+1)))
                    else:
                        f.write("%s,%s,%s,%.16e,%.16e,%.16e\n"%(int(DATA[0][i][0]),int(DATA[0][i][1]),float(DATA[0][i][2]),abs(DATA[2][i]), DATA[1][i], 0))

if __name__ == "__main__":
    Rb5P = dipole(Rb.m, (1,3/2.), [1064e-9, 20e-3, 1e-6],
                    Rb.D0P3, Rb.w0P3, Rb.lwP3, Rb.nljP3,
                    nuclear_spin = Rb.I,
                    symbol=Rb.X,
                    Ahfs = Rb.AhP3,
                    Bhfs = Rb.BhP3)
    print(getStarkShift(Rb5P))
    compareArora()
    
    