# polarisability
Functions to calculate the polarisability and hence stark shift of Rb and Cs atoms (fully generalisable to other atoms)

If you are not familiar with object oriented programming in Python, there is a simple UI that can calculate Stark shifts for you.
There are different ways you can start it:
 - runUI.bat
This windows batch file uses the local installation of Enthough Canopy Python to run AtomFieldInt_V3.py
 - python AtomFieldInt_V3.py rungui
If python is in the windows path then the above line can be typed into cmd
It runs AtomFieldInt_V3.py with the argument 'rungui' to start the UI.
 - runGUI()
If you are using an interactive Python environment then you can execute the runGUI() function.


AtomFieldInt_V3.py calculates the polarisabilities in two regimes:
 - hyperfine splitting can be ignored (J is a good quantum number, sum over <J|d|J'> matrix elements)
 - the stark shift is small compared to hyperfine splitting (so the combined Hamiltonian is diagonal in the hyperfine basis,
   F is a good quantum number, and the Wigner-Eckart theorem can be used to write <F|d|F'> in terms of <J|d|J'>)
   
Note then that the results may be inaccurate when the ac Stark shift is on the order of the hyperfine splitting.

AtomFieldInt.py attempts to use a numerical matrix diagonalisation for when there is mixing of states, however it is currently not working.
   
Several classes are implemented so that functions using the polarisability can be flexibly created:
 - atom - contains the properties of atoms used to calculate the polarisability (all in SI units):
    dipole matrix elements, quantum numbers of the corresponding states, resonant wavelengths, resonant frequencies, natural linewidths,
    mass, nuclear spin (I), and chemical symbol.
        two objects are instantiated : Rb and Cs
        the data is loaded from csv files in the TransitionData folder
        
 - Gauss - contains properties relevant to a Gaussian beam:
    power, amplitude, beam waist, polarisation (all in SI units)
    
 - dipole - contains formulas for calculating the polarisability and the ac Stark shift:
    requires data from the atom class, uses the Gauss class to calculate field amplitude.
    To calculate the polarisability, instatiate an object of this class and then use its polarisability function.

Directory structure:
 - ExtraFunctions
 scripts loading the classes and using them to make various plots
 - Graphs
collection of graphs produced using the code 
 - Literature
values for the polarisability of different states obtained from various papers
 - TransitionData
csv files containing the atomic states, resonant wavelengths, dipole matrix elements, natural linewidths, and hyperfine constants
 - TweezerSelectiveLoading
graphs produced to help choose whether to have a Rb tweezer at 880nm or 800-830nm.

Several example functions used to plot graphs (mostly comparing to literature) are included. 
While these functions take up a fair amount of space, hopefully they demonstrate the usage of the classes.
