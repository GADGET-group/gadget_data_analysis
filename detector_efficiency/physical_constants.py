##Useful physical constants and unit conversions.
#Note: This file is all in mks.
#@author Alexander Adams
from math import pi


h = 6.62606957e-34 #planks constant
hbar = h/2/pi
kb = 1.380649e-23#boltzman constant
Runiv = 8.3144621e3#J/kmolK universal gass constant
Navo = 6.022e23#avagadros number
G = 6.67384e-11#gravitational constant
sigma = 5.670374419e-8#stephen boltzmann constant
epsilon0 = 8.8541878128e-12#C^2/Nm^2 permativity of free space/electric constnat
mu0 = 1.25663706212e-6#permeability of free space/magnetic constant
c = 299792458. #speed of light
gearth = 9.80665#m/s^2

##########################
#unit convertion factors##
##########################
celsiusOffset = 273.15
poise = 0.1
rankine = 5./9.
#to use, multiply something in non mks by its conversion to factor to get
#in mks. For example 9*eV will give the value of 9eV in J
eV = 1.602176565e-19 #electron volt
cal = 4.184 #calorie (not kilocalorie)
btu = 1055.05585#british thermal unit
qe = eV #fundamental charge
keV = eV*1000
MeV = eV*1e6
GeV = eV*1e9
#pressure units
atm = 101325. #atmosphere
psi = 6894.76#pounds per square inch
mmHg = 133.322387415 #mm mercury
inHg = 3386.#inches mercury
bar = 1e5#bar
torr = 133.322
#force units
lbf = 4.448#pound force
#mass units
lbm = 0.4536#pound mass
amu = 1.660468e-27#atomic mass unit
me = 9.1093821545e-31#mass of electron
#distance units
nmi = 1852.#nautical miles
miles = 1609.344 #statute mile
ft = 0.3048 #english feet
inches = ft/12
g = 0.001
cm = 0.01
mm = 0.001
#volume units
liters = 1/1000.
gallons = 0.00378541
#units of time
minutes = 60.
hours = 60*minutes
sidereal_days =  86164.0905
jyear = 86400*365.25#julian year
