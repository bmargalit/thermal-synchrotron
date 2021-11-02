'''Example usage of thermalsyn.py module

This script illustrates two simple usage cases of the thermalsyn.py module
that calculates synchrotron emission from a combined `thermal + non-thermal`
electron distribution, as presented by Margalit & Quataert (2021). In the
examples below, the specific luminosity is calculated using the main function
thermalsyn.Lnu_shock(). In the first case an SED is plotted, whereas in the
second example a few light-curves (at different frequencies) are calculated
for a shock that expands as R ~ t^m within a wind density profile (n ~ R^{-2}).

See thermalsyn.py for more detailed documentation.
Please cite Margalit & Quataert (2021) if used for scientific purposes:
https://ui.adsabs.harvard.edu/abs/2021arXiv211100012M/abstract

'''

import numpy as np
import matplotlib.pylab as plt
plt.ion()
import thermalsyn

# define physical constants (in cgs units):
day = 24.0*3600.0
c = 2.99792458e10

# set value of microphysical parameters
p = 3.0
delta = 1e-2
epsilon_B = 1e-1
epsilon_T = 1e0

################################
###     Plot Spectrum
################################

# set value of physical parameters
v = 0.4*c
n = 1e3
R = v*25*day

# calculate luminosity as a function of frequency
nu = np.logspace(8,13,3000)
Lnu = thermalsyn.Lnu_shock(nu,n,R,v,epsilon_T=epsilon_T,epsilon_B=epsilon_B,delta=delta,p=p)

# plot results
plt.figure()
plt.loglog(nu,Lnu,'-',color='k',linewidth=5)
plt.xlim([1e8,1e13])
plt.xlabel(r'$\nu \,\,\,\,\, ({\rm Hz})$',fontsize=14)
plt.ylabel(r'$L_\nu \,\,\,\,\, ({\rm erg \, s}^{-1}\,{\rm Hz}^{-1})$',fontsize=14)

################################
###     Plot Light-curve
################################

# frequencies at which to calculate the light-curve
nu_list = np.array([6,10,30,90])*1e9
clr_list = ['grey','mediumslateblue','tomato','goldenrod']

t = np.logspace(0,3,3000)*day
# deceleration parameter
m = 0.8
# initial conditions
v0 = 0.3*c
R0 = 1e16
t0 = m*R0/v0
# time evolution
R = R0*(t/t0)**m
v = v0*(t/t0)**(m-1.0)

# wind density-profile
n0 = 3e4
n = n0*(R/R0)**(-2)

# calculate and plot results
plt.figure()

for i in range(np.size(nu_list)):
    Lnu = thermalsyn.Lnu_shock(nu_list[i],n,R,v,epsilon_T=epsilon_T,epsilon_B=epsilon_B,delta=delta,p=p)
    plt.loglog(t/day,Lnu,'-',linewidth=5,color=clr_list[i])

plt.xlim([3e0,3e2])
plt.ylim([2e27,2e30])
plt.xlabel(r'$t \,\,\,\,\, ({\rm day})$',fontsize=12)
plt.ylabel(r'$L_\nu \,\,\,\,\, ({\rm erg \, s}^{-1}\,{\rm Hz}^{-1})$',fontsize=14)

plt.show()
