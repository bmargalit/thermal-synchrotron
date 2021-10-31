# `Thermal + Non-thermal' Synchrotron Emission Model
This code accompanies the paper "Thermal Electrons in Mildly-relativistic Synchrotron Blast-waves" (Margalit &amp; Quataert 2021; MQ21). It implements the `thermal + non-thermal' model presented in MQ21, allowing calculation of the emergent synchrotron emission from an electron population comprised of both a thermal (Maxwellian) component and a non-thermal (power-law) distribution, including the effects of synchrotron self-absorption and synchrotron cooling.

# Requirements
- numpy
- scipy
- matplotlib (only required for test.py script)

# File descriptions
- Module `thermalsyn.py` implements functions from MQ21 and can be used to calculate the resulting synchrotron emission
- Script `test.py` illustrates example use cases of `thermalsyn.py`

# Basic Usage
The primary goal of this code is to compute the specific luminosity from a shock. This can be calculated using the function `thermalsyn.Lnu_shock()` and specifying the shock radius, velocity, and upstream density (as well as microphysical parameters), for example:
```python
import numpy as np
import thermalsyn

c = 2.99792458e10

nu = np.logspace(8,13,300)
n = 1e3
r = 1e16
v = 0.3*c

Lnu = thermalsyn.Lnu_shock(nu,n,r,v,epsilon_T=1.0,epsilon_B=0.1,delta=0.01,p=3.0)
```
For more general settings applicable also outside the scope of shock-powered transients, the function `thermalsyn.Lnu_fun()` can be called. This calculates emission as a direct function of the emitting plasma properties: electron density, region size, electron temperature, magnetic field, and source age.
