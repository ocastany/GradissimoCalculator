#!/usr/bin/python3
# encoding: utf-8

from gradissimo import *

# Test of consistency 
# We create a Gradissimo fiber with given segment length and compute the output
# beam. We then take the geometry of the output beam and ask back the lengths
# that should given this profile. Of course they should be the initial ones.

# Define the materials
set_wavelength(1.55e-6)
n_fiber = 1.44
n_out = 1.00    
input_fiber = SingleModeFiber(w=5.2e-6, n=n_fiber)
HS = HomogeneousSpace(n_fiber)
GI = GradientIndexFiber(n_fiber, gamma=5.7e3, diam=62.5e-6)
OUT = HomogeneousSpace(n_out)

# Gradissimo with given segment length...
G1 = Gradissimo(input_fiber, HS, GI, OUT)
G1.set_geometry(L_HS=1e-3, L_GI=0.7e-3)
G1.plot()

# Take the output beam parameters...
w_OUT = G1.beam_OUT.waist.w
L_OUT = G1.L_OUT

# Compute the segment length for producing this output beam
# Note : one oscillation in the GI fiber is added
G2 = Gradissimo(input_fiber, HS, GI, OUT)
G2.adjust_geometry(w_OUT, L_OUT, oscillations=1)
G2.plot()

# The two graphs should be identical
pyplot.show()
