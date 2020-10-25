#!/usr/bin/python3
# encoding: utf-8

from gradissimo import *
from matplotlib import pyplot

# Test of consistency 
# We create a Gradissimo fiber with segments of defined length and we calculate
# the output beam. Then, we do the reverse process. We consider the geometry of
# the output beam and we calculate the lengths that should give this profile. 
# Thankfully, we find the original values.

# Define the working wavelength and create the materials...
set_wavelength(1.55e-6)
n_fiber = 1.44
n_out = 1.00    
input_fiber = SingleModeFiber(w=5.2e-6, n=n_fiber)
HS = HomogeneousSpace(n_fiber)
GI = GradientIndexFiber(n_fiber, gamma=5.7e3, diam=62.5e-6)
OUT = HomogeneousSpace(n_out)

# Create the Gradissimo fiber and define the geometry (segment length)...
G1 = Gradissimo(input_fiber, HS, GI, OUT)
G1.set_geometry(L_HS=1e-3, L_GI=0.7e-3)
G1.plot()

# Extract the parameters of the output beam...
L_OUT = G1.beam_OUT.waist_position  # same as G1.L_OUT
w_OUT = G1.beam_OUT.waist_profile.w

# Reverse process...
# Calculate the segment lengths that produce this output beam...
G2 = Gradissimo(input_fiber, HS, GI, OUT)
G2.adjust_geometry(w_OUT, L_OUT, oscillations=1)
G2.plot()
# An oscillation in the GI fiber was added in order to recover the right length

# Or in another way...
G3 = Gradissimo(input_fiber, HS, GI, OUT)
G3.adjust_geometry(Q2=G1.Q2, oscillations=1)
G3.plot()

print("The following values should be identical for all the lines...")
print("G1 values: L_HS = {:.6f}, L_GI = {:.6f}".format(G1.L_HS, G1.L_GI))
print("G2 values: L_HS = {:.6f}, L_GI = {:.6f}".format(G2.L_HS, G2.L_GI))
print("G3 values: L_HS = {:.6f}, L_GI = {:.6f}".format(G3.L_HS, G3.L_GI))

# The two graphs that were plotted should be identical
pyplot.show()
