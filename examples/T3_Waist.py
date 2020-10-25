#!/usr/bin/python3
# encoding: utf-8

from gradissimo import *
from matplotlib import pyplot

set_wavelength(1.31e-6)

##############################################################################
# Demonstration that the diameter at waist does not depend on the material,
# but the position of the waist depends on the material.

# Consider a test profile at the entrance of the material. Remember that 
# waist diameter and reduced curvature do not change at a material interface.
P = GaussianProfile(w=30e-6, C=-1/200e-6)

# Create the two materials to compare...
HS_1 = HomogeneousSpace(1.00)
HS_2 = HomogeneousSpace(1.60)

# Build the beams...
beam1 = P.beam(HS_1)
beam2 = P.beam(HS_2)

# Print the result...
print("Beam in two different materials but with the same input profile...")

w0_1 = beam1.waist_profile.w
z0_1 = beam1.waist_position
print("Waist for beam 1: {:.4e} m at distance {:.2e} m".format(w0_1, z0_1))

w0_2 = beam2.waist_profile.w
z0_2 = beam2.waist_position
print("Waist for beam 2: {:.4e} m at distance {:.2e} m".format(w0_2, z0_2))

beam1.plot(z2=2*z0_1)
beam2.plot(z2=2*z0_2)
pyplot.show()

