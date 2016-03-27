#!/usr/bin/python3
# encoding: utf-8

from gradissimo import *
set_wavelength(1.31e-6)

##############################################################################
# Demonstration that the waist diameter does not depend on the material,
# but the position depends.

HS_1 = HomogeneousSpace(1.00)
HS_2 = HomogeneousSpace(1.60)

# Take a test profile
Q0 = GaussianProfile(w=30e-6, C=-1/200e-6).Q

beam1 = HS_1.beam(Q0)
beam2 = HS_2.beam(Q0)

w0_1 = beam1.waist.w
z0_1 = beam1.z0
w0_2 = beam2.waist.w
z0_2 = beam2.z0

print("Beam in two different materials but with the same input profile...")
print("Waist for beam 1: {:.4e} m at distance {:.4e} m".format(w0_1, z0_1))
print("Waist for beam 2: {:.4e} m at distance {:.4e} m".format(w0_2, z0_2))

