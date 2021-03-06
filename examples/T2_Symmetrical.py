#!/usr/bin/python3
# encoding: utf-8

from gradissimo import *
from numpy import pi, inf, sqrt, cos, sin, sinh, cosh, tanh, arccos, arctan
from matplotlib import pyplot

##############################################################################
# Consider a symmetrical situation, L_silica / L_GI / L_silica 
# and the same waist w0 at input and output.

# Mission: determine the length L_GI as a function of the other parameters.

# Let us take some numerical values for an example...
n_silica = 1.44
L_silica = 120e-6

w0 = 5.2e-6
lbda0 = 1.3e-6

############################# [Part 1] #################################
# Find length L_GI analytically...

# Useful quantities...
lbda = lbda0 / n_silica
zR = pi * w0**2 / lbda
q0 = 1.0j * zR

# Propagation of the beam to the GI fiber entrance...
q1 = q0 + L_silica

# Profile evolution in the GI fiber is q(z) = 1/γ tan(γz + θ).
# Origin for z is the entrance of the GI fiber.
# The value of γ (gamma) is taken such that P/4 = 365 µm 
# [value from US patent 6014483]
gamma = 2*pi/(4*365e-6)     
theta = arctan(gamma * q1)    # Complex value

# Because the situation is symmetrical, the waist is maximal at the middle of 
# the GI segment, and this corresponds to Re(γz + θ) = (k + ½) π
L_GI = 2 * (pi/2 - theta.real) / gamma
print("GI segment length...")
print("- direct     calculation : {:.3e} m".format(L_GI), end="")
w_max = sqrt(lbda/(gamma * pi * tanh(theta.imag)))
w_min = sqrt(lbda * tanh(theta.imag) / (gamma * pi))
print(" / w_max={:.2e}, w_min={:.2e}".format(w_max, w_min))
# We get L_GI = 482.1 µm, w_max = 14.7 µm, w_min = 4.55 µm

############################# [Part 2] #################################
# Check with Gradissimo code...

set_wavelength(lbda0)
input_fiber = SingleModeFiber(w0, n_silica)
HS = HomogeneousSpace(n_silica)
GI = GradientIndexFiber(n_silica, gamma)
OUT = HomogeneousSpace(n_silica)

G = Gradissimo(input_fiber, HS, GI, OUT)
G.adjust_geometry(w0, L_silica)
print("- gradissimo calculation : {:.3e} m".format(G.L_GI), end="")
print(" / w_max={:.2e}, w_min={:.2e}".format(G.beam_GI.w_max, G.beam_GI.w_min))
G.plot()
pyplot.show()
