#!/usr/bin/python3
# encoding: utf-8

from gradissimo import *

# Study of Gradissimo fibers.
# Parameters of the GI fiber are taken from table 4 in this article:
# M. Thual, Opt. Eng., 46(1), p. 015402 (2007)

# Define the wavelength and the materials
set_wavelength(1.55e-6)
n_fiber = 1.44
n_out = 1.00    
input_fiber = SingleModeFiber(w=5.2e-6, n=n_fiber)
HS = HomogeneousSpace(n_fiber)
GI = GradientIndexFiber(n_fiber, gamma=4117, diam=85e-6)
OUT = HomogeneousSpace(n_out)

# Gradissimo fiber structure...
G = Gradissimo(input_fiber, HS, GI, OUT)

# Segment lengths to be used
L_HS_list = [0, 200, 400]
L_GI_list = linspace(0, 800, num=100)

data_list = []
for (i, L_HS) in enumerate(L_HS_list):
    data_list.append([])
    for L_GI in L_GI_list:
        G.set_geometry(L_HS*1e-6, L_GI*1e-6)
        waist = 2 * G.beam_OUT.waist_profile.w * 1e6
        wd = G.beam_OUT.waist_position * 1e6
        data_list[i].append([waist, wd])

data = numpy.array(data_list)

# Trace the plots
pyplot.figure()
pyplot.title("Waist diameter versus GRIN length")
for (i, L_HS) in enumerate(L_HS_list):
    pyplot.plot(L_GI_list, data[i,:,0], label="L_HS = {:} µm".format(L_HS))
pyplot.xlabel("Gradient index segment length (GI 85/125) (µm)")
pyplot.ylabel("Waist diameter (µm)")
pyplot.legend()

pyplot.figure()
pyplot.title("Working distance versus GRIN length")
for (i, L_HS) in enumerate(L_HS_list):
    pyplot.plot(L_GI_list, data[i,:,1], label="L_HS = {:} µm".format(L_HS))
pyplot.xlabel("Gradient index segment length (GI 85/125) (µm)")
pyplot.axhline(color="k")
pyplot.ylabel("Working distance (µm)")
pyplot.legend()

pyplot.show()
