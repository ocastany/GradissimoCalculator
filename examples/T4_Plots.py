#!/usr/bin/python3
# encoding: utf-8

from gradissimo import *

# Study of Gradissimo fibers and comparison with reference articles
# [1] P. Chanclou, JLT, 17(5), p. 924 (1999)
# [2] M. Thual, Opt. Eng., 46(1), p. 015402 (2007)

# Value of the spatial pulsation
# GRIN 85/125 µm : 4294 m⁻¹ @ 1.3 µm    [1]
#                  4117 m⁻¹ @ 1.55 µm   [2]

# Refractive index of silica
# n = 1.447 @ 1.3 µm                    [1]
#     1.469 @ 1.55 µm

# We will reproduce figures 5 and 6 from [1]

# Define the wavelength and the materials
set_wavelength(1.3e-6)
n_fiber = 1.447
n_out = 1.000
input_fiber = SingleModeFiber(w=5.0e-6, n=n_fiber)
HS = HomogeneousSpace(n_fiber)
GI = GradientIndexFiber(n_fiber, gamma=4294, diam=85e-6)
OUT = HomogeneousSpace(n_out)

# Gradissimo fiber structure...
G = Gradissimo(input_fiber, HS, GI, OUT)

# Segment lengths to be used
L_HS_list = [0, 200, 400, 600]
L_GI_list = linspace(0, 800, num=200)

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
pyplot.title("Waist diameter for various lengths of silica (L_HS)")
for (i, L_HS) in enumerate(L_HS_list):
    pyplot.plot(L_GI_list, data[i,:,0], label="L_HS = {:} µm".format(L_HS))
pyplot.xlabel("Gradient index segment length (GI 85/125) (µm)")
pyplot.ylabel("Waist diameter (µm)")
pyplot.legend()

pyplot.figure()
pyplot.title("Working distance for various lengths of silica (L_HS)")
for (i, L_HS) in enumerate(L_HS_list):
    pyplot.plot(L_GI_list, data[i,:,1], label="L_HS = {:} µm".format(L_HS))
pyplot.xlabel("Gradient index segment length (GI 85/125) (µm)")
pyplot.axhline(color="k")
pyplot.ylabel("Working distance (µm)")
pyplot.ylim(ymin=0)
pyplot.legend()

pyplot.show()
