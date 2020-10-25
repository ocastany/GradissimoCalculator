from gradissimo import *

lbda = 1.3e-6               # Wavelength 
n_SiO2 = 1.447              # SiO2 @ 1.3 µm
n_Si   = 3.500              # Si   @ 1.3 µm
n0 = 1.0

set_wavelength(lbda)
SiO2 = HomogeneousSpace(n_SiO2)
Si   = HomogeneousSpace(n_Si)
air  = HomogeneousSpace(n0)

w0 = 5e-6 ; L_Si = 550e-6 ; L_SiO2 = 602e-6 ; R = -266e-6

p0 = GaussianProfile(w=w0)
p1 = p0.propagate(Si, L_Si)
p2 = p1.propagate(SiO2, L_SiO2)
p3 = p2.transform(Diopter(SiO2, air, R))
b =  p3.beam(air)

print("Waist diameter {:.1f} µm à {:.2f} mm".format(
      b.waist_profile.w*2e6, b.waist_position*1e3))

