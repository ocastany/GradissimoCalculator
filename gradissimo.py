#!/usr/bin/python3
# encoding: utf-8

"""
Calculation with Gaussian beams
Application to Gradissimo fiber 
"""

import numpy, cmath
from numpy import pi, inf, sqrt, cos, sin, sinh, cosh, tanh, arccos 
from numpy import arctan, tan, linspace
from scipy import optimize
import matplotlib.pyplot as pyplot

lbda0 = 1.31e-6     # Wavelength in vacuum

def set_wavelength(lbda):
    """Set the vacuum wavelength for subsequent calculations"""
    global lbda0
    lbda0 = lbda


class GaussianProfile:
    """Description of a Gaussian profile.
    
    The field amplitude of a Gaussian profile is 
    
    E(r) = exp(-a r²) 
            = exp(-ik0 r²/(2Q))
            = exp(-ik0 C r²/2 - r²/w²)
    
    With    a = ik0/(2 Q) = iπ/(λ0 Q)
            1/Q = C - iλ0/(πw²)
    
    Q : reduced Gaussian parameter
    w : radius at 1/e² intensity [m]
    C : reduced curvature [m⁻¹]
    a : Gaussian coefficient

    Note : 
    * reduced curvature C is positive when center is on the left.
    * if the profile is at the boundary between two different materials,
      all the quantities mentioned above (a, Q, w and C) are conserved.
    """
    
    Q = None    # Reduced Gaussian parameter
    w = None    # Radius at 1/e² intensity
    C = None    # Reduced curvature [m⁻¹]
    a = None    # Gaussian coefficient
    
    def __init__(self, Q=None, w=None, C=0.0, a=None):
        """Create a Gaussian profile.
        
        Data may be a Numpy array.
        """
        if Q is not None:
            self.set_ReducedGaussianParameter(Q)
        elif w is not None:
            self.set_Geometry(w, C)
        elif a is not None:
            self.set_GaussianCoefficient(a)
            
    def set_GaussianCoefficient(self, a):
        """Define profile by Gaussian coefficent 'a'.
        
        E(r) = exp(-a r²)
        """
        self.a = a
        self.Q = 1j*pi / (lbda0 * a)
        self._calculate_geometry(self.Q)
    
    def set_ReducedGaussianParameter(self, Q):
        """Define profile by Reduced Gaussian parameter 'Q'.
        
        E(r) = exp(-ik0 r²/(2Q))
        """
        self.a = 1j*pi / (lbda0 * Q)
        self.Q = Q
        self._calculate_geometry(self.Q)

    def set_Geometry(self, w, C):
        """Define profile by waist 'w' and reduced curvature 'C'.
        
        E(r) = exp(-ik0 C r²/2 - r²/w²)
        """
        self.w = w
        self.C = C
        self.Q = 1 / (C - 1j*lbda0/(pi * w**2))
        self.a = 1j*pi / (lbda0 * self.Q)
        
    def _calculate_geometry(self, Q):
        """Calculate w and C from Q."""
        self.w = sqrt(-lbda0/(pi* (1/Q).imag))
        self.C = (1/Q).real


class Beam:
    """Beam abstract class"""
    
    space = None        # Space in which the beam propagates
    profile = None      # Gaussian transverse profile at reference plane
    
    def __init__(self, space=None, profile=None):
        """Creates a beam for the given profile and space."""
        self.set_space(space)
        self.set_profile(profile)

    def set_space(self, space):
        """Defines the space in which the beam propagates."""
        self.space = space

    def set_profile(self, profile):
        """Defines the reference profile of the Beam."""
        self.profile = profile
        if profile.Q is not None:
            self.set_beam()

    def set_beam(self):
        """Setup the beam, based on space and reference profile."""
        raise NotImplementedError("Should be implemented in derived classes")
            
    def get_Q(self, z):
        """Return Q value at position 'z'."""
        raise NotImplementedError("Should be implemented in derived classes")

    def get_profile(self, z):
        """Return Gaussian profile at position 'z'."""
        return GaussianProfile(self.get_Q(z))        

    def change_origin(self, z):
        """Change origin to 'z'."""
        self.set_profile(self.get_profile(z))
        
    def evolution(self, z1=0.0, z2=0.0):
        """Return (z,w) evolution."""
        Z = linspace(z1,z2)
        W = self.get_profile(Z).w
        return (Z,W)

    
class Beam_HomogeneousSpace(Beam):
    """Gaussian beam in a homogeneous material of refractive index 'n'.

    In a homogeneous material, the profile is characterized by the 
    Gaussian parameter 'q'
    
    E(r) = exp(-a r²) 
         = exp(-ik0 r²/(2Q))
         = exp(-ik r²/(2q))
         = exp(-ik r²/(2R) - r²/w²)
    
    With    k = k0 * n
            q = Q * n
            1/q = 1/R - iλ/(πw²)
    
    R is the radius of curvature (positive when center is on the left)
            
    The paraxial propagation of a Gaussian beam is
    
    E(z,r) = exp(-ikz) exp(-ikr²/(2 q(z))) 
           
    with the propagation law q(z) = q(0) + z
    
    At a material interface between homogeneous materials, the quantity Q = q/n
    is identical on both sides.
    """
    
    waist_position = None   # waist position
    waist_profile = None    # Gaussian transverse profile at waist
    zR = None               # Rayleigh range
    divergence = None       # Half divergence angle (1/e²) [rad]
    
    def set_beam(self):
        """Build the Gaussian beam"""
        n = self.space.n
        q = self.profile.Q * n
        self.zR = q.imag
        self.waist_profile = GaussianProfile(1j * self.profile.Q.imag)
        self.waist_position = -q.real
        self.divergence = lbda0 / (pi * n * self.waist_profile.w)
    
    def get_Q(self, z):
        """Return Q value at position 'z'."""
        return self.profile.Q + z/self.space.n

    def plot(self, z1=0.0, z2=0.0):
        """Plot beam waist."""
        pyplot.figure()
        pyplot.axvline(0)
        pyplot.axhline(0)
        (Z, W) = self.evolution(z1, z2)
        pyplot.plot(Z, W)


class Beam_GradientIndex(Beam):
    """Beam in a gradient index fiber
    
    Q(z) = 1/(n g) tan(gz + theta)
    
    Spatial period for ray : P = 2 pi/g
    Spatial period for Gaussian beam : P/2 = pi/g
    
    Im(Q(z)) = 1/(n g) sh(θ") ch(θ") / (cos²(gz+θ') + sh²(θ"))
    C = Re(1/Q(z)) = n g (sin(gz+θ') cos(gz+θ')) / (sin²(gz+θ') + sh²(θ"))
    """
    
    theta = None        # Complex angle, reference for the oscillating beam
                        # We have Im(theta) != 0 because Im(Q) != 0
                        
    def set_beam(self):
        """Build the GI beam"""
        g = self.space.gamma
        n = self.space.n
        theta = arctan(g * n * self.profile.Q)
        self.theta = theta
        self.w_min = GaussianProfile(1j/(n*g) * tanh(theta.imag)).w
        self.w_max = self.w_min/tanh(theta.imag)
    
    def get_Q(self, z):
        """Return Q at position 'z'."""
        g = self.space.gamma
        n = self.space.n
        return 1/(n*g) * tan(g*z + self.theta)
    
    def find_imag(self, Q, C=+1, steps=0):
        """Return 'z' position where Im(Q(z)) = Im(Q)
        
        'C' = +1/-1 (sign of the curvature)
        'steps' : number of half-periods to add (P/2 = pi/gamma)
        """
        g = self.space.gamma
        n = self.space.n
        theta = self.theta
        
        cosine_sq = sinh(theta.imag) * cosh(theta.imag) / (n*g*Q.imag) \
                    - sinh(theta.imag)**2
        
        cosine = C * sqrt(cosine_sq)
        z = (arccos(cosine) - theta.real + steps*pi) / g
        return z
                
        
class HomogeneousSpace:
    """A homogeneous space of refractive index 'n'."""
    
    n = None        # refractive index of the material
    
    def __init__(self, n=1.0):
        """Defines a homogeneous space of index n."""
        self.n = n

    def propagator(self, L):
        """Returns propagator for length L."""
        n = self.n
        M = numpy.identity(2)
        M[0,1] = L / n
        return Propagator(M,n,n)
        
    def propagate(self, Q, L):
        """Propagate over length L.
        
        Q : GaussianProfile to propagate
        """
        return Q + L/self.n
        
    def beam(self, Q=None):
        """Return a GaussianBeam"""
        return Beam_HomogeneousSpace(self, GaussianProfile(Q))

    
class GradientIndexFiber:
    """MultiMode Fiber"""
    
    # MMF fiber parameters    
    n = 1.44
    gamma = 5.7e3       # spatial pulsation [m⁻¹]
    diam = 62.5e-6      # core diameter     
    
    P = 2*pi/gamma      # spatial period for a ray [m] (= 1.1 mm)
    # For a Gaussian beam, what matters is the half-period P/2 = pi/gamma
    # Typical value, P/2 = 550 µm

    
    def __init__(self, n=1.0, gamma=5.7e3, diam=None):
        """Defines a MMF"""
        self.n = n
        self.gamma = gamma
        self.P = 2*pi/gamma
        self.diam = diam
        
    def propagator(self, L):
        """Propagator for length L"""
        g = self.gamma
        n = self.n
        z = L
        M = numpy.array([[cos(g*z)         , 1/(n*g) * sin(g*z)],
                         [-n*g * sin(g*z) , cos(g*z)           ]])
        return Propagator(M,n,n)
        
    def propagate(self, Q, L):
        """Propagate profile Q over length L"""
        n = self.n
        return 1/n*self.propagator(L).propagate(Q*n)

    def beam(self, Q=None):
        """Return a Gradient Index Beam"""
        return Beam_GradientIndex(self, GaussianProfile(Q))
        
            
    
class SingleModeFiber:
    """Optical Fiber"""
    
    n = None            # fiber refractive index (cladding)
    w = None            # mode radius 1/e² [m]
    profile = None      # Gaussian profile of the mode
 
    def __init__(self, w=5.2e-6, n=1.44):
        """Create a single mode fiber with given waist.
        
        'w' : mode radius (1/e²) [m]
        'n' : refractive index of the fiber
        """
        self.w = w
        self.n = n 
        self.profile = GaussianProfile(w=w)
        
        
class Propagator:
    """ABCD matrix propagator for Q = q/n."""
    
    M = numpy.identity(2)   # ABCD matrix
    n1 = 1.0                # Refraction index for input medium
    n2 = 1.0                # Refraction index for output medium
    
    def __init__(self, M, n1=1.0, n2=1.0):
        """Creates a propagator from ABCD matrix."""
        self.n1 = n1
        self.n2 = n2
        self.M = M
        
    def propagate(self, q1):
        """Returns value of q2 for the given q1 by applying ABCD matrix."""
        [[A,B],[C,D]] = self.M
        (n1, n2) = (self.n1, self.n2)
        q2 = n2 * (A * (q1/n1) + B) / (C * (q1/n1) + D)
        return q2    


class Gradissimo:
    """Gradissimo fiber
    
    Geometry of a Gradissimo fiber:
                        
            
          input_fiber    HS       GI         OUT
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁    
        ▁▁▁▁▁▁▁▁▁▁▁▁▁          |         |         
                     |  L_HS   |  L_GI   |   L_OUT |   
        ▔▔▔▔▔▔▔▔▔▔▔▔▔          |         |       
        ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔                     
                     Q0        Q1        Q2       Q3

    Available attributes...
    - Reduced Gaussian parameters : Q0, Q1, Q2, Q3
    - Section lengths : L_HS, L_GI, L_OUT
    - Beams : beam_HS, beam_GI, beam_OUT
    """
    
    input_fiber = None      # SingleModeFiber
    HS = None               # HomogeneousSpace
    GI = None               # GradientIndexFiber
    OUT = None              # HomogeneousSpace

    Q0 = Q1 = Q2 = Q3 = None
    L_HS = L_GI = L_OUT = None
    beam_HS = beam_GI = beam_OUT = None

    def __init__(self, input_fiber, HS, GI, OUT):
        """Create a Gradissimo fiber"""
        self.input_fiber = input_fiber
        self.HS = HS
        self.GI = GI
        self.OUT = OUT
        
    def set_geometry(self, L_HS, L_GI):
        """Return output beam"""
        self.L_HS = L_HS
        self.L_GI = L_GI
        self.Q0 = self.input_fiber.profile.Q
        
        self.beam_HS = self.HS.beam(self.Q0)
        self.Q1 = self.beam_HS.get_Q(L_HS)
        # Equivalent to
        # self.Q1 = self.HS.propagate(self.Q0, L_HS)
        
        self.beam_GI = self.GI.beam(self.Q1)
        self.Q2 = self.beam_GI.get_Q(L_GI)
        # Equivalent to
        # self.Q2 = self.GI.propagate(self.Q1, L_GI)
        
        self.beam_OUT = self.OUT.beam(self.Q2)
        self.L_OUT = self.beam_OUT.waist_position
        self.Q3 = self.beam_OUT.waist_profile.Q
        # Equivalent to
        # self.Q3 = self.OUT.propagate(self.Q2, self.L_OUT)
        
        
    def adjust_geometry(self, w_OUT=None, L_OUT=None, Q2=None, oscillations=0):
        """Determine the lengths of the 'GI' and 'HS' segments.
       
        'L_OUT' : position of the output waist
        'w_OUT' : radius at output waist position

        'Q2' : reduced gaussian parameter
               (can be provided instead of L_OUT and w_OUT)

        'oscillations' : number of half period to add.
        """
        if Q2 is not None:
            self.Q2 = Q2
            self.beam_OUT = self.OUT.beam(Q2)
            self.L_OUT = self.beam_OUT.waist_position
            self.Q3 = self.beam_OUT.waist_profile.Q
        else:
            self.Q3 = GaussianProfile(w=w_OUT).Q
            self.L_OUT = L_OUT
            self.beam_OUT = self.OUT.beam(self.Q3)
            self.beam_OUT.change_origin(-L_OUT)
            self.Q2 = self.beam_OUT.profile.Q
            # Equivalent to
            # self.Q2 = self.OUT.propagate(self.Q3, -L_OUT)
        
        # Find the length of the GI segment...
        self.beam_GI = self.GI.beam(self.Q2)
        self.Q0 = self.input_fiber.profile.Q
        z1 = self.beam_GI.find_imag(self.Q0, C=+1, steps=-1-oscillations)
        self.beam_GI.change_origin(z1)
        self.L_GI = -z1
        self.Q1 = self.beam_GI.profile.Q
        # Equivalent to
        # self.Q1 = self.GI.propagate(Q2, z1)
        
        # Find the length of the HS segment...
        self.beam_HS = self.HS.beam(self.Q1)
        z0 = self.beam_HS.waist_position
        self.beam_HS.change_origin(z0)
        self.L_HS = -z0
        
    
    def plot(self):
        """Plot the beam waist along the gradissimo fiber.
        
        If needed, use pyplot.show() to show the plot.
        """
        pyplot.figure()
        pyplot.axvline(0)
        pyplot.axhline(0)
        (Z1, W1) = self.beam_HS.evolution(0, self.L_HS)
        pyplot.plot(Z1, W1)
        pyplot.axvline(self.L_HS)
        (Z2, W2) = self.beam_GI.evolution(0, self.L_GI)
        pyplot.plot(self.L_HS + Z2, W2)
        if self.GI.diam is not None:
            pyplot.axhline(self.GI.diam/2)
        pyplot.axvline(self.L_HS+self.L_GI)
        (Z3, W3) = self.beam_OUT.evolution(0, self.L_GI)
        pyplot.plot(self.L_HS + self.L_GI + Z3, W3)
    


