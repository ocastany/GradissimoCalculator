#!/usr/bin/python3
# encoding: utf-8

"""
Module for calculation of Gaussian beams.
Application to Gradissimo fibers. 

Usage: see at the bottom of this page.
"""

import numpy, cmath
from numpy import pi, inf, sqrt, cos, sin, sinh, cosh, tanh, arccos, real, imag
from numpy import arctan, tan, linspace
from scipy import optimize
from matplotlib import pyplot

lbda0 = None    # Wavelength in vacuum, for example 1.31e-6 m
                # Set to None, so that the user does not forget to set it.
n0 = 1.45   # For reference, the refractive indices of silicon and silica are
            # Si   : 3.500 @ 1.31 µm, 3.476 @ 1.55 µm
            # SiO2 : 1.447 @ 1.31 µm, 1.444 @ 1.55 µm 

def set_wavelength(lbda):
    """Set the vacuum wavelength for subsequent calculations"""
    global lbda0
    lbda0 = lbda


class GaussianProfile:
    """Description of a Gaussian profile in a transverse plane.
    
    The field amplitude of a Gaussian profile is proportional to
    
    E(r) = exp(-a r²) = exp(-ik0 r²/(2Q)) = exp(-ik0 C r²/2 - r²/w²)
    
    With    a = ik0/(2 Q) = iπ/(λ0 Q)
            1/Q = C - iλ0/(πw²)
    
    k0 = 2π / λ0
    Q : reduced Gaussian parameter
    w : radius at 1/e² intensity [m]
    C : reduced curvature [m⁻¹]
    a : coefficient of the Gaussian

    Note : 
    * If that profile belongs to a homogeneous medium of index n, the reduced
      curvature C is connected to the radius of curvature R of the wavefront 
      by C = n/R (see the class BeamInHomogeneousSpace)
    * The reduced curvature C is positive when the center is on the left.
    * If the profile is at a boundary between two different materials,
      all the quantities "a", "Q", "w" and "C" are conserved.
    * The values may be numbers or arrays of numbers.
    """

    Q = None    # Reduced Gaussian parameter
    w = None    # Radius at 1/e² intensity
    C = None    # Reduced curvature [m⁻¹]
    a = None    # Gaussian coefficient    
    
    def __init__(self, Q=None, w=None, C=0.0, a=None):
        """Create a Gaussian profile.
        
        Give either "Q", "w" or "a" (possibly as a Numpy array).
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
        self.Q = Q = 1j*pi / (lbda0 * a)
        self.w = sqrt(-lbda0/(pi* (1/Q).imag))
        self.C = (1/Q).real
        self.a = a
    
    def set_ReducedGaussianParameter(self, Q):
        """Define profile by Reduced Gaussian parameter 'Q'.
        
        E(r) = exp(-ik0 r²/(2Q))
        """
        self.Q = Q
        self.w = sqrt(-lbda0/(pi* (1/Q).imag))
        self.C = (1/Q).real
        self.a = 1j*pi / (lbda0 * Q)

    def set_Geometry(self, w, C):
        """Define profile by radius 'w' and reduced curvature 'C'.
        
        E(r) = exp(-ik0 C r²/2 - r²/w²)
        """
        self.Q = 1 / (C - 1j*lbda0/(pi * w**2))
        self.w = w
        self.C = C
        self.a = 1j*pi / (lbda0 * self.Q)
        
    def propagate(self, s, L):
        """Propagate this profile in space 's' over length 'L'."""
        return s.propagate(self, L)

    def transform(self, oe):
        """Transform profile through OpticalElement 'oe'."""
        return oe.transform(self)

    def beam(self, s):
        """Return the beam created by this profile in space 's'."""
        return s.beam(self)

    @property
    def diameter(self):
        """Beam diameter at 1/e² intensity."""
        return 2*self.w
    
    def R(self, n):
        """Radius of curvature in a medium of refractive index 'n'."""
        return n / self.C


class Beam:
    """Abstract class for a gaussian beam in a certain space."""
    
    space = None        # Space in which the beam propagates.
    profile = None      # Gaussian transverse profile at reference plane 
                        # in the space.
    
    def __init__(self, space=None, profile=None):
        """Creates a Beam with the given GaussianProfile in the given Space."""
        self.set_space(space)
        self.set_profile(profile)

    def set_space(self, space):
        """Defines the Space in which the Beam propagates."""
        self.space = space

    def set_profile(self, profile):
        """Defines the reference GaussianProfile of the Beam."""
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
        """Return GaussianProfile at position 'z'.
        
        'z' : number or array, position with respect to the reference.
        """
        return GaussianProfile(self.get_Q(z))        

    def change_origin(self, z):
        """Change origin to 'z'."""
        self.set_profile(self.get_profile(z))
        
    def evolution(self, z1=0.0, z2=0.0):
        """Return the evolution of 'w' as a function of 'z'.
        
        Returns : [Z,W] 
        where Z = linspace(z1,z2) 
              W = values of the radius at 1/e².
        """
        Z = linspace(z1,z2)
        W = self.get_profile(Z).w
        return numpy.array([Z,W])

    
class BeamInHomogeneousSpace(Beam):
    """Gaussian beam in a homogeneous material with refractive index 'n'.

    In a homogeneous material, the profile is characterized by the 
    Gaussian parameter 'q'. The paraxial propagation is
        
        E(z,r) = exp(-ikz) exp(-ikr²/(2 q(z)))
    
    with the propagation law q(z) = q(0) + z.
   
    E(r) = exp(-a r²) 
         = exp(-ik0 r²/(2Q))
         = exp(-ik r²/(2q))
         = exp(-ik r²/(2R) - r²/w²)
    
    With    k = k0 * n
            q(z) = Q(z) * n
            1/q = 1/R - iλ/(πw²)
    
    R(z) is the radius of curvature (positive when the center is on the left)
    
    The Rayleigh length is zR = Im(q(0)) = π w₀²/λ
    The half angle divergence is λ / (π w₀)
            
    At a material interface between homogeneous materials, the 
    reduced gaussian parameter Q = q/n is identical on both sides.
    """
    
    waist_position = None   # waist position [m]
    waist_profile = None    # Gaussian transverse profile at waist
    zR = None               # Rayleigh range in the space
    divergence = None       # Half divergence angle (1/e²) [rad]
    
    def set_beam(self):
        """Build the beam in our HomogeneousSpace."""
        n = self.space.n
        q = self.profile.Q * n
        self.zR = q.imag
        self.waist_profile = GaussianProfile(1j * self.profile.Q.imag)
        self.waist_position = -q.real
        self.divergence = lbda0/n / (pi * self.waist_profile.w)
    
    def get_Q(self, z):
        """Return Q value at position 'z'."""
        return self.profile.Q + z/self.space.n
        
    def get_R(self, z):
        """Return R value at position 'z'."""
        Q = self.get_Q(z)
        n = self.space.n
        return n / real(1/Q)

    def plot(self, z1=0.0, z2=0.0):
        """Plot beam radius versus position."""
        pyplot.figure()
        pyplot.title("Beam radius at 1/e² intensity")
        pyplot.axvline(0)
        pyplot.axhline(0)
        pyplot.xlabel("z (um)")
        pyplot.ylabel("w (um)")
        (Z, W) = self.evolution(z1, z2)*1e6
        pyplot.plot(Z, W)


class BeamInGradientIndex(Beam):
    """Beam in a gradient index (GRIN) fiber
    
    A gradient index fiber has a refractive index profile given by
        n(r) = n₀(1 - A/2 r²) = n₀(1 - 1/2 g² r²),

    where we defined g ≡ √(A). The evolution of a beam is
        Q(z) = 1/(n g) tan(gz + theta)
    
    Spatial pulsation for ray optics: g
    Spatial period for ray optics: P = 2 pi/g
    Spatial period for the shape of the Gaussian beam: P/2 = pi/g
   
    Note :
    Im(Q(z)) = 1/(n g) sh(θ") ch(θ") / (cos²(gz+θ') + sh²(θ"))
    C = Re(1/Q(z)) = n g (sin(gz+θ') cos(gz+θ')) / (sin²(gz+θ') + sh²(θ"))
    """
    
    theta = None        # Complex angle, reference for the oscillating beam
                        # We have Im(theta) != 0 because Im(Q) != 0
    w_min = w_max = None

    def set_beam(self):
        """Build the beam in our GradientIndexFiber."""
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
                

class Space:
    """Abstract class for a propagation space"""

    def propagator(self, L):
        """Returns propagator for length L."""
        raise NotImplementedError("Should be implemented in derived classes")
        
    def propagate_Q(self, Q, L):
        """Propagates GaussianProfile Q over length L."""
        raise NotImplementedError("Should be implemented in derived classes")
       
    def beam(self, p=None):
        """Returns a GaussianBeam defined by a GaussianProfile.

        p : GaussianProfile object or reduced curvature Q
        """
        raise NotImplementedError("Should be implemented in derived classes")
 
    def propagate(self, p, L):
        """Propagates GaussianProfile 'p' over length L.

        Return : GaussianProfile after propagation
        """
        Q_L = self.propagate_Q(p.Q, L)
        return GaussianProfile(Q_L)

       
class HomogeneousSpace(Space):
    """A homogeneous space of refractive index 'n'."""
    
    n = None        # refractive index of the material
    
    def __init__(self, n=n0):
        """Defines a homogeneous space of index n."""
        self.n = n

    def propagator(self, L):
        """Returns propagator for length L."""
        n = self.n
        M = numpy.identity(2)
        M[0,1] = L / n
        return Propagator(M,n,n)
        
    def propagate_Q(self, Q, L):
        """Propagate over length L.
        
        Q : reduced Gaussian parameter to propagate

        Return : Q after propagation
        """
        return Q + L/self.n

    def beam(self, p=None):
        """Return a GaussianBeam"""
        if not isinstance(p, GaussianProfile):
            p = GaussianProfile(p)
        return BeamInHomogeneousSpace(self, p)

    
class GradientIndexFiber(Space):
    """GRIN fiber (MultiMode Fiber)

    Refractive index profile : n(r) = n₀(1 - 1/2 g² r²)
    """
    
    # MMF fiber parameters:
    n = 1.471           # refractive index at the central point
    gamma = 5.7e3       # spatial pulsation [m⁻¹]
    diam = 62.5e-6      # core diameter     [m]
    
    P = 2*pi/gamma      # spatial period for a ray [m] 
    # For a Gaussian beam, what matters is the half-period P/2 = pi/gamma
    # Typical value: P = 1.1 mm and P/2 = 550 µm

    def __init__(self, n=1.469, gamma=None, diam=None, n_cl=None):
        """Defines a MMF"""
        self.n = n
        self.diam = diam
        if gamma is not None:
            self.gamma = gamma
        elif n_cl is not None:
            self.gamma = sqrt(2 * (n - n_cl)/n) / (diam/2)
            
        self.P = 2*pi/self.gamma
        
    def propagator(self, L):
        """Propagator for length L"""
        g = self.gamma
        n = self.n
        z = L
        M = numpy.array([[cos(g*z)         , 1/(n*g) * sin(g*z)],
                         [-n*g * sin(g*z) , cos(g*z)           ]])
        return Propagator(M,n,n)
        
    def propagate_Q(self, Q, L):
        """Propagate profile Q over length L

        Return : Q after propagation
        """
        n = self.n
        return 1/n*self.propagator(L).propagate_q(Q*n)

    def beam(self, p=None):
        """Return a Gradient Index Beam"""
        if not isinstance(p, GaussianProfile):
            p = GaussianProfile(p)
        return BeamInGradientIndex(self, p)
        
    def get_index(self, r=0):
        """Return refractive index at radius 'r'."""
        if r > self.diam/2:
            r = self.diam/2
        n = self.n * (1 - 1/2 * (self.gamma * r)**2)
        return n
    

class SingleModeFiber:
    """Optical Fiber"""
    
    n = None            # refractive index of the fiber cladding
    w = None            # mode radius at 1/e² [m]
    profile = None      # Gaussian profile of the mode
 
    def __init__(self, w=5.2e-6, n=n0):
        """Create a single-mode fiber with given mode radius.
        
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
        """Creates a propagator from the given ABCD matrix."""
        self.n1 = n1
        self.n2 = n2
        self.M = M
        
    def propagate_q(self, q1):
        """Returns value of q2 for the given q1 by applying ABCD matrix."""
        [[A,B],[C,D]] = self.M
        (n1, n2) = (self.n1, self.n2)
        q2 = n2 * (A * (q1/n1) + B) / (C * (q1/n1) + D)
        return q2    

class OpticalElement:
    """Optical elements affecting a wavefront."""

    def transform_Q(self, Q1):
        """Transform GaussianProfile Q through that OpticalElement."""
        raise NotImplementedError("Should be implemented in derived classes")

    def transform(self, profile):
        """Transform GaussianProfile 'profile' through that OpticalElement.

        See transform_Q() for more information.
        """
        return GaussianProfile(self.transform_Q(profile.Q))


class Diopter(OpticalElement):
    """Refracting surface n1 | n2, with radius of curvature R = SC

    S = Apex of the diopter   
    C = Center of curvature
    The sign is negative when C is at the left of S.
    """

    R = None        # Radius of curvature [m]
    c = None        # Surface curvature [m⁻¹]

    def __init__(self, n1=1.0, n2=1.0, R=inf, c=None):
        """Create a Diopter object with curvature

        R = SC      Radius of curvature [m]
        c = 1/SC    Curvature [m⁻¹]
        """
        if isinstance(n1, Space):
            self.n1 = n1.n
        else:
            self.n1 = n1

        if isinstance(n2, Space):
            self.n2 = n2.n
        else:
            self.n2 = n2

        if c is not None:
            if c == 0:
                self.R = inf
                self.c = 0
            else:
                self.R = 1/c
                self.c = c
        else:
            self.R = R
            self.c = 1/R

    def transform_Q(self, Q1):
        """Transform GaussianProfile Q1 through that Diopter.

        F  : object focal point      F' : image focal point
        f = SF  = -n/(n'-n) SC       f' = SF' = n'/(n'-n) SC

        1/Q - 1/Q' = n/q - n'/q' = -n/f = n'/f'

        Return : Q after the Diopter
        """
        n1, n2 = self.n1, self.n2
        inv_f = -(n2 - n1) / n1 * self.c
        Q2 = 1 / (1/Q1 + n1*inv_f)
        return Q2

class Lens(OpticalElement):
    """Lens of image focal distance 'f'."""

    def __init__(self, n=n0, f=inf):
        """Create a lens with IMAGE focal distance 'f' in medium 'n'."""
        self.f = f
        if isinstance(n, Space):
            self.n = n.n
        else:
            self.n = n

    def transform_Q(self, Q1):
        """Transform the GaussianProfile Q1 through that Lens.

        1/Q - 1/Q' = n/f

        Return : GaussianProfile after the Lens
        """
        f = self.f
        n = self.n

        Q2 = 1 / (1/Q1 - n/f)
        return Q2

class Mirror(OpticalElement):
    """Mirror with RADIUS OF CURVATURE R = SC
    
    S = Apex of the diopter   
    C = Center of curvature
    The sign is R > 0 for a diverging mirror.
    
    The OBJECT focal length is f = R/2    
    The propagation axis is reversed after the reflection.
    """
    
    def __init__(self, n=n0, R=inf):
        """Create a mirror with radius of curvature 'R' in medium 'n'."""
        self.R = R
        self.f = R/2
        if isinstance(n, Space):
            self.n = n.n
        else:
            self.n = n

    def transform_Q(self, Q1):
        """Transform the GaussianProfile Q1 through that Mirror.
        
        1/q - 1/q'  = -1/f
        1/Q - 1/Q'  = -n/f
        """
        f = self.f
        n = self.n

        Q2 = 1 / (1/Q1 + n/f)
        return Q2
        
        
class Gradissimo:
    """Gradissimo fiber
    
    Geometry of a Gradissimo fiber:
            
               IF         HS       GI         OUT
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁    
        ▁▁▁▁▁▁▁▁▁▁▁▁▁          |         |         
                     |  L_HS   |  L_GI   |   L_OUT |   
        ▔▔▔▔▔▔▔▔▔▔▔▔▔          |         |       
        ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔                     
                     Q0        Q1        Q2       Q3

    Attributes...
    - Reduced Gaussian parameters : Q0, Q1, Q2, Q3
    - Section lengths : L_HS, L_GI, L_OUT
    - Beams : beam_HS, beam_GI, beam_OUT
    """
    
    IF = None       # SingleModeFiber       (input fiber)
    HS = None       # HomogeneousSpace      (homogeneous fiber)
    GI = None       # GradientIndexFiber    (GI fiber)
    OUT = None      # HomogeneousSpace      (output space)

    Q0 = Q1 = Q2 = Q3 = None
    L_HS = L_GI = L_OUT = None
    beam_HS = beam_GI = beam_OUT = None

    def __init__(self, IF, HS, GI, OUT, L_HS=None, L_GI=None):
        """Create a Gradissimo fiber."""
        self.IF = IF ; self.HS = HS ; self.GI = GI ; self.OUT = OUT
        self.set_geometry(L_HS, L_GI)
        
    def set_geometry(self, L_HS=None, L_GI=None):
        """Calculate the characteristics of the beams."""
        if L_HS is None:
            L_HS = self.L_HS
        else:
            self.L_HS = L_HS
        if L_GI is None:
            L_GI = self.L_GI
        else:
            self.L_GI = L_GI
        if (L_HS is None) or (L_GI is None):
            return      # Give up because Gradissimo fiber is not fully defined

        self.Q0 = self.IF.profile.Q
        
        self.beam_HS = self.HS.beam(self.Q0)
        self.Q1 = self.beam_HS.get_Q(L_HS)
        # Equivalent to
        # self.Q1 = self.HS.propagate_Q(self.Q0, L_HS)
        
        self.beam_GI = self.GI.beam(self.Q1)
        self.Q2 = self.beam_GI.get_Q(L_GI)
        # Equivalent to
        # self.Q2 = self.GI.propagate_Q(self.Q1, L_GI)
        
        self.beam_OUT = self.OUT.beam(self.Q2)
        self.L_OUT = self.beam_OUT.waist_position
        self.Q3 = self.beam_OUT.waist_profile.Q
        # Equivalent to
        # self.Q3 = self.OUT.propagate_Q(self.Q2, self.L_OUT)
        
        
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
            # self.Q2 = self.OUT.propagate_Q(self.Q3, -L_OUT)
        
        # Find the length of the GI segment...
        self.beam_GI = self.GI.beam(self.Q2)
        self.Q0 = self.IF.profile.Q
        z1 = self.beam_GI.find_imag(self.Q0, C=+1, steps=-1-oscillations)
        self.beam_GI.change_origin(z1)
        self.L_GI = -z1
        self.Q1 = self.beam_GI.profile.Q
        # Equivalent to
        # self.Q1 = self.GI.propagate_Q(Q2, z1)
        
        # Find the length of the HS segment...
        self.beam_HS = self.HS.beam(self.Q1)
        z0 = self.beam_HS.waist_position
        self.beam_HS.change_origin(z0)
        self.L_HS = -z0
        
    
    def plot(self):
        """Plot the beam radius along the gradissimo fiber.
        
        If needed, use pyplot.show() to show the plot.
        """
        pyplot.figure()
        pyplot.title("Gradissimo beam radius")
        pyplot.axvline(0)
        pyplot.axhline(0)
        pyplot.xlabel("z (um)")
        pyplot.ylabel("w (um)")
        L_HS = self.L_HS * 1e6
        L_GI = self.L_GI * 1e6
        (Z1, W1) = self.beam_HS.evolution(0, self.L_HS)*1e6
        pyplot.plot(Z1, W1)
        pyplot.axvline(L_HS)
        (Z2, W2) = self.beam_GI.evolution(0, self.L_GI)*1e6
        pyplot.plot(L_HS + Z2, W2)
        if self.GI.diam is not None:
            pyplot.axhline(self.GI.diam/2 * 1e6)
        pyplot.axvline(L_HS+L_GI)
        (Z3, W3) = self.beam_OUT.evolution(0, self.L_GI)*1e6
        pyplot.plot(L_HS + L_GI + Z3, W3)
    

if __name__ == "__main__":
    print("Example of use...")
    set_wavelength(1.3e-6)          # Wavelength [m]
    # See example file EX0

