"""Geometry for a two-link rigid surrogate of diagonal cloth folding.

Coordinates are metres. Quaternions in this module are always (x, y, z, w).
No Isaac Sim, Torch, or NumPy dependency: the geometry can be checked anywhere.
"""
from dataclasses import dataclass
import math

SQRT2 = math.sqrt(2.0)
HINGE_AXIS = (1.0 / SQRT2, 1.0 / SQRT2, 0.0)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, s):
    return tuple(s * x for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def rotate(q, v):
    """Rotate a vector by a unit XYZW quaternion."""
    t = scale(cross(q[:3], v), 2.0)
    return add(v, add(scale(t, q[3]), cross(q[:3], t)))


def inverse_rotate(q, v):
    return rotate((-q[0], -q[1], -q[2], q[3]), v)


def axis_quat(axis, angle):
    return (*scale(axis, math.sin(angle / 2.0)), math.cos(angle / 2.0))


def slerp(q0, q1, u):
    d = dot(q0, q1)
    if d < 0:
        q1 = scale(q1, -1.0)
        d = -d
    d = min(1.0, max(-1.0, d))
    if d > 0.9995:
        q = add(scale(q0, 1-u), scale(q1, u))
        return scale(q, 1.0 / norm(q))
    a = math.acos(d)
    return add(scale(q0, math.sin((1-u)*a)/math.sin(a)),
               scale(q1, math.sin(u*a)/math.sin(a)))


def smoothstep(u):
    """Quintic interpolation: zero endpoint velocity and acceleration."""
    u = max(0.0, min(1.0, u))
    return u*u*u*(10.0 + u*(-15.0 + 6.0*u))


def hinge_angle(q):
    """Signed hinge angle near the working interval [0, pi], ignoring q's sign."""
    a = 2.0 * math.atan2(dot(q[:3], HINGE_AXIS), q[3])
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a + 2.0 * math.pi if a < -math.pi/2 else a


@dataclass(frozen=True)
class Board:
    side: float = 0.30
    thickness: float = 0.016
    seam: float = 0.0015
    table_top: float = 0.60
    center_xy: tuple = (0.48, 0.0)
    handle_extension: float = 0.045
    handle_radius: float = 0.009

    def __post_init__(self):
        if not (0.16 <= self.side <= 0.40):
            raise ValueError("Use a board side between 0.16 and 0.40 m for this Franka layout.")
        if not (0 < self.seam < self.side/10):
            raise ValueError("The seam must be positive and small relative to the board.")
        if self.thickness <= self.handle_radius:
            raise ValueError("The handle must clear the table at both ends of the fold.")

    @property
    def origin(self):
        return (*self.center_xy, self.table_top + 0.001 + self.thickness/2)

    @property
    def hinge_local(self):
        # Put the axis on the upper face, not through the panels' midplanes.
        # This allows the two finite-thickness panels to stack at 180 degrees.
        return (0.0, 0.0, self.thickness/2)

    @property
    def hinge_world(self):
        return add(self.origin, self.hinge_local)

    @property
    def handle_local(self):
        a = self.side/2 + self.handle_extension/SQRT2
        return (-a, a, self.thickness/2)

    def triangle(self, moving):
        a, g = self.side/2, self.seam
        # Both polygons are counterclockwise, separated by a narrow diagonal gap.
        if moving:
            return [(-a, -a+g), (a-g, a), (-a, a)]
        return [(-a+g, -a), (a, -a), (a, a-g)]

    def prism(self, moving):
        polygon = self.triangle(moving)
        t = self.thickness/2
        vertices = [(x,y,z) for z in (-t,t) for x,y in polygon]
        faces = [(2,1,0), (3,4,5), (0,1,4,3), (1,2,5,4), (2,0,3,5)]
        return vertices, faces

    def folded_point(self, point_local, angle):
        relative = sub(point_local, self.hinge_local)
        return add(self.hinge_world, rotate(axis_quat(HINGE_AXIS, angle), relative))

    def handle_world(self, angle):
        return self.folded_point(self.handle_local, angle)
