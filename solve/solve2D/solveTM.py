"""
Purpose:        Solves the frequency-domain Maxwell's equations
                in matrix form on a discretized Yee's grid using 
                user defined boundary conditions for 2D TM fields.
Author:         David Heydari (Feb. 2021)
Assumptions:    Periodic boundary conditions and uniform discretization
                Δx, Δy, and Δz.
                By TE, we mean [Hz,Ex,Ey]; jz is the only source needed.
                The material is extruded in the z-direction and constant index.
"""

from ..operator import differences
import numpy as np
from scipy.sparse import bmat
from scipy import linalg

def solveHz(Ce, Ch, ω0, Tɛ, Tμ, j):  # j: [0,0,jz]
    A = Ce * linalg.inv(Tɛ) * Ch - ω0**2 * Tμ
    b = Ce * linalg.inv(Tɛ) * j
    return linalg.inv(A)*b

def solveE(Ce, Ch, ω0, Tɛ, Tμ, j, h):  # h: [0,0,hz], j: [0,0,jz]
    return (1J/ω0) * linalg.inv(Tɛ) * (Ch * h - j)