"""
Purpose:        Solves the frequency-domain Maxwell's equations
                in matrix form on a discretized Yee's grid using 
                user defined boundary conditions for 2D TE fields.
Author:         David Heydari (Feb. 2021)
Assumptions:    Periodic boundary conditions and uniform discretization
                Δx, Δy, and Δz.
                By TE, we mean [ez,hx,hy]; we assume j = 0.
                The material is extruded in the z-direction and constant index.
"""

from ..operator import differences
import numpy as np
from scipy.sparse import bmat
from scipy import linalg

def solveEz(Ce, Ch, ω0, Tɛ, Tμ, j):
    A = Ce * linalg.inv(Tɛ) * Ch - ω0**2 * Tμ
    b = Ce * linalg.inv(Tɛ) * j
    return linalg.inv(A)*b

def solveH(Ce, Ch, ω0, Tɛ, Tμ, j, h):
    return (1J/ω0) * linalg.inv(Tɛ) * (Ch * h - j)