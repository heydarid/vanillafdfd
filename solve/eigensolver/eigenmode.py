"""
Purpose:        Implements the difference matrices on a given
                discretized Yee grid in 3d.  2d can be implemented
                by simply setting Nz = 1.  Solves the eigenmodes.
Author:         David Heydari (Jan. 2021)
Assumptions:    Uniform discretization Δx, Δy, and Δz.
"""

from .. import maxwell
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

def solveE(k_eigs, Ce, Ch, ω0, Tɛ, Tμ):
    invTμ = diags(1./Tμ.diagonal())
    A = Ch * invTμ * Ce - ω0**2 * Tɛ
    return eigs(A, k_eigs, which='SM')

def solveH(k_eigs, Ce, Ch, ω0, Tɛ, Tμ):
    invTɛ = diags(1./Tɛ.diagonal()) 
    A = Ce * invTɛ * Ch - ω0**2 * Tμ
    return eigs(A, k_eigs, which='SM')