"""
Purpose:        Solves the frequency-domain Maxwell's equations
                in matrix form on a discretized Yee's grid using 
                user defined boundary conditions.
Author:         David Heydari (Jan. 2021)
Assumptions:    Periodic boundary conditions and uniform discretization
                Δx, Δy, and Δz.
"""

"""
TODO:   Allow user to choose from different BCs
        >> FOR NOW, IMPLEMENTING PERIODIC BCs.
"""

from .operator import differences
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import spsolve

def Ce(Nx, Ny, Nz, Δx, Δy, Δz):
    return bmat([
                [None, -1*differences.Dz_f(Nx, Ny, Nz, Δz), differences.Dy_f(Nx, Ny, Nz, Δy)],
                [differences.Dz_f(Nx, Ny, Nz, Δz), None, -1*differences.Dx_f(Nx, Ny, Nz, Δx)],
                [-1*differences.Dy_f(Nx, Ny, Nz, Δy), differences.Dx_f(Nx, Ny, Nz, Δx), None]
                ])

def Ch(Nx, Ny, Nz, Δx, Δy, Δz):
    return bmat([
                [None, -1*differences.Dz_b(Nx, Ny, Nz, Δz), differences.Dy_b(Nx, Ny, Nz, Δy)],
                [differences.Dz_b(Nx, Ny, Nz, Δz), None, -1*differences.Dx_b(Nx, Ny, Nz, Δx)],
                [-1*differences.Dy_b(Nx, Ny, Nz, Δy), differences.Dx_b(Nx, Ny, Nz, Δx), None]
            ])


def solveE(Ce, Ch, ω0, Tɛ, Tμ, j, m):
    invTμ = diags(1./Tμ.diagonal())
    A = Ch * invTμ * Ce - ω0**2 * Tɛ
    b = -1J*ω0*j - Ch * invTμ * m
    return spsolve(A, b)

def solveH(Ce, Ch, ω0, Tɛ, Tμ, j, m):
    invTɛ = diags(1./Tɛ.diagonal()) 
    A = Ce * invTɛ * Ch - ω0**2 * Tμ
    b = -1J*ω0*m - Ce * invTɛ * j
    return spsolve(A, b)