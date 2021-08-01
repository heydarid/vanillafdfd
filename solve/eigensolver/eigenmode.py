"""
Purpose:        Implements the difference matrices on a given
                discretized Yee grid in 3d.  2d can be implemented
                by simply setting Nz = 1.  Solves the eigenmodes.
Author:         David Heydari (Jan. 2021)
Assumptions:    Uniform discretization Δx, Δy, and Δz.
"""
import scipy.constants as sc
from solve import maxwell
from scipy.sparse import csc_matrix, block_diag, bmat, diags
from scipy.sparse.linalg import inv, eigs
from solve.operator.differences import *

def Oh(tex, tey, tez, Nx, Ny, Δx, Δy):
    T = bmat([
                [tex, None],
                [None, tey]
            ])
    Fd1 = bmat([
                [-1*Dy_f(Nx, Ny, Δy)],
                [Dx_f(Nx, Ny, Δx)]
            ])
    Bd = bmat([
                [-1*Dy_b(Nx, Ny, Δy), Dx_b(Nx, Ny, Δx)]
            ])
    Bd2 = bmat([
                [Dx_b(Nx, Ny, Δx)],
                [Dy_b(Nx, Ny, Δy)]
            ])
    Fd2 = bmat([
                [Dx_f(Nx, Ny, Δx), Dy_f(Nx, Ny, Δy)]
            ])
    
    return csr_matrix((ω**2 * sc.mu_0 * T) + (T @ Fd1 @ inv(tez) @ Bd) + (Bd2 @ Fd2))