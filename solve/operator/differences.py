"""
Purpose:        Implements the difference matrices on a given
                discretized Yee grid in 3d.  2d can be implemented
                by simply setting Nz = 1.  
Author:         David Heydari (Jan. 2021)
Assumptions:    Periodic boundary conditions and uniform discretization
                Δx, Δy, and Δz.
"""

from scipy.sparse import block_diag, hstack, identity, csr_matrix

## Forward differences
def Dx_f(Nx, Ny, Nz, Δx):
    A = -1*identity(Nx, format='csr')
    B = -1*hstack((A[:, -1:], A[:, :-1]), format='csr') # shift left by 1
    return (1 / Δx) * block_diag([A+B] * (Ny*Nz), format='csr')

def Dy_f(Nx, Ny, Nz, Δy):
    A = block_diag([-1 * identity(Nx)] * Ny, format='csr')
    B = -1*hstack((A[:, -Nx:], A[:, :-Nx]), format='csr') # shift left by Nx
    return (1 / Δy) * block_diag(([A+B] * (Nz)))

def Dz_f(Nx, Ny, Nz, Δz):
    if Nz == 1:
        return csr_matrix((Nx*Ny, Nx*Ny))
    A = block_diag([-1 * identity(Nx*Ny)] * Nz, format='csr')
    B = -1*hstack((A[:, -Nx*Ny:], A[:, :-Nx*Ny]), format='csr') # shift left by Nx*Ny
    return (1 / Δz) * (A+B)

## Backward differences
def Dx_b(Nx, Ny, Nz, Δx):
    return -1 * Dx_f(Nx, Ny, Nz, Δx).transpose()

def Dy_b(Nx, Ny, Nz, Δy):
    return -1 * Dy_f(Nx, Ny, Nz, Δy).transpose()

def Dz_b(Nx, Ny, Nz, Δz):
    return -1 * Dz_f(Nx, Ny, Nz, Δz).transpose()