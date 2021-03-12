"""
Purpose:        Instantiates a uniform dielectric in the shape
                of a 2d circle that possess all the properties 
                needed for the solver.  Isotropic, dispersionless 
                linear media only.
Author:         David Heydari (Jan. 2021)
"""

# TODO: One day allow for anisotropic, dispersive materials.

import numpy as np
import scipy.constants as sc
from itertools import product

class Circle:
    def __init__(self, x0, y0, eps_back, eps_circ, radius):
        self.x0 = x0
        self.y0 = y0
        self.eps_back = eps_back*sc.epsilon_0
        self.eps_circ = eps_circ*sc.epsilon_0
        self.radius = radius
    def make(self, Nx, Ny):
        eps = np.full((Nx,Ny),self.eps_back)
        self.mu = np.full((Nx,Ny), sc.mu_0)
        [self.X,self.Y] = np.meshgrid(np.linspace(-1*self.radius,self.radius,Nx), \
                            np.linspace(-1*self.radius,self.radius,Ny))
        eps[(self.X**2 + self.Y**2) < self.radius**2] = self.eps_circ
        self.eps = eps
    def smoothen(self, Nx, Ny):
        for i,j in product(range(Nx),range(Ny)):
            self.eps[i,j] = (self.eps[i,j] + self.eps[i,j-1])/2.
            # parallelize with roll