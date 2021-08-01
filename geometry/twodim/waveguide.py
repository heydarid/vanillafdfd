"""
Purpose:        Instantiates a uniform dielectric in the shape
                of a 2d circle that possess all the properties 
                needed for the solver.  Isotropic, dispersionless 
                linear media only.
Author:         David Heydari (May 2021)
"""

# TODO: One day allow for anisotropic, dispersive materials.

import numpy as np
import scipy.constants as sc
from itertools import product

class Waveguide:
    def __init__(self, x0, y0, eps_clad_top, eps_clad_bot, 
                        eps_cor, wid, etch, thk, bot_thk):
        self.x0 = x0
        self.y0 = y0
        self.eps_clad_top = eps_clad_top*sc.epsilon_0
        self.eps_clad_bot = eps_clad_bot*sc.epsilon_0
        self.eps_cor = eps_cor*sc.epsilon_0
        self.wid = wid
        self.etch = etch
        self.thk = thk
        self.bot_thk = bot_thk
    def make(self, Nx, Ny):
        eps = np.full((Nx,Ny), self.eps_clad_top)
        self.mu = np.full((Nx,Ny), sc.mu_0)
        [self.X, self.Y] = np.meshgrid(np.linspace(-2*self.wid, 2*self.wid, Nx), \
                            np.linspace(-self.bot_thk, 2*self.thk, Ny))
        eps[self.Y < 0] = self.eps_clad_bot
        eps[(self.Y > 0) & (self.Y <= (self.thk - self.etch))] = self.eps_cor
        eps[(abs(self.X) <= self.wid) & (self.Y <= self.thk) & (self.Y >= 0)] = self.eps_cor
        self.eps = eps
    def smoothen(self, Nx, Ny):
        for i,j in product(range(Nx),range(Ny)):
            self.eps[i,j] = (self.eps[i,j] + self.eps[i,j-1])/2.
            # parallelize with roll