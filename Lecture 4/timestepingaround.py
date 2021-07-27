#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:20:59 2020

@author: Adithi
"""

import pytest
import numpy as np
import field
import spatial
import timesteppers

grid = field.UniformPeriodicGrid(100, 2*np.pi)
x = grid.values
IC = np.exp(-(x-np.pi)**2*8)
u = field.Field(grid, IC)

target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')
    
ts = timesteppers.BackwardDifferentiationFormula(u, d, 2)

alpha = 0.1
num_periods = 1.8
ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

error = np.max(np.abs( u.data - target))
error_est = 0.5

assert error < error_est