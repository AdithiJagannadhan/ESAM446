#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:52:41 2020

@author: Adithi
"""
import pytest
import numpy as np
import field
import spatial
import timesteppers
import equations
import matplotlib.pyplot as plt

grid = field.UniformPeriodicGrid(100, 2*np.pi)
x = grid.values

IC = np.exp(-(x-np.pi)**2*8)

u = field.Field(grid)
p = field.Field(grid)

du = spatial.FiniteDifferenceUniformGrid(1, 2, u)
dp = spatial.FiniteDifferenceUniformGrid(1, 2, p)

rho0 = 3
gamma_p0 = 1

soundwave_problem = equations.SoundWave(u, p, du, dp, rho0, gamma_p0)

u.data[:] = IC
ts = timesteppers.CNAB(soundwave_problem)
alpha = 0.2
dt = alpha*grid.dx

ts.evolve(np.pi, dt)

solution = np.loadtxt('u_c_%i.dat' %100)
error = np.max(np.abs(solution - u.data))

error_est = 0.2

#assert error < error_est

plt.plot(x, u.data)
plt.plot(x, solution)