#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:10:30 2020

@author: Adithi
"""
import numpy as np
import field
import spatial
import timesteppers
import matplotlib.pyplot as plt

grid = field.UniformPeriodicGrid(5, 5)
x = grid.values
I = field.Identity(grid, (0,0))
IC = x*0+1

u = field.Field(grid, IC)
ts = timesteppers.AdamsBashforth(u, -1*I, 3, 0.1)

num_periods = 1
ts.evolve(num_periods, 0.1)
xnew = u.data
error = np.max(np.abs(u.data - IC))
plt.plot(x, IC)
plt.plot(x, u.data)