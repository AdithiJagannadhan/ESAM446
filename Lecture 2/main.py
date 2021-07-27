#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:52:17 2020

@author: Adithi

This file uses spatial.py and field.py included. This is tested using hw2_test1.py and hw2_test2.py
"""

import numpy as np
import field
import spatial

x = np.linspace(0, 1, 100, endpoint=False)
y = 2*np.pi*(x + 0.1*np.sin(2*np.pi*x))
grid = field.PeriodicGrid(y, 2*np.pi)

length = 0.1
f = field.Field(grid, length**2/(np.cos(y) + 1 + length)**2 )

d = spatial.FiniteDifference(1, order, grid)

df = d.operate(f)
df0 = f.data*2*np.sin(y)/(np.cos(y) + 1 + length)

df_rms = np.sqrt(np.mean(df0**2))

error = np.max(np.abs(df.data - df0))/df_rms
error_est = error_bound_FD[order]

assert error < error_est
