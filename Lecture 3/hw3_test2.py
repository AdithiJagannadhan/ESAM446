import pytest
import numpy as np

import field
import spatial
import timesteppers

# import field
# import spatial
# import timesteppers

# I put my .python files in folders corresponding to the week, so
# don't forget to change that back to whatever you have it to

# CHANGES FROM ORIGINAL: Added tests for Adams Bashforth, particularly for
# 3-4, 4-2, and 4-4 schemes. I mostly stole testing values from the RK methods
# but one important change is to alpha. Alpha values larger than 0.4 generally won't
# work for the AB methods, but will be pretty accurate otherwise. I picked alpha
# restrictive in each case so if that doesn't work just reduce by like 0.05 or 0.1

resolution_list = [100, 200, 400]

error_RK_2_2 = {100: 0.5, 200: 0.15, 400: 0.05}


@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_2_2(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    u = field.Field(grid, IC)

    target = np.exp(-(x - np.pi - 2 * np.pi * 0.2) ** 2 * 8)

    d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')

    stages = 2
    a = np.array([[0, 0],
                  [1 / 2, 0]])
    b = np.array([0, 1])

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(2 * np.pi * num_periods, alpha * grid.dx)

    error = np.max(np.abs(u.data - target))
    error_est = error_RK_2_2[resolution]

    assert error < error_est


error_RK_2_4 = {100: 0.15, 200: 0.05, 400: 0.01}


@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_2_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    u = field.Field(grid, IC)

    target = np.exp(-(x - np.pi + 2 * np.pi * 0.2) ** 2 * 8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')

    stages = 2
    a = np.array([[0, 0],
                  [1 / 2, 0]])
    b = np.array([0, 1])

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(2 * np.pi * num_periods, alpha * grid.dx)

    error = np.max(np.abs(u.data - target))
    error_est = error_RK_2_4[resolution]

    assert error < error_est


error_RK_3_2 = {100: 0.5, 200: 0.2, 400: 0.05}


@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_3_2(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    u = field.Field(grid, IC)

    target = np.exp(-(x - np.pi - 2 * np.pi * 0.2) ** 2 * 8)

    d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')

    stages = 3
    a = np.array([[0, 0, 0],
                  [1 / 2, 0, 0],
                  [-1, 2, 0]])
    b = np.array([1, 4, 1]) / 6

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(2 * np.pi * num_periods, alpha * grid.dx)

    error = np.max(np.abs(u.data - target))
    error_est = error_RK_3_2[resolution]

    assert error < error_est


error_RK_3_4 = {100: 0.04, 200: 0.005, 400: 3e-4}


@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_3_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    u = field.Field(grid, IC)

    target = np.exp(-(x - np.pi + 2 * np.pi * 0.2) ** 2 * 8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')

    stages = 3
    a = np.array([[0, 0, 0],
                  [1 / 2, 0, 0],
                  [-1, 2, 0]])
    b = np.array([1, 4, 1]) / 6

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(2 * np.pi * num_periods, alpha * grid.dx)

    error = np.max(np.abs(u.data - target))
    error_est = error_RK_3_4[resolution]

    assert error < error_est


error_RK_4_2 = {100: 0.5, 200: 0.2, 400: 0.05}


@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_4_2(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    u = field.Field(grid, IC)

    target = np.exp(-(x - np.pi - 2 * np.pi * 0.2) ** 2 * 8)

    d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')

    stages = 4
    a = np.array([[0, 0, 0, 0],
                  [1 / 2, 0, 0, 0],
                  [0, 1 / 2, 0, 0],
                  [0, 0, 1, 0]])
    b = np.array([1, 2, 2, 1]) / 6

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(2 * np.pi * num_periods, alpha * grid.dx)

    error = np.max(np.abs(u.data - target))
    error_est = error_RK_4_2[resolution]

    assert error < error_est


error_RK_4_4 = {100: 0.04, 200: 0.003, 400: 2e-4}


@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_4_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    u = field.Field(grid, IC)

    target = np.exp(-(x - np.pi + 2 * np.pi * 0.2) ** 2 * 8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')

    stages = 4
    a = np.array([[0, 0, 0, 0],
                  [1 / 2, 0, 0, 0],
                  [0, 1 / 2, 0, 0],
                  [0, 0, 1, 0]])
    b = np.array([1, 2, 2, 1]) / 6

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(2 * np.pi * num_periods, alpha * grid.dx)

    error = np.max(np.abs(u.data - target))
    error_est = error_RK_4_4[resolution]

    assert error < error_est

# Tests by Richard, don't be sad if they don't pass :(


error_AB_3_4 = {100:0.04, 200:0.005, 400:3e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_3_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')

    steps = 3
    alpha = 0.2
    num_periods = 1.2
    ts = timesteppers.AdamsBashforth(u, d, steps,alpha*grid.dx)
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_AB_3_4[resolution]
    print(error)
    assert error < error_est

error_AB_4_2 = {100: 0.5, 200: 0.2, 400: 0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_4_2(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    u = field.Field(grid, IC)

    target = np.exp(-(x - np.pi - 2 * np.pi * 0.2) ** 2 * 8)

    d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')

    steps = 4
    alpha = 0.4
    num_periods = 1.8
    ts = timesteppers.AdamsBashforth(u, d, steps, alpha * grid.dx)
    ts.evolve(2 * np.pi * num_periods, alpha * grid.dx)

    error = np.max(np.abs(u.data - target))
    error_est = error_AB_4_2[resolution]
    print(error)
    assert error < error_est


error_AB_4_4 = {100:0.04, 200:0.003, 400:2e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_4_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')

    steps = 4
    alpha = 0.25
    num_periods = 1.2
    ts = timesteppers.AdamsBashforth(u, d, steps,alpha*grid.dx )

    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_AB_4_4[resolution]
    print(error)
    assert error < error_est