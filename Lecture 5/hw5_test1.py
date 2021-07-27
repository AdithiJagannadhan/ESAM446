
import pytest
import numpy as np
import field
import spatial
import timesteppers
import equations

resolution_list = [100, 200, 400]

error_BDF2 = {(100,2):0.15, (200,2):0.04, (400,2):0.008, (100,4):0.07, (200,4):0.007, (400,4):0.0012}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_BDF2(resolution, spatial_order):
    grid = field.UniformPeriodicGrid(resolution, 5)
    x = grid.values
    
    IC = 0*x
    for i, xx in enumerate(x):
        if xx > 1 and xx <= 2:
            IC[i] = (xx-1)
        elif xx > 2 and xx < 3:
            IC[i] = (3-xx)
    
    u = field.Field(grid)
    du = spatial.FiniteDifferenceUniformGrid(1, spatial_order, u)
    d2u = spatial.FiniteDifferenceUniformGrid(2, spatial_order, u)
    nu = 1e-2
    
    vburgers_problem = equations.ViscousBurgers(u, nu, du, d2u)

    u.data[:] = IC
    ts = timesteppers.BDFExtrapolate(vburgers_problem, 2)

    alpha = 0.5
    dt = alpha*grid.dx

    ts.evolve(3, dt)

    solution = np.loadtxt('u_burgers_%i.dat' %resolution)
    error = np.max(np.abs(solution - u.data))

    error_est = error_BDF2[(resolution,spatial_order)]

    assert error < error_est

error_BDF3 = {(100,2):0.15, (200,2):0.04, (400,2):0.008, (100,4):0.05, (200,4):0.004, (400,4):0.0004}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_BDF3(resolution, spatial_order):
    grid = field.UniformPeriodicGrid(resolution, 5)
    x = grid.values

    IC = 0*x
    for i, xx in enumerate(x):
        if xx > 1 and xx <= 2:
            IC[i] = (xx-1)
        elif xx > 2 and xx < 3:
            IC[i] = (3-xx)

    u = field.Field(grid)
    du = spatial.FiniteDifferenceUniformGrid(1, spatial_order, u)
    d2u = spatial.FiniteDifferenceUniformGrid(2, spatial_order, u)
    nu = 1e-2
   
    vburgers_problem = equations.ViscousBurgers(u, nu, du, d2u)

    u.data[:] = IC
    ts = timesteppers.BDFExtrapolate(vburgers_problem, 3)

    alpha = 0.5
    dt = alpha*grid.dx

    ts.evolve(3, dt)

    solution = np.loadtxt('u_burgers_%i.dat' %resolution)
    error = np.max(np.abs(solution - u.data))

    error_est = error_BDF3[(resolution,spatial_order)]

    assert error < error_est

error_wave = {(100,2):0.1, (200,2):0.03, (400,2):0.006, (100,4):0.006, (200,4):6e-4, (400,4):8e-5}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_wave(resolution, spatial_order):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)

    u = field.Field(grid)
    p = field.Field(grid)

    du = spatial.FiniteDifferenceUniformGrid(1, spatial_order, u)
    dp = spatial.FiniteDifferenceUniformGrid(1, spatial_order, p)

    rho0 = 3
    gamma_p0 = 1

    soundwave_problem = equations.SoundWave(u, p, du, dp, rho0, gamma_p0)

    u.data[:] = IC
    ts = timesteppers.CNAB(soundwave_problem)
    alpha = 0.2
    dt = alpha*grid.dx

    ts.evolve(np.pi, dt)

    solution = np.loadtxt('u_c_%i.dat' %resolution)
    error = np.max(np.abs(solution - u.data))

    error_est = error_wave[(resolution,spatial_order)]

    assert error < error_est

