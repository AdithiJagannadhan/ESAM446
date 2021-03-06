
import pytest
import numpy as np
import field
import spatial
import timesteppers
import matplotlib.pyplot as plt

resolution_list = [100, 200, 400]

error_RK_2_2 = {100:0.5, 200:0.15, 400:0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_2_2(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')
    
    stages = 2
    a = np.array([[  0,   0],
                  [1/2,   0]])
    b = np.array([0, 1])
    
    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_RK_2_2[resolution]

    assert error < error_est

error_RK_2_4 = {100:0.15, 200:0.05, 400:0.01}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_2_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')
    
    stages = 2
    a = np.array([[  0,   0],
                  [1/2,   0]])
    b = np.array([0, 1])

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_RK_2_4[resolution]

    assert error < error_est

error_RK_3_2 = {100:0.5, 200:0.2, 400:0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_3_2(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')

    stages = 3
    a = np.array([[  0, 0, 0],
                  [1/2, 0, 0],
                  [ -1, 2, 0]])
    b = np.array([1, 4, 1])/6

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_RK_3_2[resolution]

    assert error < error_est

error_RK_3_4 = {100:0.04, 200:0.005, 400:3e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_3_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')

    stages = 3
    a = np.array([[  0, 0, 0],
                  [1/2, 0, 0],
                  [ -1, 2, 0]])
    b = np.array([1, 4, 1])/6

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_RK_3_4[resolution]

    assert error < error_est

error_RK_4_2 = {100:0.5, 200:0.2, 400:0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_4_2(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 2, grid, stencil_type='centered')
    
    stages = 4
    a = np.array([[  0,   0, 0, 0],
                  [1/2,   0, 0, 0],
                  [  0, 1/2, 0, 0],
                  [  0,   0, 1, 0]])
    b = np.array([1, 2, 2, 1])/6
    
    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.8
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_RK_4_2[resolution]

    assert error < error_est

error_RK_4_4 = {100:0.04, 200:0.003, 400:2e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_4_4(resolution):
    grid = field.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    u = field.Field(grid, IC)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')

    stages = 4
    a = np.array([[  0,   0, 0, 0],
                  [1/2,   0, 0, 0],
                  [  0, 1/2, 0, 0],
                  [  0,   0, 1, 0]])
    b = np.array([1, 2, 2, 1])/6

    ts = timesteppers.Multistage(u, d, stages, a, b)

    alpha = 0.5
    num_periods = 1.2
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)

    error = np.max(np.abs( u.data - target))
    error_est = error_RK_4_4[resolution]

    assert error < error_est

def test_AB():
    L = 0.1
    func = lambda x: np.exp(-(1+np.cos(x))**2/2/L**2)
    grid = field.UniformPeriodicGrid(100, 2*np.pi)
    x = grid.values
    IC = func(x)
    
    u = field.Field(grid, IC)
    d = spatial.FiniteDifferenceUniformGrid(1, 6, grid, stencil_type='centered')
    alpha = 10/100
    ts = timesteppers.AdamsBashforth(u, d, 5, alpha*grid.dx)
    num_periods = 10
    ts.evolve(2*np.pi*num_periods, alpha*grid.dx)
    error = np.max(np.abs( u.data - IC))
    error_est = 0.001
    assert False
    
    
L = 0.1
func = lambda x: np.exp(-(1+np.cos(x))**2/2/L**2)
grid = field.UniformPeriodicGrid(100, 2*np.pi)
x = grid.values
IC = func(x)
    
u = field.Field(grid, IC)
d = spatial.FiniteDifferenceUniformGrid(1, 6, grid, stencil_type='centered')
alpha = 10/100
ts = timesteppers.AdamsBashforth(u, d, 5, alpha*grid.dx)
num_periods = 10
ts.evolve(2*np.pi*num_periods, alpha*grid.dx)
error = np.max(np.abs( u.data - IC))
error_est = 0.001
plt.plot(x, IC)
plt.plot(x, u.data)



grid = field.UniformPeriodicGrid(100, 2*np.pi)
x = grid.values
IC = np.sin(x)

u = field.Field(grid, IC)
d = spatial.FiniteDifferenceUniformGrid(1, 4, grid, stencil_type='centered')
alpha = 10/100
ts = timesteppers.AdamsBashforth(u, d, 2, alpha*grid.dx)

num_periods = 10
ts.evolve(2*np.pi*num_periods, alpha*grid.dx)
print(grid.dx*alpha)
error = np.max(np.abs(u.data - IC))
plt.plot(x, IC)
plt.plot(x, u.data)