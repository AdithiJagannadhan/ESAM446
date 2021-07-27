
import pytest
import numpy as np
import field
import spatial

order_range = [2, 4, 6, 8]

error_bound_1 = [0.003487359345263997, 7.2859015893244896e-06, 1.626790011920982e-08, 3.753470375790787e-11]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_FD(order_index):
    order = order_range[order_index]
    x = np.linspace(0, 1, 100, endpoint=False)
    y = 2*np.pi*(x + 0.1*np.sin(2*np.pi*x))
    grid = field.PeriodicGrid(y, 2*np.pi)

    f = field.Field(grid, np.sin(y))

    d = spatial.FiniteDifference(1, order, grid)

    df = d.operate(f)
    df0 = np.cos(y)

    error = np.max(np.abs(df.data - df0))
    error_est = error_bound_1[order_index]

    assert error < error_est

error_bound_2 = [0.0016531241262078115, 3.953150377056438e-06, 1.01879280197617e-08, 2.6460656709051168e-11]

order_range_odd = [1, 3, 5, 7]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_FD_2(order_index):
    order = order_range_odd[order_index]
    x = np.linspace(0, 1, 100, endpoint=False)
    y = 2*np.pi*(x + 0.1*np.sin(2*np.pi*x))
    grid = field.PeriodicGrid(y, 2*np.pi)

    f = field.Field(grid, np.sin(y))

    d = spatial.FiniteDifference(2, order, grid)

    df = d.operate(f)
    df0 = -np.sin(y)

    error = np.max(np.abs(df.data - df0))
    error_est = error_bound_2[order_index]

    assert error < error_est

error_bound_3 = [0.005224671143882285, 1.2706445107820117e-05, 3.067306559578247e-08, 7.409826640410579e-11]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_FD_3(order_index):
    order = order_range[order_index]
    x = np.linspace(0, 1, 100, endpoint=False)
    y = 2*np.pi*(x + 0.1*np.sin(2*np.pi*x))
    grid = field.PeriodicGrid(y, 2*np.pi)

    f = field.Field(grid, np.sin(y))

    d = spatial.FiniteDifference(3, order, grid)

    df = d.operate(f)
    df0 = -np.cos(y)

    error = np.max(np.abs(df.data - df0))
    error_est = error_bound_3[order_index]

    assert error < error_est

stencil_size_range = [3, 5]
error_list = [2.078060608725386e-06, 3.125277886416338e-09]
convergence_list = [4, 6]

@pytest.mark.parametrize('stencil_index', np.arange(2))
def test_CFD_error_estimate(stencil_index):
    func_stencil = stencil_size_range[stencil_index]
    grid = field.UniformPeriodicGrid(50, 2*np.pi)

    d = spatial.CompactFiniteDifferenceUniformGrid(1, 3, func_stencil, grid)

    error_est = d.error_estimate(1)

    error_ref = error_list[stencil_index]
    fraction_diff = np.abs(error_est - error_ref)/error_ref

    assert fraction_diff < 0.8

@pytest.mark.parametrize('stencil_index', np.arange(2))
def test_CFD_convergence_order(stencil_index):
    func_stencil = stencil_size_range[stencil_index]
    grid = field.UniformPeriodicGrid(50, 2*np.pi)

    d = spatial.CompactFiniteDifferenceUniformGrid(1, 3, func_stencil, grid)

    assert convergence_list[stencil_index] == d.convergence_order