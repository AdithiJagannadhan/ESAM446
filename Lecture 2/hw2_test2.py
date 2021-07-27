
import pytest
import numpy as np
import field
import spatial

stencil_size_range1 = [3, 5]

error_bound_1 = {(3,3): 4.156121217450772e-06, (3,5): 8.224415490569021e-09,(5,3): 6.250555772832676e-09, (5,5): 5.483601134649265e-12}

@pytest.mark.parametrize('deriv_stencil', stencil_size_range1)
@pytest.mark.parametrize('func_stencil', stencil_size_range1)
def test_CFD(deriv_stencil, func_stencil):
    grid = field.UniformPeriodicGrid(50, 2*np.pi)
    x = grid.values
    f = field.Field(grid, np.sin(x))

    d = spatial.CompactFiniteDifferenceUniformGrid(1, deriv_stencil, func_stencil, grid)

    df = d.operate(f)
    df0 = np.cos(x)

    error = np.max(np.abs(df.data - df0))
    error_est = error_bound_1[(deriv_stencil, func_stencil)]

    assert error < error_est

error_bound_bump = {(3,3): 0.16321049855215006, (3,5): 0.008074301026638045,(5,3): 0.006136468780244966, (5,5): 0.00013458782062368056}

@pytest.mark.parametrize('deriv_stencil', stencil_size_range1)
@pytest.mark.parametrize('func_stencil', stencil_size_range1)
def test_CFD_bump(deriv_stencil, func_stencil):
    grid = field.UniformPeriodicGrid(100, 1)
    x = grid.values

    length = 0.1
    f = field.Field(grid, length**2/(np.cos(2*np.pi*x) + 1 + length)**2 )

    d = spatial.CompactFiniteDifferenceUniformGrid(1, deriv_stencil, func_stencil, grid)

    df = d.operate(f)
    df0 = f.data*4*np.pi*np.sin(2*np.pi*x)/(np.cos(2*np.pi*x) + 1 + length)

    df_rms = np.sqrt(np.mean(df0**2))

    error = np.max(np.abs(df.data - df0))/df_rms
    error_est = error_bound_bump[(deriv_stencil, func_stencil)]

    assert error < error_est

error_bound_2 = {(3,3): 2.493672730470464e-06, (3,5): 4.994000746335218e-09, (5,3): 3.2673359721623838e-09, (5,5): 3.306904501047863e-12}

@pytest.mark.parametrize('deriv_stencil', stencil_size_range1)
@pytest.mark.parametrize('func_stencil', stencil_size_range1)
def test_CFD_2(deriv_stencil, func_stencil):
    grid = field.UniformPeriodicGrid(50, 2*np.pi)
    x = grid.values
    f = field.Field(grid, np.sin(x))

    d = spatial.CompactFiniteDifferenceUniformGrid(2, deriv_stencil, func_stencil, grid)

    df = d.operate(f)
    df0 = -np.sin(x)

    error = np.max(np.abs(df.data - df0))
    error_est = error_bound_2[(deriv_stencil, func_stencil)]

    assert error < error_est

stencil_size_range2 = [5, 7]

error_bound_3 = {(5,5): 5.316313603681697e-09, (5,7): 2.450139197442897e-10, (7,5): 2.4485958998883425e-10, (7,7): 5.2102605561043837e-11}

@pytest.mark.parametrize('deriv_stencil', stencil_size_range2)
@pytest.mark.parametrize('func_stencil', stencil_size_range2)
def test_CFD_3(deriv_stencil, func_stencil):
    grid = field.UniformPeriodicGrid(30, 2*np.pi)
    x = grid.values
    f = field.Field(grid, np.sin(x))

    d = spatial.CompactFiniteDifferenceUniformGrid(3, deriv_stencil, func_stencil, grid)

    df = d.operate(f)
    df0 = -np.cos(x)

    error = np.max(np.abs(df.data - df0))
    error_est = error_bound_3[(deriv_stencil, func_stencil)]

    assert error < error_est

error_bound_FD = {2: 0.11247269000472183, 4: 0.0023808604320459457, 6: 5.386198919230259e-05, 8: 1.2591693341030753e-06}

order_range = [2, 4, 6, 8]

@pytest.mark.parametrize('order', order_range)
def test_FD_bump(order):
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

