
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import numbers

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class PeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Field:

    def __init__(self, grid, data):
        self.grid = grid
        self.data = np.copy(data)

    def data_padded(self, pad, bc='periodic'):
        start_pad, end_pad = pad
        new_data = np.zeros(start_pad + self.grid.N + end_pad)
        if end_pad == 0:
            s = slice(start_pad, None)
        else:
            s = slice(start_pad, -end_pad)
        new_data[s] = self.data
        if bc == 'periodic':
            if start_pad > 0:
                new_data[:start_pad] = self.data[-start_pad:]
            if end_pad > 0:
                new_data[-end_pad:] = self.data[:end_pad]
        else:
            raise ValueError("Only supports periodic BC")
        return new_data


class LinearOperator:

    def __init__(self, grid):
        self.grid = grid

    def operate(self, field, out=None):
        b = field.data_padded(self.pad)
        x = self.padded_matrix @ b
        start_pad, end_pad = self.pad
        if end_pad == 0:
            s = slice(start_pad, None)
        else:
            s = slice(start_pad, -end_pad)
        if out == None:
            return Field(field.grid, x[s])
        else:
            out.data = x[s]

    def _unpadded_matrix(self, grid):
        matrix = sparse.diags(self.stencil, self.j, shape=[grid.N]*2)
        matrix = matrix.tocsr()
        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                matrix[i,-jmin+i:] = self.stencil[:jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i,:i+1] = self.stencil[-i-1:]
        return matrix

    @staticmethod
    def _plot_2D(matrix, title='FD matrix', output='matrix.pdf'):
        lim_margin = -0.05
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot()
        I, J = matrix.shape
        matrix_mag = np.log10(np.abs(matrix))
        ax.pcolor(matrix_mag[::-1])
        ax.set_xlim(-lim_margin, I+lim_margin)
        ax.set_ylim(-lim_margin, J+lim_margin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        plt.title(title)
        plt.tight_layout()

    def __add__(self, other):
        if self.grid != other.grid:
            raise ValueError("Grids must be the same to sum operators")
        if self.pad != other.pad:
            raise ValueError("Pad must be the same to sum operators")
        op = LinearOperator(self.grid)
        op.pad = self.pad
        op.matrix = self.matrix + other.matrix
        op.padded_matrix = self.padded_matrix + other.padded_matrix
        return op

    def __neg__(self):
        op = LinearOperator(self.grid)
        op.pad = self.pad
        op.matrix = -self.matrix
        op.padded_matrix = -self.padded_matrix
        return op

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            op = LinearOperator(self.grid)
            op.pad = self.pad
            op.matrix = other*self.matrix
            op.padded_matrix = other*self.padded_matrix
            return op
        else:
            raise ValueError("Can only multiply linear operators by numbers")

    def __rmul__(self, other):
        return self*other


class Identity(LinearOperator):

    def __init__(self, grid, pad):
        self.pad = pad
        self._stencil_shape()
        self._make_stencil()
        self._build_matrices(grid)
        super().__init__(grid)

    def _stencil_shape(self):
        self.j = np.arange(1)

    def _make_stencil(self):
        self.stencil = np.array([1])

    def _build_matrices(self, grid):
        shape = [grid.N + self.pad[0] + self.pad[1]] * 2
        self.padded_matrix = sparse.diags(self.stencil, self.j, shape=shape)
        self.matrix = sparse.diags(self.stencil, self.j, shape=[grid.N]*2)


class Average3(LinearOperator):

    def __init__(self, grid):
        self.pad = (1, 1)
        self._stencil_shape()
        self._make_stencil()
        self._build_matrices(grid)
        super().__init__(grid)

    def _stencil_shape(self):
        self.j = np.arange(3) - 1

    def _make_stencil(self):
        self.stencil = np.array([1/2, 0, 1/2])

    def _build_matrices(self, grid):
        shape = [grid.N + self.pad[0] + self.pad[1]] * 2
        self.padded_matrix = sparse.diags(self.stencil, self.j, shape=shape)
        self.matrix = self._unpadded_matrix(grid)

