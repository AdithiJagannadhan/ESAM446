
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg as spla
from scipy.special import factorial
import matplotlib.pyplot as plt
import math
from field import Field

class SpatialDerivative:

    def __init__(self, grid, stencil_type):
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrices(grid)

    def error_estimate(self, lengthscale):
        pass

    def operate(self, field):
        pass

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
        plt.savefig(output)
        plt.clf()


class FiniteDifferenceUniformGrid(SpatialDerivative):

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):
        if stencil_type == 'centered' and convergence_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")
        if stencil_type == 'forward' or stencil_type == 'backward':
            if derivative_order % 2 == 0:
                raise ValueError("Forward and backward finite difference only for odd derivative order.")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        super().__init__(grid, stencil_type)

    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order

        if stencil_type == 'centered':
            # cancellation if derivative order is even
            dof = dof - (1 - dof % 2)
            j = np.arange(dof) - dof//2
        if stencil_type == 'forward':
            j = np.arange(dof) - dof//2 + 1
        if stencil_type == 'backward':
            j = np.arange(dof) - dof//2
        if stencil_type == 'full forward' or stencil_type == 'full backward':
            raise NotImplementedError()

        self.dof = dof
        self.pad = (-np.min(j), np.max(j))
        self.j = j

    def _make_stencil(self, grid):

        # assume constant grid spacing
        self.dx = grid.dx
        i = np.arange(self.dof)[:, None]
        j = self.j[None, :]
        S = 1/factorial(i)*(j*self.dx)**i

        b = np.zeros( self.dof )
        b[self.derivative_order] = 1.

        self.stencil = np.linalg.solve(S, b)

    def _build_matrices(self, grid):
        shape = [grid.N + self.dof - 1] * 2
        matrix = sparse.diags(self.stencil, self.j, shape=shape)

        self.matrix = matrix

    def operate(self, field):
        b = field.data_padded(self.pad)
        x = self.matrix @ b
        start_pad, end_pad = self.pad
        if end_pad == 0:
            s = slice(start_pad, None)
        else:
            s = slice(start_pad, -end_pad)
        return Field(field.grid, x[s])

    def error_estimate(self, lengthscale):
        error_degree = self.dof
        if self.stencil_type == 'centered' and self.derivative_order % 2 == 0:
            error_degree += 1
        error = np.abs(np.sum( self.stencil*(self.j*self.dx/lengthscale)**error_degree ))
        error *= 1/math.factorial(error_degree)
        return error

    def plot_matrix(self):
        self._plot_2D(self.matrix.A)

    def fourier_representation(self):
        kh = np.linspace(-np.pi, np.pi, 100)
        derivative = np.sum(self.stencil[:,None]*np.exp(1j*kh[None,:]*self.j[:,None]),axis=0)*self.dx**self.derivative_order
        return kh, derivative


class FiniteDifference(SpatialDerivative):

    def __init__(self, derivative_order, convergence_order, grid):
        # only allow centered stencil
        self.stencil_type = 'centered'
        if (derivative_order + convergence_order) % 2 == 0:
            raise ValueError("Centered finite difference must have odd derivative plus convergence order")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.N = grid.N
        super().__init__(grid, self.stencil_type)
        
    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order

        if stencil_type == 'centered':
            # cancellation if derivative order is even
            dof = dof - (1 - dof % 2)
        else:
            raise NotImplementedError()

        self.dof = dof
        self.half = dof//2
        self.pad = (self.half, self.half)
        
        ind = np.zeros((self.N + self.dof - 1, dof), dtype = int)
        for i in range(0, self.dof):
            for j in range(0, self.N + self.dof - 1):
                if j == 0 :
                    ind[j, i] = (-1 * self.dof) + 1 + i
                else:
                    ind[j, i] = ind[j-1, i] + 1
        for i in range(0, self.dof): #columns
            for j in range(0, self.N + self.dof - 1): #rows
                if ind[j,i] < 0:
                    ind[j,i] = self.N + ind[j,i]
                if ind[j,i] > self.N - 1:
                    ind[j,i] = ind[j,i] - self.N
        
        self.j = ind
        
    def _make_stencil(self, grid):
        a = np.zeros((self.N + self.dof - 1, self.dof))
        b = np.zeros(self.dof)
        b[self.derivative_order] = 1
        c = np.arange(self.dof)[:, None]
        h = np.zeros(self.dof)
        half = self.half
        
        for i in range(0, self.N + self.dof - 1):
            sumleft = 0
            sumright = 0
            for k in range(1, half + 1):
                if self.j[i,self.half+k] == 0:
                    sumleft += grid.values[self.j[i,half-k]] - grid.values[self.j[i,half-(k-1)]]
                    sumright +=  grid.length - grid.values[-1]
                elif self.j[i,self.half-k] == 0 or self.j[i,self.half-k] == self.N-1:
                    sumleft += -1 * (grid.length - grid.values[-1])
                    sumright += grid.values[self.j[i,half+k]] - grid.values[self.j[i,half+(k-1)]]
                else:
                    sumleft += grid.values[self.j[i,half-k]] - grid.values[self.j[i,half-(k-1)]]
                    sumright += grid.values[self.j[i,half+k]] - grid.values[self.j[i,half+(k-1)]]
                h[half-k] = sumleft
                h[half+k] = sumright
            S = (1/factorial(c)) * h**c
            a[i, :] = np.linalg.solve(S,b)

        self.stencil = a
    
    def _build_matrices(self, grid):
        size = self.N + self.dof - 1
        matrix = np.zeros((size, size))
        ref = self.j.copy()
        
        for i in range(0, size): #rows
            for j in range(0, self.dof):#cols
                ref[i,j] += self.half
                if i < size/2:
                    if ref[i,j] >= self.N:
                        ref[i,j] -= self.N
                    elif ref[i,j] >= self.N - 10:
                        count = 0
                        for k in range(1, self.dof):
                            if ref[i,j+k] != 0:
                                count += 1
                            elif ref[i,j+k] == 0:
                                count += 1
                                break
                        ref[i,j] = size - count + self.half
                elif i > size/2:
                    if ref[i,j] < self.N/4:
                        ref[i,j] += self.N
                    if ref[i,j] >= size:
                        ref[i,j] -= size
                matrix[i, ref[i,j]] = self.stencil[i,j]
                
        self.matrix = matrix

    def operate(self, field):
        b = field.data_padded(self.pad)
        x = self.matrix @ b
        start_pad, end_pad = self.pad
        if end_pad == 0:
            s = slice(start_pad, None)
        else:
            s = slice(start_pad, -end_pad)
        return Field(field.grid, x[s])

class CompactFiniteDifferenceUniformGrid(SpatialDerivative):

    def __init__(self, derivative_order, derivative_stencil_size, function_stencil_size, grid):
        # only allow centered CFD
        self.stencil_type = 'centered'
        if derivative_stencil_size % 2 == 0 or function_stencil_size % 2 == 0:
            raise ValueError("Centered finite difference has odd stencil sizes")

        self.deriv_order = derivative_order
        self.deriv_size = derivative_stencil_size
        self.func_size = function_stencil_size
        super().__init__(grid, self.stencil_type)
        
    def _stencil_shape(self, stencil_type):
        if stencil_type == 'centered':
            self.tot = self.deriv_size + self.func_size
            self.a_j = np.arange(self.func_size) - self.func_size//2
            self.c_j = np.arange(self.deriv_size) - self.deriv_size//2
            self.convergence_order = self.tot - 2
        else:
            raise NotImplementedError()
    
    def _ref(self, grid):
        A = np.zeros((grid.N, self.func_size), dtype = int)
        C = np.zeros((grid.N, self.deriv_size), dtype = int)
        
        for i in range(0, grid.N): #rows
            if i == 0:
                A[i,0] = -1 * self.func_size//2 + 1
                C[i,0] = -1 * self.deriv_size//2 + 1
            else:
                A[i,0] = A[i-1,0] + 1
                C[i,0] = C[i-1,0] + 1
        for j in range(1, self.func_size):
            A[:,j] = A[:,j-1] + 1
        for j in range(1, self.deriv_size):
            C[:,j] = C[:,j-1] + 1
        for i in range(0, grid.N):
            for j in range(0, self.func_size):
                if A[i,j] < 0:
                    A[i,j] += grid.N
                elif A[i,j] >= grid.N:
                    A[i,j] -= grid.N
            for j in range(0, self.deriv_size):
                if C[i,j] < 0:
                    C[i,j] += grid.N
                elif C[i,j] >= grid.N:
                    C[i,j] -= grid.N
        return A, C
    
    def _make_stencil(self, grid):
        self.dx = grid.dx
        h = self.dx
        b = np.zeros((self.tot -1,1))
        b[self.deriv_order] = 1
        stock = np.zeros((self.tot - 1, self.tot))
        for i in range(0, self.tot - 1): # rows
            for j in range(0, self.func_size):  # func columns
                stock[i,j] = ((h*self.a_j[j])**i)/factorial(i)
            for k in range(0, self.deriv_size): # deriv columns
                if i >= self.deriv_order:
                    ind = i - self.deriv_order
                    stock[i,self.func_size+k] = ((h*self.c_j[k])**ind)/factorial(ind)
        S = np.zeros((self.tot - 1, self.tot - 1))
        for i in range(0, self.tot - 1): # rows
            for j in range(0, self.tot - 1):  # columns
                if j <= self.func_size -1 + self.deriv_size//2:
                    S[i,j] = stock[i,j]
                elif j > self.func_size -1 + self.deriv_size//2:
                    S[i,j] = stock[i,j+1]
        
        a_then_c_part = np.linalg.solve(S,b)
        a_then_c = np.zeros(self.tot)
        a_then_c[self.func_size + self.deriv_size//2] = 1
        for i in range(0, self.tot-1):
            if i <= self.func_size - 1 + self.deriv_size//2:
                a_then_c[i] = a_then_c_part[i]
            else:
                a_then_c[i+1] = a_then_c_part[i]
        
        self.stencil = a_then_c.reshape(1,self.tot)
        
    
    def _build_matrices(self, grid):
        a_ref, c_ref = self._ref(grid)
        A = np.zeros((grid.N, grid.N))
        C = np.zeros((grid.N, grid.N))
        
        for i in range(0, grid.N): #rows
            for j in range(0, self.func_size):#cols
                A[i, a_ref[i,j]] = self.stencil[0,j]
            for k in range(0, self.deriv_size):#cols
                C[i, c_ref[i,k]] = self.stencil[0,k+self.func_size]
        
        matrix = np.linalg.inv(C).dot(A)
        self.matrix = matrix
    
    def operate(self, field):
        b = field.data
        x = self.matrix @ b
        return Field(field.grid, x)
    
    def error_estimate(self, lengthscale):
        error_deg = self.tot
        consts = self.stencil[0,:]
        a = self.a_j
        c = self.c_j
        v = np.zeros(self.tot)
        d = self.deriv_order
        h = self.dx
        error = 0
        
        for i in range(0, self.tot):
            if i < self.func_size:
                v[i] = ((a[i]*h/lengthscale)**error_deg)/factorial(error_deg)
            else:
                v[i] = ((c[i-self.func_size]*h/lengthscale)**(error_deg-d))/factorial(error_deg-d)
            error += 2.44* np.abs(v[i] * consts[i])
    
        return error
