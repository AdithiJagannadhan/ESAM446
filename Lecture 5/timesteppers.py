
import numpy as np
from field import Field, FieldSystem, Identity, Average3
from scipy.special import factorial
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from collections import deque


class IMEXTimestepper:

    def __init__(self, eq_set): # remove n
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F_ops = eq_set.F_ops
        X_rhs = []
        for field in self.X.field_list:
            X_rhs.append(Field(field.grid))
        self.F = FieldSystem(X_rhs)
        self.X_rhs = X_rhs
        self.dt = None

    def evolve(self, time, dt):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self._step(dt)
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

        for i, op in enumerate(self.F_ops):
            op.evaluate(out=self.X_rhs[i])
        RHS = self.M @ self.X.data + dt*self.F.data
        np.copyto(self.X.data, self.LU.solve(RHS))


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            for i, op in enumerate(self.F_ops):
                op.evaluate(out=self.X_rhs[i])
            self.F_old = np.copy(self.F.data)
            RHS = self.M @ self.X.data + dt*self.F.data
            np.copyto(self.X.data, LU.solve(RHS))
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
                self.dt = dt

            for i, op in enumerate(self.F_ops):
                op.evaluate(out=self.X_rhs[i])
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.F.data - 1/2*dt*self.F_old
            np.copyto(self.F_old, self.F.data)
            np.copyto(self.X.data, self.LU.solve(RHS))


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps): #fix
        super().__init__(eq_set)
        self.steps = steps
        self.dt = None
        self.prev_X = {}
        self.prev_X[0] = self.X.data.copy()
        self.prev_F = {}
        
    def _step(self,dt):
        if dt != self.dt:
            # Find consts, det size
            if self.iter < self.steps:
                self.size = self.iter + 1
            else:
                self.size = self.steps
                self.dt = dt
            self._a_consts(dt)
            self._b_consts(dt)
            # Create LHS matrix
            LHS = (self.M * self.a[0]) + self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        # Update F
        self._update_F()
        # Compute RHS
        RHS = 0
        for i in range(0, self.size):
            RHS += self.prev_F[i]*self.b[i] - self.a[i+1]*self.prev_X[i]
        #Perform operation to find new u
        np.copyto(self.X.data, self.LU.solve(RHS))
        # Update X
        self._update_X()
        
    def _update_X(self):
        new_X = {}
        new_X[0] = self.X.data.copy()
        # [u1, u0] -- > [u2, u1, u0]
        for i in range(0, self.size):
            new_X[i+1] = self.prev_X[i]
        self.prev_X = new_X
        
    def _update_F(self):
        for i, op in enumerate(self.F_ops):
            op.evaluate(out=self.X_rhs[i])
        f_new = self.F.data.copy()
        new_F = {}
        new_F[0] = f_new
        # [F1, F0] --> [F2, F1, F0]
        for i in range(0, self.size - 1):
            new_F[i+1] = self.prev_F[i]
        self.prev_F = new_F
    
    def _a_consts(self, dt):
        size = self.size + 1
        b = np.zeros(size)
        b[1] = 1
        S = np.zeros((size, size))
        S[0,:] = 1
        for i in range(1, size): # row
            for j in range(1, size): # column
                t = dt * j
                S[i,j] = ((-1*t)**i)/factorial(i)
        self.a = np.linalg.solve(S, b)
    
    def _b_consts(self, dt):
        b = np.zeros(self.size)
        b[0] = 1
        S = np.zeros((self.size,self.size))
        S[0,:] = 1
        for i in range(1, self.size):
            for j in range(0, self.size):
                t = dt * (j + 1)
                S[i,j] = ((-1*t)**i)/factorial(i)
        self.b = np.linalg.solve(S, b)

class Timestepper:

    def __init__(self, u, L_op):
        self.t = 0
        self.iter = 0
        self.u = u
        self.L_op = L_op
        self.dt = None

    def evolve(self, time, dt):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self._step(dt)
        self.t += dt
        self.iter += 1


class ForwardEuler(Timestepper):

    def __init__(self, u, F):
        self.t = 0
        self.iter = 0
        self.u = u
        self.RHS = Field(u.grid)
        self.F = F

    def _step(self, dt):
        self.F.evaluate(out=self.RHS)
        self.u.data += dt*self.RHS.data

        
class LaxFriedrichs(Timestepper):

    def __init__(self, u, F):
        self.t = 0
        self.iter = 0
        self.u = u
        self.RHS = Field(u.grid)
        self.F = F
        self.I = Average3(u)

    def _step(self, dt):
        self.F.evaluate(out=self.RHS)
        self.u.data = self.I.matrix @ self.u.data + dt*self.RHS.data


class LeapFrog(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        # u_{n-1}
        self.u_old = Field(u.grid, u.data)

    def _step(self, dt):
        if iter == 0:
            I = Identity(self.u.grid, self.L_op.pad) 
            RHS = I + dt*self.L_op
            RHS.operate(self.u, out=self.u)
        else:
            if dt != self.dt:
                self.RHS = 2*dt*self.L_op
                self.dt = dt
            u_temp = self.RHS.operate(self.u)
            u_temp.data += self.u_old.data
            self.u_old.data = self.u.data
            self.u.data = u_temp.data


class LaxWendorff(Timestepper):

    def __init__(self, u, d_op, d2_op):
        self.t = 0
        self.iter = 0
        self.u = u
        self.d_op = d_op
        self.d2_op = d2_op
        self.dt = None
        self.I = Identity(u.grid, d_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            self.RHS = self.I + dt*self.d_op + dt**2/2*self.d2_op
            self.dt = dt
        self.RHS.operate(self.u, out=self.u)


class MacCormack(Timestepper):

    def __init__(self, u, op_f, op_b):
        self.t = 0
        self.iter = 0
        self.u = u
        if op_f.pad != op_b.pad:
            raise ValueError("Forward and Backward operators must have the same padding")
        self.op_f = op_f
        self.op_b = op_b
        self.dt = None
        self.I = Identity(u.grid, op_f.pad)
        self.u1 = Field(u.grid, u.data)

    def _step(self, dt):
        if dt != self.dt:
            self.RHS1 = self.I + dt*self.op_f
            self.RHS2 = 0.5*(self.I + dt*self.op_b)
            self.dt = dt
        self.RHS1.operate(self.u, out=self.u1)
        self.u.data = 0.5*self.u.data + self.RHS2.operate(self.u1).data


class BackwardEuler(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        self.I = Identity(u.grid, L_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.I - dt*self.L_op
            self.LU = spla.splu(LHS.matrix.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        self.u.data = self.LU.solve(self.u.data)


class CrankNicolson(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        self.I = Identity(u.grid, L_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.I - dt/2*self.L_op
            self.RHS = self.I + dt/2*self.L_op
            self.LU = spla.splu(LHS.matrix.tocsc(), permc_spec='NATURAL')
            self.dt = dt
        self.RHS.operate(self.u, out=self.u)
        self.u.data = self.LU.solve(self.u.data)


