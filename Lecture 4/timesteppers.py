
import numpy as np
from field import Field, Identity, Average3
from scipy.special import factorial
import scipy.sparse.linalg as spla
from collections import deque

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

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        self.I = Identity(u.grid, L_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            self.RHS = self.I + dt*self.L_op
        self.RHS.operate(self.u, out=self.u)

        
class LaxFriedrichs(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        self.I = Average3(u.grid)

    def _step(self, dt):
        if dt != self.dt:
            self.RHS = self.I + dt*self.L_op
        self.RHS.operate(self.u, out=self.u)


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
        self.RHS1.operate(self.u, out=self.u1)
        self.u.data = 0.5*self.u.data + self.RHS2.operate(self.u1).data


class Multistage(Timestepper):

    def __init__(self, u, L_op, stages, a, b):
        super().__init__(u, L_op)
        self.I = Identity(u.grid, L_op.pad)
        self.stages = stages
        self.a = a
        self.b = b
    
    def _step(self, dt):
        if dt != self.dt:
            pass
        return


class AdamsBashforth(Timestepper):

    def __init__(self, u, L_op, steps, dt):
        super().__init__ (u, L_op)
        self.I = Identity(u.grid, L_op.pad)
        self.steps = steps
        self.dt = dt


class BackwardEuler(Timestepper):

    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        self.I = Identity(u.grid, L_op.pad)

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.I - dt*self.L_op
            self.LU = spla.splu(LHS.matrix.tocsc(), permc_spec='NATURAL')
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
        self.RHS.operate(self.u, out=self.u)
        self.u.data = self.LU.solve(self.u.data)


class BackwardDifferentiationFormula(Timestepper):

    def __init__(self, u, L_op, steps):
        super().__init__(u, L_op)
        self.u = u
        self.L_op = L_op
        self.steps = steps
        self.prev_u = {}
        self.prev_u[0] = self.u.data
        self.prev_ts = np.zeros(self.steps)
        self.I = Identity(u.grid, L_op.pad)
        self.switch = False
        self.diff_t = True
        self.dt = None
    
    def _step(self, dt):
        if dt != self.dt:
            if self.iter == self.steps:
                self.switch = True
            # Determine sizes of parameters to add on RHS
            if self.switch:
                size = self.steps
                # Determine if need to recalculate a and update ts
                if self.iter > self.steps:
                    self._need_to_upd(dt)
            else:
                size = self.iter + 1
            
            if self.diff_t:
                # Update time vector
                self._update_ts(size, dt)
                
                # Find constants
                self.a = self._find_consts(size)

                #Solve equation
                self.LHS = self.L_op - self.a[0]*self.I
                self.LU = spla.splu(self.LHS.matrix.tocsc(), permc_spec='NATURAL')
        RHS = 0
        for i in range(0, size):
            RHS += self.a[i+1] * self.prev_u[i]
        self.u.data = self.LU.solve(RHS)
            
        # Update u
        self._update_u(size)
        return
    
    def _need_to_upd(self, dt):
        for i in range(0, self.steps):
            if dt != self.prev_ts[i]:
                self.diff_t = True
            else:
                self.diff_t = False
    
    # Updates u dict
    def _update_u(self, n):
        new_u = {}
        new_u[0] = self.u.data
        size = n + 1
        for i in range(1, size):
            new_u[i] = self.prev_u[i-1]
        self.prev_u = new_u

    # Updates ts
    def _update_ts(self, size, dt): 
        new_ts = np.zeros(size)
        new_ts[0] = dt
        for i in range(1, size): # [t0] --> [t1, t0]
            new_ts[i] =  self.prev_ts[i-1]
        self.prev_ts = new_ts
    
    def _find_consts(self, n):
        size = n + 1
        b = np.zeros(size)
        b[1] = 1
        S = np.zeros((size, size))
        S[0,:] = 1
        for i in range(1, size): # row
            for j in range(1, size): # column
                t = sum(self.prev_ts[0:j])
                S[i,j] = ((-1*t)**i)/factorial(i)
        ans = np.linalg.solve(S, b)
        return ans

    