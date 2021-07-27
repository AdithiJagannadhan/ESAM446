
import numpy as np
from field import Field, Identity, Average3
from scipy.special import factorial
from collections import deque
import sympy as sp


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
        print(self.iter)


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
        self.t = 0 # initial time
        self.iter = 0 # 0 iterations so far
        self.u = u # grid, function
        if op_f.pad != op_b.pad: # L but forward and backwards differencing, sizes need to be the same
            raise ValueError("Forward and Backward operators must have the same padding")
        self.op_f = op_f  #forward
        self.op_b = op_b #backwards
        self.dt = None # no change in time yet
        self.I = Identity(u.grid, op_f.pad)
        self.u1 = Field(u.grid, u.data) # u1 = u

    def _step(self, dt):
        if dt != self.dt: # if not at the end of the evovle fcn
            self.RHS1 = self.I + dt*self.op_f # un + dt*du/dt
            self.RHS2 = 0.5*(self.I + dt*self.op_b) 
        self.RHS1.operate(self.u, out=self.u1) # applies (I+dt*L) dot u --> u.data changed
        self.u.data = 0.5*self.u.data + self.RHS2.operate(self.u1).data  
        
            

class Multistage(Timestepper):

    def __init__(self, u, L_op, stages, a, b):
        self.t = 0
        self.iter = 0
        self.u = u # field (.grid, .data)
        self.L_op = L_op # derivative matrix dx
        self.stages = stages
        self.a = a # matrix to multiply for ki values
        self.b = b # vector for linear comb of ki values
        self.dt = None
        self.k = {}
        for i in range(0,stages):
            self.k[i] = {}
    
    def _step(self, dt):
        if dt != self.dt:
            if self.stages > 2:
                k1 = self.L_op.operate(self.u)
                RHS2 = self.u.data + 0.5*dt*k1.data
                k2 = self.L_op.operate(Field(self.u.grid, RHS2))
                RHS3 = self.u.data + dt*(2*k2.data-k1.data)
                k3 = self.L_op.operate(Field(self.u.grid,RHS3))
                self.u.data += (1/6)*dt*(k1.data+4*k2.data+k3.data)
            else:
                k1 = self.L_op.operate(self.u)
                RHS2 = self.u.data + 0.5*dt*k1.data
                k2 = self.L_op.operate(Field(self.u.grid, RHS2))
                self.u.data += dt*k2.data

class AdamsBashforth(Timestepper):

    def __init__(self, u, L_op, steps, dt):
        self.t = 0
        self.iter = 0
        self.u = u
        self.L_op = L_op
        self.steps = steps
        self.dt = None
        self.timestep = dt
        self.I = Identity(u.grid, L_op.pad)
        self.consts = AdamsBashforth.find_constants(steps, dt) # gives the constants ('a_i' on the lecture notes)
        self.prev = {} # stores previous values of u.grid.data
        self.prev[0] = self.u.data
        
    def _step(self, dt):
        if dt != self.dt:
            ##################################################################################################
            # I think something goes wrong here because somewhere here the values start going negative. I tried following the lecture notes, but might have messed up
            if self.iter + 1 < self.steps:
                consts = AdamsBashforth.find_constants(self.iter + 1, dt) # calculates constant for self.iter order AB
                addition = self.u.data.copy() # previous u.data
                for i in range(0, self.iter+1):
                    arg = self.prev[(self.iter) - i].copy() # pulls u.data in decsending order
                    f = self.L_op.operate(Field(self.u.grid, arg)) # .operate
                    addition += consts[i] * f.data # multiplies the derived u.data (now f.data) by consts
                self.u.data = addition # updates u.data
                self.prev[self.iter + 1] = self.u.data.copy() # adds values to self.prev
            ##################################################################################################
            # The code below is basically the same as the 2nd elif statement but uses a constant size self.prev and self.consts
            else:
                addition = self.u.data.copy()
                for i in range(0, self.steps):
                    arg = self.prev[(self.steps - 1) - i].copy()
                    f = (self.L_op.operate(Field(self.u.grid, arg))).data
                    addition += self.consts[i] * f
                self.u.data = addition
                self.update_dict()
        return
        
    # Function returns values of vector of constants - pretty sure this is correct
    def find_constants(steps, timestep):
        u = sp.Symbol('u')
        consts = np.zeros(steps)
        if steps == 1:
            consts = np.array([1]) * timestep
        else:
            for i in range(1, steps + 1):
                num  = 1
                den = 1
                for j in range(1, steps + 1):
                    if j!= i:
                        num *= (u - j)
                        den *= (i - j)
                inside = sp.integrate(num/den)
                consts[i-1] = ((inside.subs(u, 1) - inside.subs(u, 0)).evalf())*timestep
        return consts
    
    # Function changes the values of self.prev
    def update_dict(self):
        new = {}
        for i in range(0, self.steps - 1):
            new[i] = self.prev[i+1].copy()
        new[self.steps-1] = self.u.data.copy()
        self.prev = new
