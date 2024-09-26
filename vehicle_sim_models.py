from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy

from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_st import init_st
from vehiclemodels.init_mb import init_mb
from vehiclemodels.init_std import init_std
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std

from vehicle_dynamics_st_disturbance import vehicle_dynamics_st_disturbance

from casadi import *

class ks_car_sim:
    def __init__(self, sx0, sy0, steerAng0, v0, carAng0):
        self.p = parameters_vehicle2()
        dotCarAng0 = 0
        beta = 0
        initialState = [sx0, sy0, steerAng0, v0, carAng0, dotCarAng0, beta]
        x0 = init_ks(initialState)
        self.stateVector = x0
        
    def setInitialState(self, sx0, sy0, steerAng0, v0, carAng0):
        dotCarAng0 = 0
        beta = 0
        initialState = [sx0, sy0, steerAng0, v0, carAng0, dotCarAng0, beta]
        x0 = init_ks(initialState)
        self.stateVector = x0

    def simNext(self, useDisturbance, t_interval, t_resolution, vSteerAng, accel):
        u = [vSteerAng, accel]
        t = numpy.arange(0, t_interval, t_resolution)

        x_next = odeint(self.func_KS, self.stateVector, t, args=(u, self.p))
        print(x_next[-1])
        self.stateVector = x_next[-1]
        return self.stateVector
    
    def getSingleTrackStates(self):
        return self.stateVector[:5]
    
    def func_KS(self, x, t, u, p):
        f = vehicle_dynamics_ks(x, u, p)
        return f

class st_car_sim:
    def __init__(self, sx0, sy0, steerAng0, v0, carAng0):
        self.p = parameters_vehicle2()
        dotCarAng0 = 0
        beta = 0
        initialState = [sx0, sy0, steerAng0, v0, carAng0, dotCarAng0, beta]
        x0 = init_st(initialState)
        self.stateVector = x0
        self.np_rng = np.random.default_rng(seed=0)

    def setInitialState(self, sx0, sy0, steerAng0, v0, carAng0):
        dotCarAng0 = 0
        beta = 0
        initialState = [sx0, sy0, steerAng0, v0, carAng0, dotCarAng0, beta]
        x0 = init_st(initialState)
        self.stateVector = x0

    def setRNGseed(self, seed):
        self.np_rng = np.random.default_rng(seed=seed)

    def simNext(self, useDisturbance, t_interval, t_resolution, vSteerAng, accel):
        u = [vSteerAng, accel]
        t = numpy.arange(0, t_interval, t_resolution)

        dynamics = self.func_ST
        if useDisturbance == True:
            dynamics = self.func_ST_disturbance

        x_next = odeint(dynamics, self.stateVector, t, args=(u, self.p))
        print(x_next[-1])
        # stateLen = len(self.stateVector)
        noise = 0
        if useDisturbance == True:
            dsx = self.np_rng.normal(0, 0.05)
            dsy = self.np_rng.normal(0, 0.005)
            dsteerAng = self.np_rng.normal(0, 1e-4)
            dv = self.np_rng.normal(0, 1e-3)
            dcarAng = self.np_rng.normal(0, 5e-4)
            dvCarAng = self.np_rng.normal(0, 1e-4)
            dslipAng = self.np_rng.normal(0, 1e-4)
            noise = np.array([dsx, dsy, dsteerAng, dv, dcarAng, dvCarAng, dslipAng]) 
            # print("noise: ")
            # print(noise)
        # rawNextStateVector = x_next[-1]
        self.stateVector = x_next[-1] + noise
        return self.stateVector
    
    def getSingleTrackStates(self):
        return self.stateVector[:5]
    
    def func_ST(self, x, t, u, p):
        f = vehicle_dynamics_st(x, u, p)
        return f
    
    def func_ST_disturbance(self, x, t, u, p):
        f = vehicle_dynamics_st_disturbance(x, u, p)
        return f
    
class std_car_sim:
    def __init__(self, sx0, sy0, steerAng0, v0, carAng0):
        self.p = parameters_vehicle2()
        dotCarAng0 = 0
        beta = 0
        initialState = [sx0, sy0, steerAng0, v0, carAng0, dotCarAng0, beta]
        x0 = init_std(initialState, self.p)
        self.stateVector = x0

    def setInitialState(self, sx0, sy0, steerAng0, v0, carAng0):
        dotCarAng0 = 0
        beta = 0
        initialState = [sx0, sy0, steerAng0, v0, carAng0, dotCarAng0, beta]
        x0 = init_std(initialState, self.p)
        self.stateVector = x0

    def simNext(self, useDisturbance, t_interval, t_resolution, vSteerAng, accel):
        u = [vSteerAng, accel]
        t = numpy.arange(0, t_interval, t_resolution)

        # x_next = odeint(self.func_STD, self.stateVector, t, args=(u, self.p))
        x_next = solve_ivp(vehicle_dynamics_std, (0, t_interval), self.stateVector, method='Radau', args=(u, self.p))
        # ode(self.func_STD).set_integrator('zvode', method='bdf')
        # x_next.set_initial_value(self.stateVector, 0).set_f_params((u, self.p))
        
        print(x_next.message)
        print(x_next.y[:, -1])
        self.stateVector = x_next.y[:, -1]
        return self.stateVector
    
    def getSingleTrackStates(self):
        return self.stateVector[:5]
    
    def func_STD(self, t, x, u, p):
        f = vehicle_dynamics_std(t, x, u, p)
        return f


class mb_car_sim:
    def __init__(self, sx0, sy0, steerAng0, v0, carAng0):
        self.p = parameters_vehicle2()
        self.sx = sx0
        self.sy = sy0
        self.steerAng = steerAng0
        self.v = v0
        self.carAng = carAng0

        self.dotCarAng = 0
        self.beta = 0


        self.initialState = [self.sx, self.sy, self.steerAng, self.v, self.carAng, self.dotCarAng, self.beta]
        
        x0_MB = init_mb(self.initialState, self.p)  # initial state for multi-body model
        self.stateVector = x0_MB
        # print(x0_MB)

    def setInitialState(self, sx0, sy0, steerAng0, v0, carAng0):
        self.p = p = parameters_vehicle2()
        self.sx = sx0
        self.sy = sy0
        self.steerAng = steerAng0
        self.v = v0
        self.carAng = carAng0
        self.dotCarAng = 0
        self.beta = 0

        self.initialState = [self.sx, self.sy, self.steerAng, self.v, self.carAng, self.dotCarAng, self.beta]
        
        x0_MB = init_mb(self.initialState, self.p)  # initial state for multi-body model
        self.stateVector = x0_MB
        
    def simNext(self, useDisturbance, t_interval, t_resolution, vSteerAng, accel):
        u = [vSteerAng, accel]
        t = numpy.arange(0, t_interval, t_resolution)

        x_next = odeint(self.func_MB, self.stateVector, t, args=(u, self.p))
        print(x_next[-1])
        self.stateVector = x_next[-1]
        return self.stateVector
    
    def resetToInitialState(self):
        x0_MB = init_mb(self.initialState, self.p)  # initial state for multi-body model
        self.stateVector = x0_MB

    def getSingleTrackStates(self):
        return self.stateVector[:5]

    def func_MB(self, x, t, u, p):
        f = vehicle_dynamics_mb(x, u, p)
        return f