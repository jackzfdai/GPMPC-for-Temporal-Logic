import numpy as np
from scipy.integrate import odeint
import math
from casadi import *

import gp_dynamics as gpdyn

np_rng = np.random.default_rng(seed=0)

class cstrModel:
    def __init__(self, paramD, paramB, paramGamma, paramBeta):
        self.paramD = paramD
        self.paramB = paramB
        self.paramGamma = paramGamma
        self.paramBeta = paramBeta
        self.x = SX.sym('x', 2) #[x1: reactant concentration, x2: reactor temperature]
        self.u = SX.sym('u', 1) #[u: cooling jacket temperature]
        self.x1lim = [-1, 2]
        self.x2lim = [-1, 2]
        self.ulim = [-1, 0]

        self.stateVarLim = [1e-8, 1]
        self.stateCovarLim = [-1, 1]

        self.gpReady = False
        self.Bd = 0

    def setLimits(self, x1lim, x2lim, ulim):
        self.x1lim = x1lim
        self.x2lim = x2lim
        self.ulim = ulim

    def setGPResiduals(self, trainingIts, trainingInputData, trainingOutputData):
        self.gp_x1 = gpdyn.gpResidual(2, 1)
        self.gp_x2 = gpdyn.gpResidual(2, 1)

        self.gp_x1.train(trainingIts, trainingInputData[0], trainingOutputData[0])
        self.gp_x2.train(trainingIts, trainingInputData[1], trainingOutputData[1])

        self.Bd = np.transpose(np.array([0, 1]))
        # self.Bd = np.eye(2)
        # self.Bd = np.diag([0, 1])
        self.gpReady = True

    def setGPResidualsFromFile(self, trainingIts, gp_x1_file = None, gp_x2_file = None):
        self.gp_x1 = gpdyn.gpResidual(2, 1)
        self.gp_x2 = gpdyn.gpResidual(2, 1)

        if gp_x1_file is not None:
            self.gp_x1.parametersFromFile(gp_x1_file)

        if gp_x2_file is not None:
            self.gp_x2.parametersFromFile(gp_x2_file)

        self.Bd = np.transpose(np.array([0, 1]))
        # self.Bd = np.eye(2)
        # self.Bd = np.diag([0, 1])
        self.gpReady = True

    def writeGPResidualsToFiles(self, gp_x1_file = None, gp_x2_file = None):
        if gp_x1_file is not None:
            self.gp_x1.writeModelTo(gp_x1_file)

        if gp_x2_file is not None:
            self.gp_x2.writeModelTo(gp_x2_file)

    def getStateLimits(self):
        return self.x1lim, self.x2lim
    
    def getStateCovarLimits(self):
        lowerLim = np.array([[self.stateVarLim[0], self.stateCovarLim[0]],
                             [self.stateCovarLim[0], self.stateVarLim[0]]])
        
        upperLim = np.array([[self.stateVarLim[1], self.stateCovarLim[1]],
                             [self.stateCovarLim[1], self.stateVarLim[1]]])
        
        lowerLim = np.reshape(np.transpose(lowerLim), (1, 4)).tolist()[0]
        upperLim = np.reshape(np.transpose(upperLim), (1, 4)).tolist()[0]
        
        return lowerLim, upperLim
    
    def getInputLimits(self):
        return self.ulim
    
    def getContinuousDynamics(self): 
        xdot = vertcat(-1*self.x[0] + self.paramD*(1 - self.x[0])*exp(self.x[1]/(1 + self.x[1]/self.paramGamma)),
                       -1*self.x[1] + self.paramB*self.paramD*(1 - self.x[0])*exp(self.x[1]/(1 + self.x[1]/self.paramGamma)) + self.paramBeta*(self.u - self.x[1]))
        
        return xdot
    
    def getDiscreteDynamics(self, controlInterval, RK_steps, addGP):
        DT = controlInterval/RK_steps
        f = Function('f', [self.x, self.u], [self.getContinuousDynamics()])
        
        X0 = MX.sym('X0', 2)
        U = MX.sym('U', 1)
        X = X0
        for j in range(RK_steps):
            k1 = f(X, U)
            k2 = f(X + DT/2 * k1, U)
            k3 = f(X + DT/2 * k2, U)
            k4 = f(X + DT * k3, U)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        
        F = Function('F', [X0, U], [X],['x0', 'u'],['xf'])
        if addGP and self.gpReady:
            # x1Residual = 0 #self.gp_x1.getResidualFunction()
            x2Residual = self.gp_x2.getResidualFunction()
            XCOVAR0 = MX.sym('XCOVAR0', 4)
            # Use f to find covariances 
            DCOVAR0 = x2Residual(x=X0, u=U)['covar'] #diag(vertcat(0, x2Residual(x=X0, u=U)['covar'])) #x1Residual(x=X0, u=U)['covar']
            XUCOVAR0 = GenMX_zeros(2, 1)
            XDCOVAR0 = GenMX_zeros(2, 1)
            UCOVAR0 = GenMX_zeros(1, 1)
            UDCOVAR0 = GenMX_zeros(1, 1)
            COVAR0 = vertcat(horzcat(reshape(XCOVAR0, (2, 2)), XUCOVAR0, XDCOVAR0), horzcat(transpose(XUCOVAR0), UCOVAR0, UDCOVAR0), horzcat(transpose(XDCOVAR0), transpose(UDCOVAR0), DCOVAR0))
            XCOVAR = reshape(horzcat(jacobian(X, X0), jacobian(X, U), self.Bd) @ COVAR0 @ transpose(horzcat(jacobian(X, X0), jacobian(X, U), self.Bd)), (4, 1))
            # Now add g to f
            X = X + vertcat(0, x2Residual(x=X0, u=U)['mean']) #x1Residual(x=X0, u=U)['mean']
            F = Function('F', [X0, XCOVAR0, U], [X, XCOVAR],['x0', 'xcovar0', 'u'],['xf', 'xcovarf'])
            
        return F
    
    def getStateVar(self):
        return self.x
    
    def getInputVar(self):
        return self.u
    
class cstrSimModel:
    def __init__(self, paramD, paramB, paramGamma, paramBeta, x1_0, x2_0) -> None:
        self.paramD = paramD
        self.paramB = paramB
        self.paramGamma = paramGamma
        self.paramBeta = paramBeta
        self.stateVector = [x1_0, x2_0]

    def initState(self, x1_0, x2_0):
        self.stateVector = [x1_0, x2_0]

    def simNext(self, addNoise, interval, simResolution, u):
        t = numpy.arange(0, interval, simResolution)

        x_next = odeint(self.dynamics, self.stateVector, t, args=(u,))
        print(x_next[-1])
        dx1 = np_rng.normal(0, 0.003)
        dx2 = np_rng.normal(0, 0.003)
        noise = np.array([dx1, dx2])
        print("x next dist: ", x_next[-1])
        print("noise: ", noise)
        self.stateVector = x_next[-1] + noise
        return self.stateVector
    
    def dynamics(self, x, t, u):
        ad = 0.03
        bd = 1.5
        xdot = [-1*x[0] + self.paramD*(1 - x[0])*math.exp(x[1]/(1 + x[1]/self.paramGamma)),
                -1*x[1] + self.paramB*self.paramD*(1 - x[0])*exp(x[1]/(1 + x[1]/self.paramGamma)) + self.paramBeta*(u - x[1]) + ad*math.exp(-1*bd*u)] #
        return xdot
    
    def nomDynamics(self, x, t, u):
        xdot = [-1*x[0] + self.paramD*(1 - x[0])*math.exp(x[1]/(1 + x[1]/self.paramGamma)),
                -1*x[1] + self.paramB*self.paramD*(1 - x[0])*exp(x[1]/(1 + x[1]/self.paramGamma)) + self.paramBeta*(u - x[1])]
        return xdot
