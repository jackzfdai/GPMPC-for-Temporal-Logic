import numpy as np
from casadi import *

import gp_dynamics as gpdyn

class kinematicSingleTrackCarModel:
    def __init__(self, lwb, params): #, x0, y0, v0, carAng0
        self.lwb = lwb
        self.params = params 
        self.x = SX.sym('x', 5) #[x, y, steerAng, v, carAng]
        self.u = SX.sym('u', 2) #[vSteerAng, accel]
        self.xlim = [0, 0]
        self.ylim = [0, 0]
        self.steerAnglim = [0, 0]
        #self.deltaCovarLim
        self.vlim = [0, 0]
        self.carAngLim = [0, 0]
        self.vSteerAngLim = [0, 0]
        self.accelLim = [0, 0]
        
        self.stateCovarLim = [-15, 15]
        self.stateVarLim = [0, 15]

        self.gpReady = False
        self.Bd = 0

        self.ltvReadyNom = False
        self.ltvReadyGP = False

    def setLimits(self, xlim, ylim, steerAngLim, vlim, carAngLim, vSteerAngLim, accelLim):
        self.xlim = xlim
        self.ylim = ylim
        self.steerAngLim = steerAngLim
        self.vlim = vlim
        self.carAngLim = carAngLim
        self.vSteerAngLim = vSteerAngLim
        self.accelLim = accelLim
        

    def setGPResiduals(self, stateDims, inputDims, trainingIts, trainingInputData, trainingOutputData):
        self.gp_x = gpdyn.gpResidual(stateDims[0], inputDims[0])
        self.gp_y = gpdyn.gpResidual(stateDims[1], inputDims[1])
        self.gp_steerAng = 0 #gpdyn.gpResidual(stateDims[2], inputDims[2], trainingIts, trainingInputData[2], trainingOutputData[2])
        self.gp_v = gpdyn.gpResidual(stateDims[3], inputDims[3])
        self.gp_carAng = gpdyn.gpResidual(stateDims[4], inputDims[4])
        
        self.gp_x.train(trainingIts, trainingInputData[0], trainingOutputData[0])
        self.gp_y.train(trainingIts, trainingInputData[1], trainingOutputData[1])
        # self.gp_steerAng.train
        self.gp_v.train(trainingIts, trainingInputData[3], trainingOutputData[3])
        self.gp_carAng.train(trainingIts, trainingInputData[4], trainingOutputData[4])
        
        self.Bd = np.diag([1, 1, 0, 1, 1])

        # testInput1 = torch.tensor([[1, 0, 15, 0, 0, 0]])
        # testInput2 = torch.tensor([[3, 0, 15, 0, 0, 0]])
        # testInput3 = torch.tensor([[1, 0, 20, 0, 0, 0]])

        # testOut1 = self.gp_y.getPrediction(testInput1)
        # testOut2 = self.gp_y.getPrediction(testInput2)
        # testOut3 = self.gp_y.getPrediction(testInput3)

        # print(testOut1)
        # print(testOut2)
        # print(testOut3)
        self.gpReady = True

    def setGPResidualsFromFile(self, stateDims, inputDims, gp_x_file = None, gp_y_file = None, gp_steerAng_file = None, gp_v_file = None, gp_carAng_file = None):
        self.gp_x = gpdyn.gpResidual(stateDims[0], inputDims[0])
        self.gp_y = gpdyn.gpResidual(stateDims[1], inputDims[1])
        self.gp_steerAng = 0 #gpdyn.gpResidual(stateDims[2], inputDims[2], trainingIts, trainingInputData[2], trainingOutputData[2])
        self.gp_v = gpdyn.gpResidual(stateDims[3], inputDims[3])
        self.gp_carAng = gpdyn.gpResidual(stateDims[4], inputDims[4])
        
        if gp_x_file is not None:
            self.gp_x.parametersFromFile(gp_x_file)

        if gp_y_file is not None:
            self.gp_y.parametersFromFile(gp_y_file)
        
        if gp_steerAng_file is not None: 
            self.gp_steerAng.parametersFromFile(gp_steerAng_file)
        
        if gp_v_file is not None:
            self.gp_v.parametersFromFile(gp_v_file)

        if gp_carAng_file is not None:
            self.gp_carAng.parametersFromFile(gp_carAng_file)

        self.Bd = np.diag([1, 1, 0, 1, 1])

        # testInput1 = torch.tensor([[1, 0, 15, 0, 0, 0]])
        # testInput2 = torch.tensor([[3, 0, 15, 0, 0, 0]])
        # testInput3 = torch.tensor([[1, 0, 20, 0, 0, 0]])

        # testOut1 = self.gp_y.getPrediction(testInput1)
        # testOut2 = self.gp_y.getPrediction(testInput2)
        # testOut3 = self.gp_y.getPrediction(testInput3)

        # print(testOut1)
        # print(testOut2)
        # print(testOut3)
        self.gpReady = True

    def writeGPResidualsToFiles(self, gp_x_file = None, gp_y_file = None, gp_steerAng_file = None, gp_v_file = None, gp_carAng_file = None):
        if gp_x_file is not None:
            self.gp_x.writeModelTo(gp_x_file)

        if gp_y_file is not None:
            self.gp_y.writeModelTo(gp_y_file)
        
        if gp_steerAng_file is not None: 
            self.gp_steerAng.writeModelTo(gp_steerAng_file)
        
        if gp_v_file is not None:
            self.gp_v.writeModelTo(gp_v_file)

        if gp_carAng_file is not None:
            self.gp_carAng.writeModelTo(gp_carAng_file)

    def getStateLimits(self):
        return self.xlim, self.ylim, self.steerAngLim, self.vlim, self.carAngLim
    
    def getStateCovarLimits(self):
        lowerLim = np.array([[self.stateVarLim[0], self.stateCovarLim[0], self.stateCovarLim[0], self.stateCovarLim[0], self.stateCovarLim[0]],
                             [self.stateCovarLim[0], self.stateVarLim[0], self.stateCovarLim[0], self.stateCovarLim[0], self.stateCovarLim[0]],
                             [self.stateCovarLim[0], self.stateCovarLim[0], self.stateVarLim[0], self.stateCovarLim[0], self.stateCovarLim[0]],
                             [self.stateCovarLim[0], self.stateCovarLim[0], self.stateCovarLim[0], self.stateVarLim[0], self.stateCovarLim[0]],
                             [self.stateCovarLim[0], self.stateCovarLim[0], self.stateCovarLim[0], self.stateCovarLim[0], self.stateVarLim[0]]])
        
        upperLim = np.array([[self.stateVarLim[1], self.stateCovarLim[1], self.stateCovarLim[1], self.stateCovarLim[1], self.stateCovarLim[1]],
                             [self.stateCovarLim[1], self.stateVarLim[1], self.stateCovarLim[1], self.stateCovarLim[1], self.stateCovarLim[1]],
                             [self.stateCovarLim[1], self.stateCovarLim[1], self.stateVarLim[1], self.stateCovarLim[1], self.stateCovarLim[1]],
                             [self.stateCovarLim[1], self.stateCovarLim[1], self.stateCovarLim[1], self.stateVarLim[1], self.stateCovarLim[1]],
                             [self.stateCovarLim[1], self.stateCovarLim[1], self.stateCovarLim[1], self.stateCovarLim[1], self.stateVarLim[1]]])
        
        lowerLim = np.reshape(np.transpose(lowerLim), (1, 25)).tolist()[0]
        upperLim = np.reshape(np.transpose(upperLim), (1, 25)).tolist()[0]
        
        return lowerLim, upperLim
    
    def getInputLimits(self):
        return self.vSteerAngLim, self.accelLim
    
    def getStateVar(self):
        return self.x
    
    def getInputVar(self):
        return self.u
    
    def getSwitchingVelocity(self):
        return self.params.longitudinal.v_switch
    
    def getContinuousDynamics(self): #, DT=1, RK_steps=4
        # x indices: 0 - Sx, 1 - Sy, 2 - steerAng, 3 - V, 4 - carAng
        # u indices: 0 - vSteerAng, 1 - accel
        xdot = vertcat(self.x[3]*cos(self.x[4]),
                       self.x[3]*sin(self.x[4]),
                       self.u[0],
                       self.u[1],
                       self.x[3]/self.lwb*tan(self.x[2]))
        
        return xdot
    
    def getDiscreteDynamics(self, controlInterval, RK_steps, addGP):
        DT = controlInterval/RK_steps
        f = Function('f', [self.x, self.u], [self.getContinuousDynamics()])
        
        X0 = MX.sym('X0', 5)
        XCOVAR0 = MX.sym('XCOVAR0', 25)
        U = MX.sym('U', 2)
        X = X0
        for j in range(RK_steps):
            k1 = f(X, U)
            k2 = f(X + DT/2 * k1, U)
            k3 = f(X + DT/2 * k2, U)
            k4 = f(X + DT * k3, U)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        
        F = Function('F', [X0, U], [X],['x0', 'u'],['xf'])
        if addGP and self.gpReady:
            xResidual = self.gp_x.getResidualFunction()
            yResidual = self.gp_y.getResidualFunction()
            steerAngResidual = 0 #self.gp_steerAng.getResidualFunction()
            vResidual = self.gp_v.getResidualFunction()
            carAngResidual = self.gp_carAng.getResidualFunction()
            # steerAngResidual(x=X0, u=U)['covar']
            # Use f to find covariances 
            DCOVAR0 = diag(vertcat(xResidual(x=X0[1:], u=U)['covar'], yResidual(x=X0[1:], u=U)['covar'], steerAngResidual, vResidual(x=X0[1:], u=U)['covar'], carAngResidual(x=X0[1:], u=U)['covar']))
            XUCOVAR0 = GenMX_zeros(5, 2)
            XDCOVAR0 = GenMX_zeros(5, 5)
            UCOVAR0 = GenMX_zeros(2, 2)
            UDCOVAR0 = GenMX_zeros(2, 5)
            COVAR0 = vertcat(horzcat(reshape(XCOVAR0, (5, 5)), XUCOVAR0, XDCOVAR0), horzcat(transpose(XUCOVAR0), UCOVAR0, UDCOVAR0), horzcat(transpose(XDCOVAR0), transpose(UDCOVAR0), DCOVAR0))
            XCOVAR = reshape(horzcat(jacobian(X, X0), jacobian(X, U), self.Bd) @ COVAR0 @ transpose(horzcat(jacobian(X, X0), jacobian(X, U), self.Bd)), (25, 1))
            print(jacobian(X, X0))
            # Now add g to f
            #steerAngResidual(x=X0, u=U)['mean']
            X = X + vertcat(xResidual(x=X0[1:], u=U)['mean'], yResidual(x=X0[1:], u=U)['mean'], steerAngResidual, vResidual(x=X0[1:], u=U)['mean'], carAngResidual(x=X0[1:], u=U)['mean'])
            F = Function('F', [X0, XCOVAR0, U], [X, XCOVAR],['x0', 'xcovar0', 'u'],['xf', 'xcovarf'])
            # print(jacobian(X, X0))
        return F #, xResidual, yResidual, vResidual, carAngResidual
    
    def initDiscreteLTVJacobians(self, controlInterval, RK_steps, addGP):
        DT = controlInterval/RK_steps
        f = Function('f', [self.x, self.u], [self.getContinuousDynamics()])
        Xref = MX.sym('Xref', 5)
        Uref = MX.sym('Uref', 2)
        X = Xref
        for j in range(RK_steps):
            k1 = f(X, Uref)
            k2 = f(X + DT/2 * k1, Uref)
            k3 = f(X + DT/2 * k2, Uref)
            k4 = f(X + DT * k3, Uref)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)

        XCOVARref = MX.sym('XCOVARref',25)

        if addGP and self.gpReady:
            xResidual = self.gp_x.getResidualFunction()
            yResidual = self.gp_y.getResidualFunction()
            steerAngResidual = 0 #self.gp_steerAng.getResidualFunction()
            vResidual = self.gp_v.getResidualFunction()
            carAngResidual = self.gp_carAng.getResidualFunction()
        #     # Use f to find covariances 
            DCOVAR0 = diag(vertcat(xResidual(x=Xref[1:], u=Uref)['covar'], yResidual(x=Xref[1:], u=Uref)['covar'], steerAngResidual, vResidual(x=Xref[1:], u=Uref)['covar'], carAngResidual(x=Xref[1:], u=Uref)['covar']))
            XUCOVAR0 = GenMX_zeros(5, 2)
            XDCOVAR0 = GenMX_zeros(5, 5)
            UCOVAR0 = GenMX_zeros(2, 2)
            UDCOVAR0 = GenMX_zeros(2, 5)
            COVAR0 = vertcat(horzcat(reshape(XCOVARref, (5, 5)), XUCOVAR0, XDCOVAR0), horzcat(transpose(XUCOVAR0), UCOVAR0, UDCOVAR0), horzcat(transpose(XDCOVAR0), transpose(UDCOVAR0), DCOVAR0))
            XCOVAR = reshape(horzcat(jacobian(X, Xref), jacobian(X, Uref), self.Bd) @ COVAR0 @ transpose(horzcat(jacobian(X, Xref), jacobian(X, Uref), self.Bd)), (25, 1))
            
            jacXCOVAR = horzcat(jacobian(XCOVAR, Xref), jacobian(XCOVAR, Uref), jacobian(XCOVAR, XCOVARref))
            FCOVAR = Function('FCOVAR', [Xref, Uref, XCOVARref], [XCOVAR, jacXCOVAR], ['xref', 'uref', 'xcovarref'], ['xcovarf', 'jacxcovarreff'])

            self.jaccovar = FCOVAR 
            
            X = X + vertcat(xResidual(x=Xref[1:], u=Uref)['mean'], yResidual(x=Xref[1:], u=Uref)['mean'], steerAngResidual, vResidual(x=Xref[1:], u=Uref)['mean'], carAngResidual(x=Xref[1:], u=Uref)['mean'])
            jacX = horzcat(jacobian(X, Xref), jacobian(X, Uref))
            F = Function('F', [Xref, Uref], [X, jacX], ['xref', 'uref'], ['xf', 'jacxf'])

            self.jacmean_GP = F 
            self.ltvReadyGP = True
        else:
            jacX = horzcat(jacobian(X, Xref), jacobian(X, Uref))
            F = Function('F', [Xref, Uref], [X, jacX], ['xref', 'uref'], ['xf', 'jacxf'])

            self.jacmean = F 
            self.ltvReadyNom = True
        return 
    
    def getDiscreteLTVDynamicsFor(self, kCurr, controlInterval, RK_steps, statekRef, controlkRef, stateCovarkRef, addGP):
        if self.ltvReadyNom == False and addGP == False:
            self.initDiscreteLTVJacobians(controlInterval, RK_steps, addGP)
        if self.ltvReadyGP == False and addGP == True:
            self.initDiscreteLTVJacobians(controlInterval, RK_steps, addGP)

        statekRefVal = DM(statekRef)
        controlkRefVal = DM(controlkRef)
        
        X0 = MX.sym('X0', 5)
        U = MX.sym('U', 2)

        if addGP and self.gpReady:
            XCOVAR0 = MX.sym('XCOVAR0', 25)

            stateCovarkRefVal = DM(stateCovarkRef)
            XCOVAR_LTVparams = self.jaccovar(xref = statekRefVal, uref=controlkRefVal, xcovarref=stateCovarkRefVal)
            xCovarOffset = XCOVAR_LTVparams['xcovarf']
            xCovarSlope = XCOVAR_LTVparams['jacxcovarreff']


            LTVparams = self.jacmean_GP(xref = statekRefVal, uref=controlkRefVal)
            offset = LTVparams['xf']
            slope = LTVparams['jacxf']
            # print(xCovarOffset)

            # print(xCovarOffset + xCovarSlope @ vertcat(X0 - statekRefVal, U - controlkRefVal, XCOVAR0 - stateCovarkRefVal))
            F_LTV = Function('F_LTV' + str(kCurr), [X0, U, XCOVAR0], [offset + slope @ vertcat(X0 - statekRefVal, U - controlkRefVal), xCovarOffset + xCovarSlope @ vertcat(X0 - statekRefVal, U - controlkRefVal, XCOVAR0 - stateCovarkRefVal)], ['x', 'u', 'xcovar'], ['xf_ltv', 'xcovarf_ltv'])
        else:
            LTVparams = self.jacmean(xref = statekRefVal, uref=controlkRefVal)
            offset = LTVparams['xf']
            slope = LTVparams['jacxf']
            # print(slope)

            # print(offset + slope @ vertcat(X0 - statekRefVal, U - controlkRefVal))
            F_LTV = Function('F_LTV' + str(kCurr), [X0, U], [offset + slope @ vertcat(X0 - statekRefVal, U - controlkRefVal)], ['x', 'u'], ['xf_ltv'])


        return F_LTV
    
    def resetLTVstatus(self):
        self.ltvReadyNom = False
        self.ltvReadyGP = False

class singleTrackCarModel: #used as sim model, implemented as control model here to use as oracle
    def __init__(self, lwb, params): #, x0, y0, v0, carAng0
        self.lwb = lwb
        self.params = params 
        self.x = SX.sym('x', 7) #[x, y, steerAng, v, carAng]
        self.u = SX.sym('u', 2) #[vSteerAng, accel]
        self.xlim = [0, 0]
        self.ylim = [0, 0]
        self.steerAnglim = [0, 0]
        #self.deltaCovarLim
        self.vlim = [0, 0]
        self.carAngLim = [0, 0]
        self.dotCarAngLim = [0, 0]
        self.slipAngLim = [0, 0]
        self.vSteerAngLim = [0, 0]
        self.accelLim = [0, 0]

    def setLimits(self, xlim, ylim, steerAngLim, vlim, carAngLim, dotCarAngLim, slipAngLim, vSteerAngLim, accelLim):
        self.xlim = xlim
        self.ylim = ylim
        self.steerAngLim = steerAngLim
        self.vlim = vlim
        self.carAngLim = carAngLim
        self.dotCarAngLim = dotCarAngLim
        self.slipAngLim = slipAngLim
        self.vSteerAngLim = vSteerAngLim
        self.accelLim = accelLim

    def getStateLimits(self):
        return self.xlim, self.ylim, self.steerAngLim, self.vlim, self.carAngLim, self.dotCarAngLim, self.slipAngLim
    
    def getInputLimits(self):
        return self.vSteerAngLim, self.accelLim
    
    def getStateVar(self):
        return self.x
    
    def getInputVar(self):
        return self.u
    
    def getSwitchingVelocity(self):
        return self.params.longitudinal.v_switch
    
    def getContinuousDynamics(self, useDisturbance):
        # set gravity constant
        g = 9.81  # [m/s^2]

        # create equivalent bicycle parameters    
        mu = self.params.tire.p_dy1
        if useDisturbance:
            mu = self.params.tire.p_dy1*0.85 + self.params.tire.p_dy1*0.15*(sin(2*pi/3*(self.x[1] - 3/4) - 1))
        C_Sf = -self.params.tire.p_ky1 / self.params.tire.p_dy1
        C_Sr = -self.params.tire.p_ky1 / self.params.tire.p_dy1
        lf = self.params.a
        lr = self.params.b
        h = self.params.h_s
        m = self.params.m
        I = self.params.I_z

        # system dynamics
        xdot = vertcat(self.x[3] * cos(self.x[6] + self.x[4]),
                       self.x[3] * sin(self.x[6] + self.x[4]),
                       self.u[0],
                       self.u[1],
                       self.x[5],
                       -mu * m / (self.x[3] * I * (lr + lf)) * (
                        lf ** 2 * C_Sf * (g * lr - self.u[1] * h) + lr ** 2 * C_Sr * (g * lf + self.u[1] * h)) * self.x[5] \
                        + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + self.u[1] * h) - lf * C_Sf * (g * lr - self.u[1] * h)) * self.x[6] \
                        + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - self.u[1] * h) * self.x[2],
                        (mu / (self.x[3] ** 2 * (lr + lf)) * (C_Sr * (g * lf + self.u[1] * h) * lr - C_Sf * (g * lr - self.u[1] * h) * lf) - 1) *
                        self.x[5] \
                        - mu / (self.x[3] * (lr + lf)) * (C_Sr * (g * lf + self.u[1] * h) + C_Sf * (g * lr - self.u[1] * h)) * self.x[6] \
                        + mu / (self.x[3] * (lr + lf)) * (C_Sf * (g * lr - self.u[1] * h)) * self.x[2])
        
        return xdot
    
    def getDiscreteDynamics(self, controlInterval, RK_steps):
        DT = controlInterval/RK_steps
        f = Function('f', [self.x, self.u], [self.getContinuousDynamics(True)]) 
        
        X0 = MX.sym('X0', 7)
        U = MX.sym('U', 2)
        X = X0
        for j in range(RK_steps):
            k1 = f(X, U)
            k2 = f(X + DT/2 * k1, U)
            k3 = f(X + DT/2 * k2, U)
            k4 = f(X + DT * k3, U)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        
        F = Function('F', [X0, U], [X],['x0', 'u'],['xf'])

        return F