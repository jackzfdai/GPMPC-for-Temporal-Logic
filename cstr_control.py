import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
from casadi import *

import cstr_model as cstr

smoothOp_state_trace_file = open("./trace_data/cstr_30x30_state_traces_smoothOp.txt", "w")
smoothOp_control_trace_file = open("./trace_data/cstr_30x30_control_traces_smoothOp.txt", "w")
GP_state_trace_file = open("./trace_data/cstr_30x30_state_traces_GP.txt", "w")
GP_control_trace_file = open("./trace_data/cstr_30x30_control_traces_GP.txt", "w")
Nom_state_trace_file = open("./trace_data/cstr_30x30_state_traces_nom.txt", "w")
Nom_control_trace_file = open("./trace_data/cstr_30x30_control_traces_nom.txt", "w")

cstr_paramD = 0.078
cstr_paramB = 8
cstr_paramGamma = 20
cstr_paramBeta = 0.3

x1_0 = 0
x2_0 = 0.0
xCovar0 = [1e-6]*4

cstr_sys = cstr.cstrModel(cstr_paramD, cstr_paramB, cstr_paramGamma, cstr_paramBeta)
cstr_sim_sys = cstr.cstrSimModel(cstr_paramD, cstr_paramB, cstr_paramGamma, cstr_paramBeta, x1_0, x2_0)

cstr_x1lim, cstr_x2lim = cstr_sys.getStateLimits()
cstr_ulim = cstr_sys.getInputLimits()

#GP Parameter Estimation
var_residual = 0.05
K_residual = 0.25

torch.manual_seed(0)

#Initial conditions and problem parameters
T = 6
N = 12 

bigMx1 = (cstr_x1lim[1] - cstr_x1lim[0]) + 1
bigMx2 = (cstr_x2lim[1] - cstr_x2lim[0]) + 1

C_th = 0.1
T_th = 0.5

def controlToSetPoint(plot, T, N, cstr, x1_0, x2_0, setpoint, controlEffortCost, referenceVarTraj = None, referenceControlTraj = None):
    feasible = False

    # Declare model variables
    x = cstr.getStateVar()
    # xCovar = residual.getStateCovarVar()
    u = cstr.getInputVar()

    x1lim, x2lim = cstr.getStateLimits()
    ulim = cstr.getInputLimits()

    lbx = [x1lim[0], x2lim[0]]
    ubx = [x1lim[1], x2lim[1]]

    lbu = [ulim[0]]
    ubu = [ulim[1]]

    x_init = [x1_0, x2_0]

    M = 4 # RK4 steps per interval
    DT = T/N/M

    cstr_dynamics = cstr.getDiscreteDynamics(T/N, M, False)

    # Objective term
    L = 100*(x[1] - setpoint)**2  
    L_controlEffort = u[0] ** 2 

    control_cost = Function('cost', [x, u], [L_controlEffort])
    terminal_cost = Function('term_cost', [x], [L])
            
    # Initial guess for u
    u_start = [DM([0])] * N

    # Get a feasible trajectory as an initial guess
    xk = DM(x_init)
    x_start = [xk]
    for k in range(N):
        sysk = cstr_dynamics(x0=xk, u=u_start[k])
        xk = sysk['xf']
        x_start += [xk]

    print(x_start)
    
    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    discrete = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # "Lift" initial conditions
    X0 = MX.sym('X0', 2)
    w += [X0]
    lbw += x_init
    ubw += x_init
    w0 += [x_start[0]]
    discrete += [False]*2

    Xk = X0

    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), 1)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += [u_start[k]]
        discrete += [False]

        # Integrate till the end of the interval
        sysk = cstr_dynamics(x0=Xk, u=Uk)
        Xk_end = sysk['xf']
        if controlEffortCost:
            J=J+control_cost(Xk, Uk)

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), 2)
        w   += [Xk]
        lbw += lbx
        ubw += ubx
        w0  += [x_start[k+1]]
        discrete += [False]*2

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*2
        ubg += [0]*2

    J = J + terminal_cost(Xk)

    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    nlp_prob = {'f': J, 'x': w, 'g': g}
    nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete})
    #nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete})
    #nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

    # Solve the NLP
    sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(nlp_solver.stats())

    solver_stats = nlp_solver.stats()
    proc_runtime = solver_stats['t_proc_total']
    feasible = solver_stats['success']

    w1_opt = sol['x']
    
    state_opt = []
    control_opt = []

    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x1_opt = w1_opt[0:3*(N)+2][0::3]
        x2_opt = w1_opt[0:3*(N)+2][1::3]
        u_opt = w1_opt[0:3*(N)+2][2::3]

        print(x1_opt)
        print(x2_opt)
        print(u_opt)

        state_opt = np.transpose(np.vstack([x1_opt, x2_opt]))
        control_opt = u_opt

        if plot == True:
            plotSol(N, x1_opt, x2_opt, u_opt)

    return proc_runtime, feasible, state_opt, control_opt

def solveMINLP(plot, T, N, cstr, x1_0, x2_0, xCovar0, addGP, referenceVarTraj = None, referenceControlTraj = None):
    feasible = False
    proc_runtime = 0
    state_opt = []
    control_opt = []
    
    # Declare model variables
    x = cstr.getStateVar()
    u = cstr.getInputVar()

    stateLen = x.size()[0]
    stateCovarLen = stateLen**2 
    controlLen = u.size()[0]

    stateTrajRef = None
    stateCovarTrajRef = None
    binVarTrajRef = None
    controlTrajRef = None

    if referenceVarTraj is not None:
        stateTrajRef = referenceVarTraj[:, 0:stateLen]
        stateCovarTrajRef = referenceVarTraj[:, stateLen:stateLen+stateCovarLen]
        binVarTrajRef = referenceVarTraj[:, stateLen+stateCovarLen:]

    if referenceControlTraj is not None:
        controlTrajRef = referenceControlTraj

    x1lim, x2lim = cstr.getStateLimits()
    stateCovarLowerLim, stateCovarUpperLim = cstr.getStateCovarLimits()
    ulim = cstr.getInputLimits()

    lbx = [x1lim[0], x2lim[0]]
    ubx = [x1lim[1], x2lim[1]]

    lbxcovar = stateCovarLowerLim
    ubxcovar = stateCovarUpperLim

    lbu = [ulim[0]]
    ubu = [ulim[1]]

    lbbin = [0]
    ubbin = [1]

    invCDFVarphiEpsilon = norm.ppf(0.25)
    smallEps = 1e-6

    x_init = [x1_0, x2_0]
    xcovar_init = xCovar0

    M = 4 # RK4 steps per interval
    DT = T/N/M

    cstr_dynamics = cstr.getDiscreteDynamics(T/N, M, addGP)

    # Objective term
    L = u**2 

    cost = Function('cost', [x, u], [L])
            
    # Initial guess for u
    u_start = [DM([0])] * N
    if controlTrajRef is not None:
        u_start = controlTrajRef

    # Get a feasible trajectory as an initial guess
    x_start = []
    xcovar_start = []
    if stateTrajRef is None or stateCovarTrajRef is None: 
        xk = DM(x_init)
        x_start = [xk]
        xcovark = DM(xcovar_init)
        xcovar_start = [xcovark]
        for k in range(N):
            if addGP:
                sysk = cstr_dynamics(x0=xk,xcovar0=xcovark, u=u_start[k])
                xk = sysk['xf']
                x_start += [xk]
                xcovark = sysk['xcovarf']
                xcovar_start += [xcovark]
            else:
                sysk = cstr_dynamics(x0=xk, u=u_start[k])
                xk = sysk['xf']
                x_start += [xk]
    
    if stateTrajRef is not None: 
        x_start = stateTrajRef
    if stateCovarTrajRef is not None:
        xcovar_start = stateCovarTrajRef

    print(x_start)
    print(xcovar_start)

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    discrete = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # "Lift" initial conditions
    X0 = MX.sym('X0', stateLen)
    w += [X0]
    lbw += x_init
    ubw += x_init
    w0 += [x_start[0]]
    discrete += [False]*stateLen

    if addGP:
        XCOVAR0 = MX.sym('XCOVAR0', stateCovarLen)
        w += [XCOVAR0]
        lbw += xcovar_init
        ubw += xcovar_init
        w0 += [xcovar_start[0]]
        discrete += [False]*stateCovarLen

    ZC0 = MX.sym('ZC0', 1)
    w += [ZC0]
    lbw += lbbin
    ubw += ubbin
    w0 += [0]
    discrete += [True]

    ZT0 = MX.sym('ZT0', 1)
    w += [ZT0]
    lbw += lbbin
    ubw += ubbin
    w0 += [0]
    discrete += [True]

    if addGP: 
        g += [X0[0] - C_th + invCDFVarphiEpsilon * sqrt(XCOVAR0[0] + smallEps) + bigMx1*(1 - ZC0)]
        g += [T_th - X0[1] + invCDFVarphiEpsilon * sqrt(XCOVAR0[3] + smallEps) + bigMx2*(1 - ZT0)] 
    else: 
        g += [X0[0] - C_th + bigMx1*(1 - ZC0)]
        g += [T_th - X0[1] + bigMx2*(1 - ZT0)] 
    
    g += [ZC0 + ZT0]

    lbg += [0, 0, 1]
    ubg += [bigMx1, bigMx2, 2]

    # Formulate the NLP
    Xk = X0
    
    if addGP:
        XCOVARk = XCOVAR0

    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), controlLen)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += [u_start[k]]
        discrete += [False]*controlLen

        # Integrate till the end of the interval
        if addGP:
            sysk = cstr_dynamics(x0=Xk, xcovar0=XCOVARk, u=Uk)
            Xk_end = sysk['xf']
            XCOVARk_end = sysk['xcovarf']
        else:
            sysk = cstr_dynamics(x0=Xk, u=Uk)
            Xk_end = sysk['xf']
        
        J=J+cost(Xk, Uk)

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), stateLen)
        w   += [Xk]
        lbw += lbx
        ubw += ubx
        w0  += [x_start[k+1]]
        discrete += [False]*stateLen

        if addGP:
            XCOVARk = MX.sym('XCOVAR_' + str(k+1), stateCovarLen)
            w += [XCOVARk]
            lbw += lbxcovar
            ubw += ubxcovar
            w0 += [xcovar_start[k+1]]
            discrete += [False]*stateCovarLen

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*stateLen
        ubg += [0]*stateLen

        if addGP:
            g   += [XCOVARk_end - XCOVARk]
            lbg += [0]*stateCovarLen
            ubg += [0]*stateCovarLen

        # STL constraints
        ZCk = MX.sym('ZC' + str(k+1), 1)
        w += [ZCk]
        lbw += lbbin
        ubw += ubbin
        w0 += [0]
        discrete += [True]

        ZTk = MX.sym('ZT' + str(k+1), 1)
        w += [ZTk]
        lbw += lbbin
        ubw += ubbin
        w0 += [0]
        discrete += [True]

        if addGP:
            g += [Xk[0] - C_th + invCDFVarphiEpsilon * sqrt(XCOVARk[0] + smallEps) + bigMx1*(1 - ZCk)]
            g += [T_th - Xk[1] + invCDFVarphiEpsilon * sqrt(XCOVARk[3] + smallEps) + bigMx2*(1 - ZTk)] 
        else:
            g += [Xk[0] - C_th + bigMx1*(1 - ZCk)]
            g += [T_th - Xk[1] + bigMx2*(1 - ZTk)] 

        g += [ZCk + ZTk]

        lbg += [0, 0, 1]
        ubg += [bigMx1, bigMx2, 2]
        
        
    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    nlp_prob = {'f': J, 'x': w, 'g': g}
    nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete})
    #nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete})
    #nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

    # Solve the NLP
    sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(nlp_solver.stats())

    solver_stats = nlp_solver.stats()
    proc_runtime = solver_stats['t_proc_total']
    feasible = solver_stats['success']

    w1_opt = sol['x']
    
    state_opt = []
    control_opt = []

    numBinVarsPerTimestep = 2
    numVarsPerTimestep = stateLen + numBinVarsPerTimestep + controlLen
    numStateVarsPerTimestep = stateLen + numBinVarsPerTimestep
    if addGP: 
        numVarsPerTimestep = stateLen + stateCovarLen + numBinVarsPerTimestep + controlLen
        numStateVarsPerTimestep = stateLen + stateCovarLen + numBinVarsPerTimestep
    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x1_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][0::numVarsPerTimestep]
        x2_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][1::numVarsPerTimestep]
        if addGP:
            x11covar_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][2::numVarsPerTimestep]
            x12covar_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][3::numVarsPerTimestep]
            x21covar_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][4::numVarsPerTimestep]
            x22covar_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][5::numVarsPerTimestep]

        zc_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][numVarsPerTimestep-3::numVarsPerTimestep]
        zt_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][numVarsPerTimestep-2::numVarsPerTimestep]
        u_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][numVarsPerTimestep-1::numVarsPerTimestep]

        print(x1_opt)
        print(x2_opt)
        if addGP:
            print(x22covar_opt)
        print(zc_opt)
        print(zt_opt)
        print(u_opt)

        state_opt = np.transpose(np.vstack([x1_opt, x2_opt]))
        control_opt = u_opt

        if plot == True:
            plotSol(N, x1_opt, x2_opt, u_opt)

    return proc_runtime, feasible, state_opt, control_opt

def solveSmoothRobustness(plot, T, N, cstr, x1_0, x2_0, referenceVarTraj = None, referenceControlTraj = None):
    feasible = False
    
    # Declare model variables
    x = cstr.getStateVar()
    u = cstr.getInputVar()

    stateLen = x.size()[0]
    controlLen = u.size()[0]

    x1lim, x2lim = cstr.getStateLimits()
    ulim = cstr.getInputLimits()

    lbx = [x1lim[0], x2lim[0]]
    ubx = [x1lim[1], x2lim[1]]

    lbu = [ulim[0]]
    ubu = [ulim[1]]

    lbbin = [0]
    ubbin = [1]

    x_init = [x1_0, x2_0]

    M = 4 # RK4 steps per interval
    DT = T/N/M

    cstr_dynamics = cstr.getDiscreteDynamics(T/N, M, False)

    # Objective term
    L = 100*u**2 

    cost = Function('cost', [x, u], [L])
            
    # Initial guess for u
    u_start = [DM([0])] * N

    # Get a feasible trajectory as an initial guess
    xk = DM(x_init)
    x_start = [xk]
    for k in range(N):
        sysk = cstr_dynamics(x0=xk, u=u_start[k])
        xk = sysk['xf']
        x_start += [xk]

    print(x_start)

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    discrete = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # "Lift" initial conditions
    X0 = MX.sym('X0', stateLen)
    w += [X0]
    lbw += x_init
    ubw += x_init
    w0 += [x_start[0]]
    discrete += [False]*stateLen

    totalRobustness = exp(-1*log(exp(X0[0] - C_th) + exp(T_th - X0[1])))
    # Formulate the NLP
    Xk = X0

    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), controlLen)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += [u_start[k]]
        discrete += [False]*controlLen

        # Integrate till the end of the interval
        sysk = cstr_dynamics(x0=Xk, u=Uk)
        Xk_end = sysk['xf']
        
        # J=J+cost(Xk, Uk)

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), stateLen)
        w   += [Xk]
        lbw += lbx
        ubw += ubx
        w0  += [x_start[k+1]]
        discrete += [False]*stateLen

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*stateLen
        ubg += [0]*stateLen

        # STL constraints
        totalRobustness += exp(-1*log(exp(Xk[0] - C_th) + exp(T_th - Xk[1])))

        
    J += 100*log(totalRobustness)
    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    nlp_prob = {'f': J, 'x': w, 'g': g}
    # nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete})
    #nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete})
    nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

    # Solve the NLP
    sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(nlp_solver.stats())

    solver_stats = nlp_solver.stats()
    proc_runtime = solver_stats['t_proc_total']
    feasible = solver_stats['success']

    w1_opt = sol['x']
    
    state_opt = []
    control_opt = []

    numBinVarsPerTimestep = 0
    numVarsPerTimestep = stateLen + numBinVarsPerTimestep + controlLen
    numStateVarsPerTimestep = stateLen + numBinVarsPerTimestep
    
    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x1_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][0::numVarsPerTimestep]
        x2_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][1::numVarsPerTimestep]
        u_opt = w1_opt[0:numVarsPerTimestep*(N)+numStateVarsPerTimestep][numVarsPerTimestep-1::numVarsPerTimestep]

        print(x1_opt)
        print(x2_opt)
        print(u_opt)

        state_opt = np.transpose(np.vstack([x1_opt, x2_opt]))
        control_opt = u_opt

        if plot == True:
            plotSol(N, x1_opt, x2_opt, u_opt)

    return proc_runtime, feasible, state_opt, control_opt

def controlLoop(plot, individual_plot, writeTraces, specificationController, T, N, cstr_control_model, cstr_sim_model, x1_0, x2_0, controller, controllerExtraParams=(), stateTrajFile=[], controlTrajFile=[]):
    dT = T/N
    simRes = 0.01
    
    maxSolveTime = 0
    totalSolveTime = 0

    cstr_sim_model.initState(x1_0, x2_0)
    stateVector = np.array([x1_0, x2_0])

    x1_openLoopList = []
    x2_openLoopList = []
    u_openLoopList = []

    closedLoopStates = [stateVector]
    closedLoopPredictedStates = [stateVector]
    closedLoopControls = []
    
    recursivelyFeasible = False

    varTrajRef = None
    controlVarTrajRef = None

    num_iters = 0
    for i in range(N):
        if specificationController:
            if stateVector[0] < C_th and stateVector[1] > T_th:
                recursivelyFeasible = False
                print("Starting state infeasible, exiting.")
                break
    
        solveTime, feasible, stateTraj, controlTraj = controller(individual_plot, T-i*(T/N), N-i, cstr_control_model, stateVector[0], stateVector[1], *controllerExtraParams, referenceVarTraj = varTrajRef, referenceControlTraj = controlVarTrajRef)

        num_iters += 1

        totalSolveTime += solveTime
        if solveTime > maxSolveTime:
            maxSolveTime = solveTime

        if not feasible: 
            recursivelyFeasible = False
            print("Optimization failed, exiting.")
            break
        else:
            recursivelyFeasible = True
        
        x1_openLoopList += [stateTraj[:, 0]]
        x2_openLoopList += [stateTraj[:, 1]]
        u_openLoopList += [controlTraj]

        print("Current state: ", stateTraj[0, :])
        print("Control input: ", controlTraj[0])
        print("Predicted next state: ", stateTraj[1, :])
        closedLoopPredictedStates += [stateTraj[1, :]]

        stateVector = cstr_sim_model.simNext(True, dT, simRes, controlTraj[0])

        print("Simulated next state: ", stateVector)

        closedLoopControls += [controlTraj[0]]
        closedLoopStates += [stateVector]

    averageSolveTime = totalSolveTime/N

    closedLoopStates = np.vstack(closedLoopStates)
    if len(closedLoopControls) > 0:
        closedLoopControls = np.vstack(closedLoopControls)
    else:
        closedLoopControls = np.array([])
    closedLoopPredictedStates = np.vstack(closedLoopPredictedStates)
    
    manualCost = np.sum(np.square(closedLoopControls))

    if plot == True:
        plotSol(N, closedLoopStates[:, 0], closedLoopStates[:, 1], closedLoopControls, x1_openLoopList, x2_openLoopList, u_openLoopList)

    print("Max solve time: ", maxSolveTime, "s")
    print("Average solve time: ", averageSolveTime, "s")

    if writeTraces:
        for i in range(closedLoopStates.shape[0]):
            stateTrajFile.write(str(closedLoopStates[i, 0]) + "," + str(closedLoopStates[i, 1]) + "\n")
        stateTrajFile.write("~\n")
        for i in range(closedLoopControls.shape[0]):
            controlTrajFile.write(str(closedLoopControls[i, 0]) + "\n")
        controlTrajFile.write("~\n")

    return recursivelyFeasible, averageSolveTime, maxSolveTime, totalSolveTime, num_iters, manualCost, closedLoopControls, closedLoopPredictedStates, closedLoopStates

def plotSol(N, x1, x2, u, x1_openloops = [], x2_openloops = [], u_openloops = []):
    plt.figure()

    N_use = min(N, len(u))

    tgrid = [k for k in range(N_use+1)]
    plt.plot(tgrid, x1, '-.')
    plt.plot(tgrid, x2, '--')
    plt.step(tgrid, vertcat(u, DM.nan(1)), '.')
    plt.xlabel('t')
    plt.legend(['x1','x2', 'u'])
    plt.grid(True)

    if len(x1_openloops) >= N_use and len(x2_openloops) >= N_use and len(u_openloops) >= N_use:
        for i in range(N_use): 
            tgrid_i = [k for k in range(i, N+1)]
            plt.plot(tgrid_i, x1_openloops[i], alpha=0.5, linestyle='-', linewidth=1)
            plt.plot(tgrid_i, x2_openloops[i], alpha=0.5, linestyle='-', linewidth=1)
            plt.plot(tgrid_i, vertcat(u_openloops[i], DM.nan(1)), '-', alpha=0.5, linewidth=1)

        
#-----begin GP training from data----#
# g_x1_list = []
# g_x2_list = []
# controlList = []
# stateInputDataList = []

# setpointList = [0.5, 0.7, 0.5, 0.8, 0.6]
# x1_0List = [0, 0.2, 0.05, 0.15, 0]
# x2_0List = [0, 0.1, 0.5, 0., 0.1]

# for i in range(3): 
#     _, _, _, _, _, _, controlData, predictedStateData, actualStateData = controlLoop(True, False, False, False, T/2, int(N/2), cstr_sys, cstr_sim_sys, x1_0List[i], x2_0List[i], controlToSetPoint, (setpointList[i], True))
    
#     controlList += [controlData]
#     stateInputDataList += [actualStateData[:-1, :]]
#     g_x1_list += [actualStateData[1:, 0] - predictedStateData[1:, 0]]
#     g_x2_list += [actualStateData[1:, 1] - predictedStateData[1:, 1]]

# g_x1 = np.hstack(g_x1_list)
# g_x2 = np.hstack(g_x2_list)

# controlData = np.vstack(controlList)
# stateInputData = np.vstack(stateInputDataList)

# trainInputs = torch.tensor(np.hstack([stateInputData, controlData]))
# train_x1 = torch.tensor(g_x1)
# train_x2 = torch.tensor(g_x2)
# trainOutputs = [train_x1, train_x2]

# cstr_sys.setGPResiduals(150, [trainInputs]*2, trainOutputs)

# cstr_GP_x1_file = open("./gp_model_data/cstr_GP_x1.txt", "w")
# cstr_GP_x2_file = open("./gp_model_data/cstr_GP_x2.txt", "w")

# cstr_sys.writeGPResidualsToFiles(cstr_GP_x1_file, cstr_GP_x2_file)

# cstr_GP_x1_file.close()
# cstr_GP_x2_file.close()
#-----end GP training from data----#

#-----begin GP load from file----#
cstr_GP_x1_file = open("./gp_model_data/cstr_GP_x1.txt", "r")
cstr_GP_x2_file = open("./gp_model_data/cstr_GP_x2.txt", "r")

cstr_sys.setGPResidualsFromFile(100, cstr_GP_x1_file, cstr_GP_x2_file)

cstr_GP_x1_file.close()
cstr_GP_x2_file.close()
#-----end GP load from file----#

numTrials = 30
numConfigs = 30

np_rng = np.random.default_rng(seed=0)
x1_0_configs = []
x2_0_configs = []
while len(x1_0_configs) < numConfigs :
    x_0_candidate = np_rng.random(2) * 0.1 
    if x_0_candidate[0] > C_th or x_0_candidate[1] < T_th:
        x1_0_configs += [x_0_candidate[0]]
        x2_0_configs += [x_0_candidate[1]]

totalCostSmoothOpList = []
totalCostGPList = []
totalCostNomList = []

maxSolveTimeSmoothOpList = []
maxSolveTimeGPList = []
maxSolveTimeNomList = []

bigTotalTimeSmoothOp = 0
bigTotalTimeGP = 0
bigTotalTimeNom = 0

totalItersSmoothOp = 0
totalItersGp = 0
totalItersNom = 0

for i in range(numConfigs):
    x1_0_i = x1_0_configs[i]
    x2_0_i = x2_0_configs[i]

    satSmoothOp = 0
    satGP = 0
    satNom = 0

    totalCostSmoothOp = 0
    totalCostGP = 0
    totalCostNom = 0

    maxSolveTimeForConfigSmoothOp = 0
    maxSolveTimeForConfigGP = 0
    maxSolveTimeForConfigNom = 0

    for j in range(numTrials):
        recursiveFeasibilitySmoothOp, avgSolveTimeSmoothOp, maxSolveTimeSmoothOp, totalSolveTimeSmoothOp, itersCompletedSmoothOp, costSmoothOp, _, _, stateTrajSmoothOp = controlLoop(False, False, True, True, T, N, cstr_sys, cstr_sim_sys, x1_0_i, x2_0_i, solveSmoothRobustness, (), smoothOp_state_trace_file, smoothOp_control_trace_file,)
        bigTotalTimeSmoothOp += totalSolveTimeSmoothOp
        totalItersSmoothOp += itersCompletedSmoothOp
        
        if maxSolveTimeSmoothOp > maxSolveTimeForConfigSmoothOp:
            maxSolveTimeForConfigSmoothOp = maxSolveTimeSmoothOp

        useGP = True
        recursiveFeasibilityGP, avgSolveTimeGP, maxSolveTimeGP, totalSolveTimeGP, itersCompletedGP, costGP, _, _, stateTrajGP = controlLoop(False, False, True, True, T, N, cstr_sys, cstr_sim_sys, x1_0_i, x2_0_i, solveMINLP, (xCovar0, useGP), GP_state_trace_file, GP_control_trace_file)
        bigTotalTimeGP += totalSolveTimeGP
        totalItersGp += itersCompletedGP
        
        if maxSolveTimeGP > maxSolveTimeForConfigGP:
            maxSolveTimeForConfigGP = maxSolveTimeGP

        useGP = False
        recursiveFeasibilityNom, avgSolveTimeNom, maxSolveTimeNom, totalSolveTimeNom, itersCompletedNom, costNom, _, _, stateTrajNom = controlLoop(False, False, True, True, T, N, cstr_sys, cstr_sim_sys, x1_0_i, x2_0_i, solveMINLP, (xCovar0, useGP), Nom_state_trace_file, Nom_control_trace_file)
        bigTotalTimeNom += totalSolveTimeNom
        totalItersNom += itersCompletedNom
        
        if maxSolveTimeNom > maxSolveTimeForConfigNom:
            maxSolveTimeForConfigNom = maxSolveTimeNom

    
    totalCostSmoothOpList += [totalCostSmoothOp]
    totalCostGPList += [totalCostGP]
    totalCostNomList += [totalCostNom]

    maxSolveTimeSmoothOpList += [maxSolveTimeForConfigSmoothOp]
    maxSolveTimeGPList += [maxSolveTimeForConfigGP]
    maxSolveTimeNomList += [maxSolveTimeForConfigNom]

print("-----initial states-----")
print("x1: ", x1_0_configs)
print("x2: ", x2_0_configs)

totalAvgTimeSmoothOp = -1
totalAvgTimeGP = -1
totalAvgTimeNom = -1

if totalItersSmoothOp > 0:
    totalAvgTimeSmoothOp = bigTotalTimeSmoothOp/totalItersSmoothOp
if totalItersGp > 0:
    totalAvgTimeGP = bigTotalTimeGP/totalItersGp
if totalItersNom > 0:
    totalAvgTimeNom = bigTotalTimeNom/totalItersNom

print("-----Time stats-----")
print("Avg time smooth op: ", totalAvgTimeSmoothOp)
print("Avg time GP: ", totalAvgTimeGP)
print("Avg time Nom: ", totalAvgTimeNom)
print("Max times smooth op: ")
print(maxSolveTimeSmoothOpList)
print("Max times GP: ")
print(maxSolveTimeGPList)
print("Max times Nom: ")
print(maxSolveTimeNomList)

print("-----Debug total iters stats-----")
print("Smooth Op: ", totalItersSmoothOp)
print("GP: ", totalItersGp)
print("Nom: ", totalItersNom)

smoothOp_state_trace_file.close()
smoothOp_control_trace_file.close()
GP_state_trace_file.close()
GP_control_trace_file.close()
Nom_state_trace_file.close()
Nom_control_trace_file.close()

plt.show()