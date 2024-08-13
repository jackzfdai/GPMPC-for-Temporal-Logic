import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

from casadi import *

import single_track_model_car as stcar
from vehicle_sim_models import st_car_sim

smoothOp_state_trace = open("./trace_data/autocar_state_traces_smoothOp.txt", "w")
smoothOp_control_trace = open("./trace_data/autocar_control_traces_smoothOp.txt", "w")
LTVGP_state_trace = open("./trace_data/autocar_state_trace_LTVGP.txt", "w")
LTVGP_control_trace = open("./trace_data/autocar_control_trace_LTVGP.txt", "w")
LTVGP_state_trace_offlineCovar = open("./trace_data/autocar_state_trace_LTVGP_offlineCovar.txt", "w")
LTVGP_control_trace_offlineCovar = open("./trace_data/autocar_control_trace_LTVGP_offlineCovar.txt", "w")
nom_state_trace = open("./trace_data/autocar_state_trace_nom.txt", "w")
nom_control_trace = open("./trace_data/autocar_control_trace_nom.txt", "w")

# load parameters
p = parameters_vehicle2()
g = 9.81  # [m/s^2]

# initial state set 1 --------------------------------------------------------------
steerAng0 = 0
vel0 = 15
Psi0 = 0.
dotPsi0 = 0
beta0 = 0
sy0 = 1.8
sx0 = 1.4

# initial state set 2 --------
steerAng0_2 = 0.
vel0_2 = 14.99
Psi0_2 = 0.
dotPsi0_2 = 0
beta0_2 = 0
sy0_2 = 1.8
sx0_2 = 2

# initial state set 3 --------
steerAng0_3 = 0.
vel0_3 = 19.5
Psi0_3 = -0.
dotPsi0_3 = 0
beta0_3 = 0
sy0_3 = 2.5
sx0_3 = 1

# initial state set 4 --------
steerAng0_4 = 0.
vel0_4 = 17.5
Psi0_4 = -0.
dotPsi0_4 = 0
beta0_4 = 0
sy0_4 = 3.5
sx0_4 = 1

# initial state set 5 --------
steerAng0_5 = 0.
vel0_5 = 15.5
Psi0_5 = 0.0
dotPsi0_5 = 0
beta0_5 = 0
sy0_5 = 2
sx0_5 = 1

# initial state set 6 --------
steerAng0_6 = -0.
vel0_6 = 18
Psi0_6 = -0.
dotPsi0_6 = 0
beta0_6 = 0
sy0_6 = 2.8
sx0_6 = 0.5

# control model -----------------------------------------------------------------
carlf = 1.156 # this isn't in the framework
carlr = 1.422
xlim = [0, 100] #m
xCovarLim = [0, 100]
ylim = [0, 6] #m
yCovarLim = [0, 1000]
steerAngLim = [-1.066, 1.066]
vlim = [0, 50.8] #m/s
vCovarLim = [0, 100]
carAngLim = [-0.49*math.pi, 0.49*math.pi] 
carAngCovarLim = [0, 100]
vSteerAngLim = [-0.39, 0.39]
accelLim = [-11.4, 11.4]

car_sys = stcar.singleTrackCarModel(lwb = carlf + carlr, params = p)
car_sys.setLimits(xlim, ylim, steerAngLim, vlim, carAngLim, vSteerAngLim, accelLim)

car_sim = st_car_sim(sx0, sy0, steerAng0, vel0, Psi0)

simulation_resolution = 0.01

# GP Residual 
torch.manual_seed(0)

numTrainingIts = 250

car_residualStateDims = [4, 4, 4, 4, 4]
car_residualInputDims = [2, 2, 2, 2, 2]

# Reach avoid parameters
goal_x = [45, 100]
goal_y = [0, 3] 
obstacle_x = [30, 45]
obstacle_y = [0, 3]
goal_min_speed = 15
goal_carAng = [-1/20*math.pi, 1/20*math.pi]

bigMx = xlim[1] - xlim[0] + 1
bigMy = ylim[1] - ylim[0] + 1
bigMv = vlim[1] - vlim[0] + 1
bigMcarAng = carAngLim[1] - carAngLim[0] + 1

epsilon = 0

# Horizon 
T = 4
N = 32

def solveSetpointControl(plot, T, N, car, x0, y0, steerAng0, v0, carAng0, xN, yN, steerAngN, vN, carAngN):
    feasible = False

    # Declare model variables
    x = car.getStateVar()
    u = car.getInputVar()

    stateLen = x.size()[0]

    controlLen = u.size()[0]

    xlim, ylim, steerAnglim, vlim, carAngLim = car.getStateLimits()
    vSteerAngLim, aLim = car.getInputLimits()

    v_switch = car.getSwitchingVelocity()

    lbx = [xlim[0], ylim[0], steerAnglim[0], vlim[0], carAngLim[0]]
    ubx = [xlim[1], ylim[1], steerAnglim[1], vlim[1], carAngLim[1]]

    lbu = [vSteerAngLim[0], aLim[0]]
    ubu = [vSteerAngLim[1], aLim[1]]

    x_init = [x0, y0, steerAng0, v0, carAng0]

    M = 4 # RK4 steps per interval
    DT = T/N/M

    # Model equations
    car_dynamics = car.getDiscreteDynamics(T/N, M, False)

    # Objective term
    L_terminal = (x[0] - xN)**2 + (x[1] - yN)**2 + 10*(x[2] - steerAngN)**2 + (x[3] - vN)**2 + 10*(x[4] - carAngN)**2
    cost_terminal = Function('cost_term', [x, u], [L_terminal])
            
    # Initial guess for u
    u_start = [DM([0., 1.])] * N

    # Get a feasible trajectory as an initial guess
    xk = DM(x_init)
    x_start = [xk]
    for k in range(N):
        sysk = car_dynamics(x0=xk, u=u_start[k])
        xk = sysk['xf']
        x_start += [xk]
    
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

        # Acceleration limits
        g += [ubu[1] * v_switch/(log(exp(v_switch) + exp(Xk[3]))) - Uk[1]]
        
        lbg += [0]
        ubg += [ubu[1] - lbu[1]]

        # Integrate till the end of the interval
        Fk = car_dynamics(x0=Xk, u=Uk)
        Xk_end = Fk['xf']
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

    J += cost_terminal(Xk, Uk)

    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    nlp_prob = {'f': J, 'x': w, 'g': g}
    nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

    # Solve the NLP
    sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(nlp_solver.stats())

    solver_stats = nlp_solver.stats()
    proc_runtime = solver_stats['t_proc_total']
    feasible = solver_stats['success']

    w1_opt = sol['x']
    
    state_opt = np.array([])
    control_opt = np.array([])

    numSTLVar = 0
    numVarPerTimeStep = stateLen + numSTLVar + controlLen
    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][0::numVarPerTimeStep]
        y_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][1::numVarPerTimeStep]
        steerAng_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][2::numVarPerTimeStep]
        v_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][3::numVarPerTimeStep]
        carAng_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][4::numVarPerTimeStep]
        vSteerAng_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][numVarPerTimeStep-2::numVarPerTimeStep] 
        accel_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][numVarPerTimeStep-1::numVarPerTimeStep]

        state_opt = np.transpose(np.concatenate([[x_opt], [y_opt], [steerAng_opt], [v_opt], [carAng_opt]], axis=0))
        print(np.transpose(state_opt))
        control_opt = np.transpose(np.concatenate([[vSteerAng_opt], [accel_opt]], axis=0))
        print(np.transpose(control_opt))

        if plot == True:
            plotSol(N, x_opt, y_opt, steerAng_opt, v_opt, carAng_opt, vSteerAng_opt, accel_opt)
            plotTraj(x_opt, y_opt)

    return proc_runtime, feasible, state_opt, control_opt

def solveSmoothedRobustnessNLP(plot, controlCost, T, N, car, x0, y0, steerAng0, v0, carAng0, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y):
    feasible = False

    # Declare model variables
    x = car.getStateVar()
    u = car.getInputVar()

    stateLen = x.size()[0]
    controlLen = u.size()[0]

    xlim, ylim, steerAngLim, vlim, carAngLim = car.getStateLimits()
    vSteerAngLim, aLim = car.getInputLimits()

    lbx = [xlim[0], ylim[0], steerAngLim[0], vlim[0], carAngLim[0]]
    ubx = [xlim[1], ylim[1], steerAngLim[1], vlim[1], carAngLim[1]]

    lbu = [vSteerAngLim[0], aLim[0]]
    ubu = [vSteerAngLim[1], aLim[1]]

    x_init = [x0, y0, steerAng0, v0, carAng0]

    v_switch = car.getSwitchingVelocity()

    M = 4 # RK4 steps per interval
    DT = T/N/M

    # Model equations
    car_dynamics = car.getDiscreteDynamics(T/N, M, False)

    # Objective term
    L = 0 
    if controlCost:
        L = 0.5*(5*u[0]**2 + u[1]**2) 
    
    L_terminal = 10*((x[0] - 40) ** 2 + (x[1] - 1.5))
    cost = Function('cost', [x, u], [L])
    cost_terminal = Function('cost_terminal', [x, u], [L_terminal])
            
    # Initial guess for u
    u_start = [DM([0, 1])] * N

    # Get a feasible trajectory as an initial guess
    xk = DM(x_init)
    x_start = [xk]
    for k in range(N):
        sysk = car_dynamics(x0=xk, u=u_start[k])
        xk = sysk['xf']
        x_start += [xk]

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
    
    #Integer variables for STL 
    Ravoid_0 = MX.sym('Ravoid_0', 3)
    w += [Ravoid_0]
    lbw += [-1*bigMx]*3
    ubw += [1*bigMx]*3
    w0 += [0]*3
    discrete += [False]*3

    Rreach_0 = MX.sym('Rreach_0', 7)
    w += [Rreach_0]
    lbw += [-1*bigMx]*7
    ubw += [1*bigMx]*7
    w0 += [0]*7
    discrete += [False]*7

    RreachTotal_0 = MX.sym('RreachTotal_0', 1)
    w += [RreachTotal_0]
    lbw += [-1*bigMx]*1
    ubw += [1*bigMx]*1
    w0 += [0]*1
    discrete += [False]*1

    #STL constraints
    #avoid
    g += [obstacle_polygon_x[0] - X0[0] - Ravoid_0[0]] 
    g += [X0[0] - obstacle_polygon_x[1] - Ravoid_0[1]] 
    g += [X0[1] - obstacle_polygon_y[1] - Ravoid_0[2]] 

    lbg += [-1*epsilon, -1*epsilon, -1*epsilon]
    ubg += [epsilon, epsilon, epsilon]

    alwaysAvoidObstacle1Cumulation = exp(-1*log(exp(Ravoid_0[0]) + exp(Ravoid_0[1]) + exp(Ravoid_0[2])))

    #envelope
    alwaysEnvelope1Cumulation = exp(-1*(X0[1] - ylim[0])) + exp(-1*(ylim[1] - X0[1]))

    eventuallyAlways = [exp(-RreachTotal_0)]
    #reach
    g += [X0[0] - goal_A_polygon_x[0] - Rreach_0[0]] 
    g += [goal_A_polygon_x[1] - X0[0] - Rreach_0[1]] 
    g += [X0[1] - goal_A_polygon_y[0] - Rreach_0[2]] 
    g += [goal_A_polygon_y[1] - X0[1] - Rreach_0[3]] 
    g += [X0[3] - goal_min_speed - Rreach_0[4]]
    g += [X0[4] - goal_carAng[0] - Rreach_0[5]]
    g += [goal_carAng[1] - X0[4] - Rreach_0[6]]
    g += [-1*log(exp(-1*Rreach_0[0]) + exp(-1*Rreach_0[1]) + exp(-1*Rreach_0[2]) + exp(-1*Rreach_0[3]) + exp(-1*Rreach_0[4]) + exp(-1*Rreach_0[5]) + exp(-1*Rreach_0[6])) - RreachTotal_0]

    lbg += [-1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon]
    ubg += [epsilon, epsilon, epsilon, epsilon, epsilon, epsilon, epsilon, epsilon]

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

        # Acceleration limits
        g += [ubu[1] * v_switch/(log(exp(v_switch) + exp(Xk[3]))) - Uk[1]]
        
        lbg += [0]
        ubg += [ubu[1] - lbu[1]]

        # Integrate till the end of the interval
        Fk = car_dynamics(x0=Xk, u=Uk)
        Xk_end = Fk['xf']
        J=J+cost(Xk, Uk)

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

        #Integer variables for STL 
        Ravoid_k = MX.sym('Ravoid_' + str(k+1), 3)
        w += [Ravoid_k]
        lbw += [-1*bigMx]*3
        ubw += [1*bigMx]*3
        w0 += [0]*3
        discrete += [False]*3

        Rreach_k = MX.sym('Rreach_' + str(k+1), 7)
        w += [Rreach_k]
        lbw += [-1*bigMx]*7
        ubw += [1*bigMx]*7
        w0 += [0]*7
        discrete += [False]*7

        RreachTotal_k = MX.sym('RreachTotal_k' + str(k+1), 1)
        w += [RreachTotal_k]
        lbw += [-1*bigMx]*1
        ubw += [1*bigMx]*1
        w0 += [0]*1
        discrete += [False]*1

        #STL constraints
        #avoid
        g += [obstacle_polygon_x[0] - Xk[0] - Ravoid_k[0]] 
        g += [Xk[0] - obstacle_polygon_x[1] - Ravoid_k[1]] 
        g += [Xk[1] - obstacle_polygon_y[1] - Ravoid_k[2]] 
    
        lbg += [-1*epsilon, -1*epsilon, -1*epsilon]
        ubg += [epsilon, epsilon, epsilon]

        alwaysAvoidObstacle1Cumulation += exp(-1*log(exp(Ravoid_k[0]) + exp(Ravoid_k[1]) + exp(Ravoid_k[2])))
        
        #envelope
        alwaysEnvelope1Cumulation += exp(-1*(Xk[1] - ylim[0])) + exp(-1*(ylim[1] - Xk[1]))

        #reach
        g += [Xk[0] - goal_A_polygon_x[0] - Rreach_k[0]] 
        g += [goal_A_polygon_x[1] - Xk[0] - Rreach_k[1]] 
        g += [Xk[1] - goal_A_polygon_y[0] - Rreach_k[2]] 
        g += [goal_A_polygon_y[1] - Xk[1] - Rreach_k[3]] 
        g += [Xk[3] - goal_min_speed - Rreach_k[4]]
        g += [Xk[4] - goal_carAng[0] - Rreach_k[5]]
        g += [goal_carAng[1] - Xk[4] - Rreach_k[6]]
        g += [-1*log(exp(-1*Rreach_k[0]) + exp(-1*Rreach_k[1]) + exp(-1*Rreach_k[2]) + exp(-1*Rreach_k[3]) + exp(-1*Rreach_k[4]) + exp(-1*Rreach_k[5]) + exp(-1*Rreach_k[6])) - RreachTotal_k]

        lbg += [-1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon, -1*epsilon,]
        ubg += [epsilon, epsilon, epsilon, epsilon, epsilon, epsilon, epsilon, epsilon]

        for j in range(k+1):
            eventuallyAlways[j] = eventuallyAlways[j] + exp(-RreachTotal_k)

        eventuallyAlways += [exp(-RreachTotal_k)]

    eventuallyAlwaysConstraint = 0
    for k in range(N+1):
        eventuallyAlwaysConstraint = eventuallyAlwaysConstraint + exp(-1*log(eventuallyAlways[k]))

    eventuallyAlwaysConstraint = log(eventuallyAlwaysConstraint)

    envelopeCostScalar = 0.8
    if controlCost:
        envelopeCostScalar = 0.25
    J += 25*log(alwaysAvoidObstacle1Cumulation + envelopeCostScalar*alwaysEnvelope1Cumulation + exp(-1*eventuallyAlwaysConstraint))

    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    nlp_prob = {'f': J, 'x': w, 'g': g}
    nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

    # Solve the NLP
    sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(nlp_solver.stats())

    solver_stats = nlp_solver.stats()
    proc_runtime = solver_stats['t_proc_total']
    feasible = solver_stats['success']

    w1_opt = sol['x']
    
    state_opt = np.array([])
    control_opt = np.array([])

    numSTLVar = 11
    numVarPerTimeStep = stateLen + numSTLVar + controlLen
    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][0::numVarPerTimeStep]
        y_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][1::numVarPerTimeStep]
        steerAng_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][2::numVarPerTimeStep]
        v_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][3::numVarPerTimeStep]
        carAng_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][4::numVarPerTimeStep]
        vSteerAng_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][numVarPerTimeStep-2::numVarPerTimeStep] 
        accel_opt = w1_opt[0:numVarPerTimeStep*(N)+stateLen][numVarPerTimeStep-1::numVarPerTimeStep]

        state_opt = np.concatenate([[x_opt], [y_opt], [steerAng_opt], [v_opt], [carAng_opt]], axis=0)
        # print(np.transpose(state_opt))
        control_opt = np.concatenate([[vSteerAng_opt], [accel_opt]], axis=0)
        # print(np.transpose(control_opt))

        if plot == True:
            plotSol(N, x_opt, y_opt, steerAng_opt, v_opt, carAng_opt, vSteerAng_opt, accel_opt)
            plotTraj(x_opt, y_opt, goal_A_polygon_x=goal_A_polygon_x, goal_A_polygon_y=goal_A_polygon_y, obstacle_polygon_x=obstacle_polygon_x, obstacle_polygon_y=obstacle_polygon_y)

    return proc_runtime, feasible, state_opt, control_opt

def solveNomLTV(plot, T, N, car, stateTrajRef, controlTrajRef, sx0, sy0, steerAng0, v0, carAng0, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y):
    feasible = False

    # Declare model variables
    x = car.getStateVar()
    u = car.getInputVar()

    stateLen = x.size()[0]
    controlLen = u.size()[0]

    xlim, ylim, steerAnglim, vlim, carAngLim = car.getStateLimits()
    aLim, vSteerAngLim = car.getInputLimits()

    lbx = [xlim[0], ylim[0], steerAnglim[0], vlim[0], carAngLim[0]]
    ubx = [xlim[1], ylim[1], steerAnglim[1], vlim[1], carAngLim[1]]

    lbu = [aLim[0], vSteerAngLim[0]]
    ubu = [aLim[1], vSteerAngLim[1]]

    v_switch = car.getSwitchingVelocity()

    x_init = [sx0, sy0, steerAng0, v0, carAng0]
    xcovar_init = [1e-9]*(stateLen**2)

    smallEpsilon = 1e-6
    smallRho = 0.25 #Since system is treated as discrete, even though discrete positions at 0,...,N might not be hitting obstacle, the 
                    #plots interpolate these points and sometimes the interpolating lines look like they're going through the obstacle. 
                    #Add small buffer around obstacle to remove this effect.

    sxRobustness = 0
    syRobustness = 0
    steerAngRobustness = -0 
    vRobustness = 0
    carAngRobustness = 0

    M = 4 # RK4 steps per interval
    DT = T/N/M

    #-----Debug-----#
    xref = SX.sym('xref', stateLen)
    #-----End Debug-----#

    # Objective term
    L = (u[0])**2 + (u[1])**2 
    L_terminal = x[4]**2
    cost = Function('cost', [x, xref, u], [L])

    terminal_cost = Function('terminal_cost', [x, u], [L_terminal])

    det_cost = Function('det_cost', [x, u], [L])

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
    w0 += [stateTrajRef[0, :]]
    discrete += [False]*stateLen

    #Integer variables for STL 
    Zavoid_0 = MX.sym('Zavoid_0', 3)
    w += [Zavoid_0]
    lbw += [0]*3
    ubw += [1]*3
    w0 += [1]*3
    discrete += [True]*3

    Zreach_0 = MX.sym('Zreach_0', 1)
    w += [Zreach_0]
    lbw += [0]*1
    ubw += [1]*1
    w0 += [0]*1
    discrete += [True]*1

    eventuallyAlwaysList = [Zreach_0]

    # STL constraints
    # avoid
    g += [obstacle_polygon_x[0] - X0[0] - smallRho + sxRobustness + bigMx*(1 - Zavoid_0[0])] 
    g += [X0[0] - obstacle_polygon_x[1] - smallRho + sxRobustness + bigMx*(1 - Zavoid_0[1])] 
    g += [X0[1] - obstacle_polygon_y[1] - smallRho + syRobustness + bigMy*(1 - Zavoid_0[2])] 
    
    g += [Zavoid_0[0] + Zavoid_0[1] + Zavoid_0[2]]

    lbg += [0, 0, 0, 1]
    ubg += [2*bigMx, 2*bigMx, 2*bigMy, 3]

    # # #reach
    g += [X0[0] - goal_A_polygon_x[0] + sxRobustness + bigMx*(1 - Zreach_0)] 
    g += [goal_A_polygon_x[1] - X0[0] + sxRobustness + bigMx*(1 - Zreach_0)] 
    g += [goal_A_polygon_y[1] - X0[1] + syRobustness + bigMy*(1 - Zreach_0)] 
    g += [X0[1] - goal_A_polygon_y[0] + syRobustness + bigMy*(1 - Zreach_0)]
    g += [X0[3] - goal_min_speed + vRobustness + bigMv*(1 - Zreach_0)]
    g += [X0[4] - goal_carAng[0] + carAngRobustness + bigMcarAng*(1 - Zreach_0)]
    g += [goal_carAng[1] - X0[4] + carAngRobustness + bigMcarAng*(1 - Zreach_0)]

    lbg += [0, 0, 0, 0, 0, 0, 0]
    ubg += [2*bigMx, 2*bigMx, 2*bigMy, 2*bigMy, 2*bigMv, 2*bigMcarAng, 2*bigMcarAng]

    # envelope
    g += [X0[1] - ylim[0]]
    g += [ylim[1] - X0[1]]

    lbg += [0, 0]
    ubg += [2*bigMy, 2*bigMy]


    # Formulate the NLP
    Xk = X0

    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), controlLen)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += [controlTrajRef[k, :]]
        discrete += [False]*controlLen

        # Acceleration limits
        g += [ubu[1] * v_switch/(log(exp(v_switch) + exp(Xk[3]))) - Uk[1]]
        
        lbg += [0]
        ubg += [ubu[1] - lbu[1]]

        # Integrate till the end of the interval
        car_dynamics_k = car_sys.getDiscreteLTVDynamicsFor(k, T/N, M, stateTrajRef[k, :], controlTrajRef[k, :], [], False)
        Fk = car_dynamics_k(x=Xk, u=Uk)
        Xk_end = Fk['xf_ltv']
        J=J+det_cost(Xk, Uk)
        print("___xk end___")
        print(Xk_end)

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), stateLen)
        w   += [Xk]
        lbw += lbx
        ubw += ubx
        w0  += [stateTrajRef[k+1, :]]
        discrete += [False]*stateLen

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*stateLen
        ubg += [0]*stateLen

        #Integer variables for STL 
        Zavoid_k = MX.sym('Zavoid_' + str(k+1), 3)
        w += [Zavoid_k]
        lbw += [0]*3
        ubw += [1]*3
        w0 += [1]*3
        discrete += [True]*3

        Zreach_k = MX.sym('Zreach_' + str(k+1), 1)
        w += [Zreach_k]
        lbw += [0]*1
        ubw += [1]*1
        w0 += [0]*1
        discrete += [True]*1

        eventuallyAlwaysList += [Zreach_k]

        # STL constraints
        # avoid
        g += [obstacle_polygon_x[0] - Xk[0] - smallRho + sxRobustness + bigMx*(1 - Zavoid_k[0])] 
        g += [Xk[0] - obstacle_polygon_x[1] - smallRho + sxRobustness + bigMx*(1 - Zavoid_k[1])] 
        g += [Xk[1] - obstacle_polygon_y[1] - smallRho + syRobustness + bigMy*(1 - Zavoid_k[2])] 
    
        g += [Zavoid_k[0] + Zavoid_k[1] + Zavoid_k[2]]
    
        lbg += [0, 0, 0, 1]
        ubg += [2*bigMx, 2*bigMx, 2*bigMy, 3]

        # reach
        for j in range(k+2):
            g += [Xk[0] - goal_A_polygon_x[0] + sxRobustness +  bigMx*(1 - eventuallyAlwaysList[j])] 
            g += [goal_A_polygon_x[1] - Xk[0] + sxRobustness + bigMx*(1 - eventuallyAlwaysList[j])] 
            g += [goal_A_polygon_y[1] - Xk[1] + syRobustness + bigMy*(1 - eventuallyAlwaysList[j])] 
            g += [Xk[1] - goal_A_polygon_y[0] + syRobustness + bigMy*(1 - eventuallyAlwaysList[j])]
            g += [Xk[3] - goal_min_speed + vRobustness + bigMv*(1 - eventuallyAlwaysList[j])]
            g += [Xk[4] - goal_carAng[0] + carAngRobustness + bigMcarAng*(1 - eventuallyAlwaysList[j])]
            g += [goal_carAng[1] - Xk[4] + carAngRobustness + bigMcarAng*(1 - eventuallyAlwaysList[j])]

            lbg += [0, 0, 0, 0, 0, 0, 0]
            ubg += [2*bigMx, 2*bigMx, 2*bigMy, 2*bigMy, 2*bigMv, 2*bigMcarAng, 2*bigMcarAng]

        # envelope
        g += [Xk[1] - ylim[0]]
        g += [ylim[1] - Xk[1]]

        lbg += [0, 0]
        ubg += [2*bigMy, 2*bigMy]
        
    eventuallyAlways = 0
    for k in range(N+1):
        eventuallyAlways += eventuallyAlwaysList[k]

    g += [eventuallyAlways]
    
    lbg += [1]
    ubg += [N+1]

    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # J += terminal_cost(Xk, Uk)

    # Create an NLP solver
    qp_prob = {'f': J, 'x': w, 'g': g}
    qp_solver = qpsol('qp_solver', 'gurobi', qp_prob, {"discrete": discrete, "error_on_fail": False})

    # Solve the NLP
    sol = qp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(qp_solver.stats())

    solver_stats = qp_solver.stats()
    proc_runtime = solver_stats['t_wall_solver']
    feasible = solver_stats['success']

    w1_opt = sol['x']
    
    state_opt = []
    control_opt = []

    numBinVars = 4
    varPerTimeStep = stateLen + numBinVars + controlLen 
    stateRelatedVarPerTimeStep = stateLen + numBinVars

    # feasible = np.any(w1_opt.full().flatten())
    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][0::varPerTimeStep]
        y_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][1::varPerTimeStep]
        steerAng_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][2::varPerTimeStep]
        v_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][3::varPerTimeStep]
        carAng_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][4::varPerTimeStep]
        zReach_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][varPerTimeStep-3::varPerTimeStep]
        vSteerAng_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][varPerTimeStep-2::varPerTimeStep]
        accel_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][varPerTimeStep-1::varPerTimeStep]

        state_opt = np.transpose(np.concatenate([[x_opt], [y_opt], [steerAng_opt], [v_opt], [carAng_opt]], axis=0))
        # print(state_opt)
        control_opt = np.transpose(np.concatenate([[vSteerAng_opt], [accel_opt]], axis=0))
        # print(control_opt)
        manual_objective = np.sum([np.square(accel_opt), 10*np.square(vSteerAng_opt)])
        # print("objective (manually tallied): ", manual_objective)

        if plot == True:
            plotSol(N, x_opt, y_opt, steerAng_opt, v_opt, carAng_opt, vSteerAng_opt, accel_opt)
            plotTraj(x_opt, y_opt, goal_A_polygon_x=goal_A_polygon_x, goal_A_polygon_y=goal_A_polygon_y, obstacle_polygon_x=obstacle_polygon_x, obstacle_polygon_y=obstacle_polygon_y)

    return proc_runtime, feasible, state_opt, control_opt

def solveLTV(plot, onlineCovar, T, N, car, stateTrajRef, controlTrajRef, stateCovarTrajRef, sx0, sy0, steerAng0, v0, carAng0, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y):
    feasible = False

    # Declare model variables
    x = car.getStateVar()
    u = car.getInputVar()

    stateLen = x.size()[0]
    stateCovarLen = stateLen**2 
    controlLen = u.size()[0]

    xlim, ylim, steerAnglim, vlim, carAngLim = car.getStateLimits()
    stateCovarLowerLim, stateCovarUpperLim = car.getStateCovarLimits()
    aLim, vSteerAngLim = car.getInputLimits()

    lbx = [xlim[0], ylim[0], steerAnglim[0], vlim[0], carAngLim[0]]
    ubx = [xlim[1], ylim[1], steerAnglim[1], vlim[1], carAngLim[1]]

    lbxCovar = stateCovarLowerLim
    ubxCovar = stateCovarUpperLim

    lbu = [aLim[0], vSteerAngLim[0]]
    ubu = [aLim[1], vSteerAngLim[1]]

    v_switch = car.getSwitchingVelocity()

    x_init = [sx0, sy0, steerAng0, v0, carAng0]
    xcovar_init = [1e-9]*stateCovarLen

    invCDFVarphiEpsilonRef1 = norm.ppf(0.15)
    invCDFVarphiEpsilonRef2 = norm.ppf(0.35)
    invCDFVarphiEpsilon = invCDFVarphiEpsilonRef1
    invCDFVarphiEpsilonCarAngRef1 = norm.ppf(0.15)
    invCDFVarphiEpsilonCarAngRef2 = norm.ppf(0.25)
    invCDFVarphiEpsilonCarAng = invCDFVarphiEpsilonCarAngRef1
    uncertaintyLookaheadN = 50
    uncertaintyLookaheadN2 = 20
    smallEpsilon = 1e-5
    smallRho = 0.25 #Since system is treated as discrete, even though discrete positions at 0,...,N might not be hitting obstacle, the 
                    #plots interpolate these points and sometimes the interpolating lines look like they're going through the obstacle. 
                    #Add small buffer around obstacle to remove this effect.

    M = 4 # RK4 steps per interval
    DT = T/N/M

    #-----Debug-----#
    xref = SX.sym('xref', stateLen)
    #-----End Debug-----#

    # Objective term
    L = (u[0])**2 + (u[1])**2 
    L_terminal = x[4]**2
    cost = Function('cost', [x, u], [L])

    terminal_cost = Function('terminal_cost', [x, u], [L_terminal])

    det_cost = Function('det_cost', [x, u], [L])

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
    w0 += [stateTrajRef[0, :]]
    discrete += [False]*stateLen

    if onlineCovar == True:
        XCOVAR0 = MX.sym('XCOVAR0', stateCovarLen)
        w += [XCOVAR0]
        lbw += xcovar_init
        ubw += xcovar_init
        w0 += [stateCovarTrajRef[0, :]]
        discrete += [False]*stateCovarLen

    #Integer variables for STL 
    Zavoid_0 = MX.sym('Zavoid_0', 3)
    w += [Zavoid_0]
    lbw += [0]*3
    ubw += [1]*3
    w0 += [1]*3
    discrete += [True]*3

    Zreach_0 = MX.sym('Zreach_0', 1)
    w += [Zreach_0]
    lbw += [0]*1
    ubw += [1]*1
    w0 += [0]*1
    discrete += [True]*1

    eventuallyAlwaysList = [Zreach_0]

    if N == 8:
        print("test")
    # STL constraints
    # avoid
    if onlineCovar == True:
        g += [obstacle_polygon_x[0] - X0[0] - smallRho + invCDFVarphiEpsilon * sqrt(XCOVAR0[0] + smallEpsilon) + bigMx*(1 - Zavoid_0[0])] 
        g += [X0[0] - obstacle_polygon_x[1] - smallRho + invCDFVarphiEpsilon * sqrt(XCOVAR0[0] + smallEpsilon) + bigMx*(1 - Zavoid_0[1])] 
        g += [X0[1] - obstacle_polygon_y[1] - smallRho + invCDFVarphiEpsilon * sqrt(XCOVAR0[6] + smallEpsilon) + bigMy*(1 - Zavoid_0[2])]
    else:
        g += [obstacle_polygon_x[0] - X0[0] - smallRho + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 0]) + bigMx*(1 - Zavoid_0[0])] 
        g += [X0[0] - obstacle_polygon_x[1] - smallRho + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 0]) + bigMx*(1 - Zavoid_0[1])] 
        g += [X0[1] - obstacle_polygon_y[1] - smallRho + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 6]) + bigMy*(1 - Zavoid_0[2])] 
    
    g += [Zavoid_0[0] + Zavoid_0[1] + Zavoid_0[2]]

    lbg += [0, 0, 0, 1]
    ubg += [2*bigMx, 2*bigMx, 2*bigMy, 3]

    # # #reach
    if onlineCovar == True:
        g += [X0[0] - goal_A_polygon_x[0] + invCDFVarphiEpsilon * sqrt(XCOVAR0[0] + smallEpsilon) + bigMx*(1 - Zreach_0)] 
        g += [goal_A_polygon_x[1] - X0[0] + invCDFVarphiEpsilon * sqrt(XCOVAR0[0] + smallEpsilon) + bigMx*(1 - Zreach_0)] 
        g += [goal_A_polygon_y[1] - X0[1] + invCDFVarphiEpsilon * sqrt(XCOVAR0[6] + smallEpsilon) + bigMy*(1 - Zreach_0)] 
        g += [X0[1] - goal_A_polygon_y[0] + invCDFVarphiEpsilon * sqrt(XCOVAR0[6] + smallEpsilon) + bigMy*(1 - Zreach_0)]
        g += [X0[3] - goal_min_speed + invCDFVarphiEpsilon * sqrt(XCOVAR0[18] + smallEpsilon) + bigMv*(1 - Zreach_0)]
        g += [X0[4] - goal_carAng[0] + invCDFVarphiEpsilonCarAng * sqrt(XCOVAR0[24] + smallEpsilon) + bigMcarAng*(1 - Zreach_0)]
        g += [goal_carAng[1] - X0[4] + invCDFVarphiEpsilonCarAng * sqrt(XCOVAR0[24] + smallEpsilon) + bigMcarAng*(1 - Zreach_0)]
    else:
        g += [X0[0] - goal_A_polygon_x[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 0]) + bigMx*(1 - Zreach_0)] 
        g += [goal_A_polygon_x[1] - X0[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 0]) + bigMx*(1 - Zreach_0)] 
        g += [goal_A_polygon_y[1] - X0[1] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 6]) + bigMy*(1 - Zreach_0)] 
        g += [X0[1] - goal_A_polygon_y[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 6]) + bigMy*(1 - Zreach_0)]
        g += [X0[3] - goal_min_speed + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 18]) + bigMv*(1 - Zreach_0)]
        g += [X0[4] - goal_carAng[0] + invCDFVarphiEpsilonCarAng * sqrt(stateCovarTrajRef[0, 24]) + bigMcarAng*(1 - Zreach_0)]
        g += [goal_carAng[1] - X0[4] + invCDFVarphiEpsilonCarAng * sqrt(stateCovarTrajRef[0, 24]) + bigMcarAng*(1 - Zreach_0)]

    lbg += [0, 0, 0, 0, 0, 0, 0]
    ubg += [2*bigMx, 2*bigMx, 2*bigMy, 2*bigMy, 2*bigMv, 2*bigMcarAng, 2*bigMcarAng]

    # envelope
    if onlineCovar == True:
        g += [X0[1] - ylim[0] + invCDFVarphiEpsilon * sqrt(XCOVAR0[6] + smallEpsilon)]
        g += [ylim[1] - X0[1] + invCDFVarphiEpsilon * sqrt(XCOVAR0[6] + smallEpsilon)]
    else:
        g += [X0[1] - ylim[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 6])]
        g += [ylim[1] - X0[1] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[0, 6])]

    lbg += [0, 0]
    ubg += [2*bigMy, 2*bigMy]


    # Formulate the NLP
    Xk = X0
    if onlineCovar == True:
        XCOVARk = XCOVAR0   

    for k in range(N):
        if onlineCovar: 
            if k < uncertaintyLookaheadN:
                invCDFVarphiEpsilon = invCDFVarphiEpsilonRef1
                invCDFVarphiEpsilonCarAng = invCDFVarphiEpsilonCarAngRef1
            else:
                invCDFVarphiEpsilon = 0
                invCDFVarphiEpsilonCarAng = 0
        else: 
            invCDFVarphiEpsilon = invCDFVarphiEpsilonRef1

        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), controlLen)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += [controlTrajRef[k, :]]
        discrete += [False]*controlLen

        # Acceleration limits
        g += [ubu[1] * v_switch/(log(exp(v_switch) + exp(Xk[3]))) - Uk[1]]
        
        lbg += [0]
        ubg += [ubu[1] - lbu[1]]

        # Integrate till the end of the interval
        car_dynamics_k = car_sys.getDiscreteLTVDynamicsFor(k, T/N, M, stateTrajRef[k, :], controlTrajRef[k, :], stateCovarTrajRef[k, :], True)
        if onlineCovar == True: 
            Fk = car_dynamics_k(x=Xk, u=Uk, xcovar=XCOVARk)
            Xk_end = Fk['xf_ltv']
            XCOVARk_end = Fk['xcovarf_ltv']
            J=J+cost(Xk, Uk)
            print("___xk end___")
            print(Xk_end)
            print("__xcovar end___")
            print(XCOVARk_end)
        else:
            Fk = car_dynamics_k(x=Xk, u=Uk, xcovar=DM(stateCovarTrajRef[k, :]))
            Xk_end = Fk['xf_ltv']
            J=J+det_cost(Xk, Uk)
            print("___xk end___")
            print(Xk_end)

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), stateLen)
        w   += [Xk]
        lbw += lbx
        ubw += ubx
        w0  += [stateTrajRef[k+1, :]]
        discrete += [False]*stateLen

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*stateLen
        ubg += [0]*stateLen

        if onlineCovar == True:
            XCOVARk = MX.sym('XCOVARk' + str(k+1), stateCovarLen)
            w += [XCOVARk]
            lbw += lbxCovar
            ubw += ubxCovar
            w0 += [stateCovarTrajRef[k+1, :]]
            discrete += [False]*stateCovarLen

            g += [XCOVARk_end - XCOVARk]
            lbg += [0]*stateCovarLen
            ubg += [0]*stateCovarLen

        #Integer variables for STL 
        Zavoid_k = MX.sym('Zavoid_' + str(k+1), 3)
        w += [Zavoid_k]
        lbw += [0]*3
        ubw += [1]*3
        w0 += [1]*3
        discrete += [True]*3

        Zreach_k = MX.sym('Zreach_' + str(k+1), 1)
        w += [Zreach_k]
        lbw += [0]*1
        ubw += [1]*1
        w0 += [0]*1
        discrete += [True]*1

        eventuallyAlwaysList += [Zreach_k]

        # STL constraints
        # avoid
        if onlineCovar == True:
            g += [obstacle_polygon_x[0] - Xk[0] - smallRho + invCDFVarphiEpsilon * sqrt(XCOVARk[0] + smallEpsilon) + bigMx*(1 - Zavoid_k[0])] 
            g += [Xk[0] - obstacle_polygon_x[1] - smallRho + invCDFVarphiEpsilon * sqrt(XCOVARk[0] + smallEpsilon) + bigMx*(1 - Zavoid_k[1])] 
            g += [Xk[1] - obstacle_polygon_y[1] - smallRho + invCDFVarphiEpsilon * sqrt(XCOVARk[6] + smallEpsilon) + bigMy*(1 - Zavoid_k[2])]
        else:
            g += [obstacle_polygon_x[0] - Xk[0] - smallRho + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 0]) + bigMx*(1 - Zavoid_k[0])] 
            g += [Xk[0] - obstacle_polygon_x[1] - smallRho + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 0]) + bigMx*(1 - Zavoid_k[1])] 
            g += [Xk[1] - obstacle_polygon_y[1] - smallRho + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 6]) + bigMy*(1 - Zavoid_k[2])] 
        
        g += [Zavoid_k[0] + Zavoid_k[1] + Zavoid_k[2]]
    
        lbg += [0, 0, 0, 1]
        ubg += [2*bigMx, 2*bigMx, 2*bigMy, 3]

        # reach
        for j in range(k+2):
            if onlineCovar == True:
                g += [Xk[0] - goal_A_polygon_x[0] + invCDFVarphiEpsilon * sqrt(XCOVARk[0] + smallEpsilon) + bigMx*(1 - eventuallyAlwaysList[j])] 
                g += [goal_A_polygon_x[1] - Xk[0] + invCDFVarphiEpsilon * sqrt(XCOVARk[0] + smallEpsilon) + bigMx*(1 - eventuallyAlwaysList[j])] 
                g += [goal_A_polygon_y[1] - Xk[1] + invCDFVarphiEpsilon * sqrt(XCOVARk[6] + smallEpsilon) + bigMy*(1 - eventuallyAlwaysList[j])] 
                g += [Xk[1] - goal_A_polygon_y[0] + invCDFVarphiEpsilon * sqrt(XCOVARk[6] + smallEpsilon) + bigMy*(1 - eventuallyAlwaysList[j])]
                g += [Xk[3] - goal_min_speed + invCDFVarphiEpsilon * sqrt(XCOVARk[18] + smallEpsilon) + bigMv*(1 - eventuallyAlwaysList[j])]
                g += [Xk[4] - goal_carAng[0] + invCDFVarphiEpsilonCarAng * sqrt(XCOVARk[24] + smallEpsilon) + bigMcarAng*(1 - eventuallyAlwaysList[j])]
                g += [goal_carAng[1] - Xk[4] + invCDFVarphiEpsilonCarAng * sqrt(XCOVARk[24] + smallEpsilon) + bigMcarAng*(1 - eventuallyAlwaysList[j])]
            else:
                g += [Xk[0] - goal_A_polygon_x[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 0]) + bigMx*(1 - eventuallyAlwaysList[j])] 
                g += [goal_A_polygon_x[1] - Xk[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 0]) + bigMx*(1 - eventuallyAlwaysList[j])] 
                g += [goal_A_polygon_y[1] - Xk[1] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 6]) + bigMy*(1 - eventuallyAlwaysList[j])] 
                g += [Xk[1] - goal_A_polygon_y[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 6]) + bigMy*(1 - eventuallyAlwaysList[j])]
                g += [Xk[3] - goal_min_speed + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 18]) + bigMv*(1 - eventuallyAlwaysList[j])]
                g += [Xk[4] - goal_carAng[0] + invCDFVarphiEpsilonCarAng * sqrt(stateCovarTrajRef[k, 24]) + bigMcarAng*(1 - eventuallyAlwaysList[j])]
                g += [goal_carAng[1] - Xk[4] + invCDFVarphiEpsilonCarAng * sqrt(stateCovarTrajRef[k, 24]) + bigMcarAng*(1 - eventuallyAlwaysList[j])]

            lbg += [0, 0, 0, 0, 0, 0, 0]
            ubg += [2*bigMx, 2*bigMx, 2*bigMy, 2*bigMy, 2*bigMv, 2*bigMcarAng, 2*bigMcarAng]

        # envelope
        if onlineCovar == True:
            g += [Xk[1] - ylim[0] + invCDFVarphiEpsilon * sqrt(XCOVARk[6] + smallEpsilon)]
            g += [ylim[1] - Xk[1] + invCDFVarphiEpsilon * sqrt(XCOVARk[6] + smallEpsilon)]
        else:
            g += [Xk[1] - ylim[0] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 6])]
            g += [ylim[1] - Xk[1] + invCDFVarphiEpsilon * sqrt(stateCovarTrajRef[k, 6])]

        lbg += [0, 0]
        ubg += [2*bigMy, 2*bigMy]
            
    eventuallyAlways = 0
    for k in range(N+1):
        eventuallyAlways += eventuallyAlwaysList[k]

    g += [eventuallyAlways]
    
    lbg += [1]
    ubg += [N+1]

    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # J += terminal_cost(Xk, Uk)

    bonmin_options = {"node_limit": 300000}
    # Create an NLP solver
    qp_prob = {'f': J, 'x': w, 'g': g}
    qp_solver = qpsol('qp_solver', 'gurobi', qp_prob, {"discrete": discrete, "error_on_fail": False})

    # Solve the NLP
    sol = qp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(qp_solver.stats())

    solver_stats = qp_solver.stats()
    proc_runtime = solver_stats['t_wall_solver']
    feasible = solver_stats['success']

    w1_opt = sol['x']
    lam_w_opt = sol['lam_x']
    lam_g_opt = sol['lam_g']

    u1_ret = 0
    u2_ret = 0
    x1_ret = np.array([])
    x2_ret = np.array([])
    
    stateCovar_opt = []
    state_opt = []
    control_opt = []

    numBinVars = 4
    varPerTimeStep = stateLen + numBinVars + controlLen 
    stateRelatedVarPerTimeStep = stateLen + numBinVars
    if onlineCovar == True:
        varPerTimeStep = stateLen + stateCovarLen + numBinVars + controlLen 
        stateRelatedVarPerTimeStep = stateLen + stateCovarLen

    # feasible = np.any(w1_opt.full().flatten())
    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][0::varPerTimeStep]
        y_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][1::varPerTimeStep]
        steerAng_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][2::varPerTimeStep]
        v_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][3::varPerTimeStep]
        carAng_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][4::varPerTimeStep]
        if onlineCovar == True:
            xCovar_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][stateLen::varPerTimeStep]
            yCovar_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][stateLen + 6::varPerTimeStep]
            steerAngCovar_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][stateLen + 12::varPerTimeStep]
            vCovar_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][stateLen + 18::varPerTimeStep]
            carAngCovar_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][stateLen + 24::varPerTimeStep]
        zReach_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][varPerTimeStep-3::varPerTimeStep]
        vSteerAng_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][varPerTimeStep-2::varPerTimeStep]
        accel_opt = w1_opt[0:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][varPerTimeStep-1::varPerTimeStep]

        # if onlineCovar == True:
        #     print("Xcovar opt _____")
        #     print(xCovar_opt)
        #     print("yCovar opt _____")
        #     print(yCovar_opt)
        #     print("vcovar opt _____")
        #     print(vCovar_opt)
        #     print("carAngCovar opt _____")
        #     print(carAngCovar_opt)

        state_opt = np.transpose(np.concatenate([[x_opt], [y_opt], [steerAng_opt], [v_opt], [carAng_opt]], axis=0))
        # print(state_opt)
        control_opt = np.transpose(np.concatenate([[vSteerAng_opt], [accel_opt]], axis=0))
        # print(control_opt)
        if onlineCovar:
            stateCovar_opt = []
            for i in range(N+1):
                stateCovar_opt += [w1_opt[varPerTimeStep*i:varPerTimeStep*(N)+stateRelatedVarPerTimeStep][stateLen:stateLen+stateCovarLen]]
            stateCovar_opt = np.vstack(stateCovar_opt)
        else:
            stateCovarTrajSim = generateCovariancePredictions(T/N, M, car, state_opt, control_opt)
            stateCovar_opt = stateCovarTrajSim
        manual_objective = np.sum([np.square(accel_opt), 10*np.square(vSteerAng_opt)])
        print("objective (manually tallied): ", manual_objective)

        if plot == True:
            plotSol(N, x_opt, y_opt, steerAng_opt, v_opt, carAng_opt, vSteerAng_opt, accel_opt)
            plotTraj(x_opt, y_opt, goal_A_polygon_x=goal_A_polygon_x, goal_A_polygon_y=goal_A_polygon_y, obstacle_polygon_x=obstacle_polygon_x, obstacle_polygon_y=obstacle_polygon_y)

    return proc_runtime, feasible, state_opt, control_opt, stateCovar_opt

def generateCovariancePredictions(DT, RK_steps, car, stateTraj, controlTraj):
    car_dynamics = car.getDiscreteDynamics(DT, RK_steps, True)
    covarLen = stateTraj.shape[1]**2
    stateCovarPred = [np.zeros((1, covarLen))]
    for k in range(controlTraj.shape[0]):
        sysk = car_dynamics(x0=DM(stateTraj[k, :]), xcovar0=DM([0]*covarLen), u=DM(controlTraj[k, :]))
        xk = sysk['xf']
        xcovark = sysk['xcovarf']
        print("xcovark: ")
        print(reshape(xcovark, (5, 5)).full())
        stateCovarPred += [np.hstack(xcovark.full())]

    stateCovarPred = np.vstack(stateCovarPred)

    return stateCovarPred

def simulateTraj(plot, T, N, RK_steps, car, state0, stateCovar0, controlTraj):
    car_dynamics = car.getDiscreteDynamics(T/N, RK_steps, True)

    # Get a feasible trajectory as an initial guess
    xk = DM(state0)
    x_start = [np.hstack(xk.full())]
    xcovark = DM(stateCovar0)
    xcovar_start = [np.hstack(xcovark.full())]
    for k in range(N):
        sysk = car_dynamics(x0=xk, xcovar0=xcovark, u=DM(controlTraj[:, k]))
        xk = sysk['xf']
        x_start += [np.hstack(xk.full())]
        xcovark = sysk['xcovarf']
        print("xcovark: ")
        print(reshape(xcovark, (5, 5)).full())
        xcovar_start += [np.hstack(xcovark.full())]

    trajSim = np.vstack(x_start)
    trajCovarSim = np.vstack(xcovar_start)

    print("____sim x____")
    print(trajSim)
    print("____sim covar____")
    print(trajCovarSim)
    print("____sim u____")
    print(controlTraj)

    if plot:
        plotTraj(trajSim[:, 0], trajSim[:, 1])

    return trajSim, controlTraj, trajCovarSim

def plotTraj(x, y, x1_openloops = [], x2_openloops = [], goal_A_polygon_x = [], goal_A_polygon_y = [], obstacle_polygon_x = [], obstacle_polygon_y = []):
    
    plt.figure()
    ax = plt.gca() 
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel('x')
    plt.ylabel('y')
    if len(goal_A_polygon_x) > 0:
        goal_A_polygon_x_plot = [goal_A_polygon_x[0], goal_A_polygon_x[0], goal_A_polygon_x[1], goal_A_polygon_x[1]]
        goal_A_polygon_y_plot = [goal_A_polygon_y[1], goal_A_polygon_y[0], goal_A_polygon_y[0], goal_A_polygon_y[1]] 
        obstacle_polygon_x_plot = [obstacle_polygon_x[0], obstacle_polygon_x[0], obstacle_polygon_x[1], obstacle_polygon_x[1]]
        obstacle_polygon_y_plot = [obstacle_polygon_y[1], obstacle_polygon_y[0], obstacle_polygon_y[0], obstacle_polygon_y[1]]
        plt.fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        plt.plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        plt.fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'r', alpha=0.5)
        plt.plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'r')
    plt.scatter(x[0], y[0], s=120, facecolors='none', edgecolors='black')
    ax.set_aspect('equal')
    plt.plot(x, y, '-o')
    print(x)
    print("___")
    print(y)
    for i in range(len(x1_openloops)):
        if i % 2 == 0: #plot open loops only 1 every 5 control intervals for clarity 
            plt.plot(x1_openloops[i], x2_openloops[i], alpha=0.5, linestyle='-', linewidth=1)
    plt.grid(True)

def plotSol(N, x, y, steerAng, v, carAng, vSteerAng, accel):
    tgrid = [k for k in range(N+1)]
    plt.figure()
    plt.plot(tgrid, x, '-')
    plt.plot(tgrid, y, '--')
    plt.plot(tgrid, steerAng, '-')
    plt.plot(tgrid, v, '--')
    plt.plot(tgrid, carAng, '-')
    plt.step(tgrid, vertcat(DM.nan(1), vSteerAng), '-.')
    plt.step(tgrid, vertcat(DM.nan(1), accel), '.')
    plt.xlabel('t')
    plt.legend(['x','y', 'steerAng', 'v', 'carAng', 'vSteerAng', 'accel'])
    plt.grid(True)

def controlLoopNomLTV(plot, writeTraces, T, N, simRes, car_control_model, car_sim_model, stateTrajRef0, controlTrajRef0, x0, y0, steerAng0, v0, carAng0, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y, stateTrajFile = [], controlTrajFile = []):
    car_sim_model.setInitialState(x0, y0, steerAng0, v0, carAng0)

    stateVector = np.array([x0, y0, steerAng0, v0, carAng0])
    closedLoopStates = [stateVector]
    sx_openloops = []
    sy_openloops = []

    closedLoopControlTraj = []
    closedLoopPredictedStates = [stateVector]

    stateTrajRef = stateTrajRef0
    controlTrajRef = controlTrajRef0
    
    totalSolveTime = 0
    maxSolveTime = 0

    num_iters = 0
    for i in range(N):
        if stateVector[0] > obstacle_polygon_x[0] and stateVector[0] < obstacle_polygon_x[1] and stateVector[1] > obstacle_polygon_y[0] and stateVector[1] < obstacle_polygon_y[1]:
            print("Starting state infeasible. Exiting")
            break
    
        stateTraj = []
        controlTraj = []
        feasible = False
        solveTime, feasible, stateTraj, controlTraj = solveNomLTV(False, T-i*(T/N), N-i, car_control_model, stateTrajRef, controlTrajRef, stateVector[0], stateVector[1], stateVector[2], stateVector[3], stateVector[4], goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y)
        
        num_iters += 1 
        totalSolveTime += solveTime
        if solveTime > maxSolveTime:
            maxSolveTime = solveTime

        if feasible == False:
            print("Optimization failed. Exiting")
            break

        sx_openloops += [stateTraj[:, 0]]
        sy_openloops += [stateTraj[:, 1]]

        vSteerAng = controlTraj[0, 0]
        accel = controlTraj[0, 1]
        closedLoopControlTraj += [controlTraj[0, :]]
        car_sim_model.simNext(True, T/N, simRes, vSteerAng, accel)
        
        closedLoopPredictedStates += [stateTraj[1, :]]

        print("current state: ")
        print(stateVector)
        print("control input: ")
        print(controlTraj[0, :])
        print("predicted state traj: ")
        print(stateTraj[1, :])

        nextStateVector = car_sim_model.getSingleTrackStates()
        print("simulated state traj" )
        print(nextStateVector)

        closedLoopStates += [nextStateVector]
        stateVector = nextStateVector

        stateTrajRef = np.concatenate((np.array([stateVector]), stateTraj[2:, :]))
        controlTrajRef = controlTraj[1:, :]
        

    closedLoopStates = np.vstack(closedLoopStates)
    if plot == True:
        plotTraj(closedLoopStates[:, 0], closedLoopStates[:, 1], sx_openloops, sy_openloops, goal_A_polygon_x=goal_A_polygon_x, goal_A_polygon_y=goal_A_polygon_y, obstacle_polygon_x=obstacle_polygon_x, obstacle_polygon_y=obstacle_polygon_y)
    
    if len(closedLoopControlTraj) > 0:
        closedLoopControlTraj = np.vstack(closedLoopControlTraj)
    else:
        closedLoopControlTraj = np.array([])

    closedLoopPredictedStates = np.vstack(closedLoopPredictedStates)

    if num_iters > 0:
        avgSolveTime = totalSolveTime/num_iters
    else:
        avgSolveTime = -1

    if writeTraces:
        for i in range(closedLoopStates.shape[0]):
            stateTrajFile.write(str(closedLoopStates[i, 0]) + "," + str(closedLoopStates[i, 1]) + "," + str(closedLoopStates[i, 2]) + "," + str(closedLoopStates[i, 3]) + "," + str(closedLoopStates[i, 4]) + "\n")
        stateTrajFile.write("~\n")
        for i in range(closedLoopControlTraj.shape[0]):
            controlTrajFile.write(str(closedLoopControlTraj[i, 0]) + "," + str(closedLoopControlTraj[i, 1]) + "\n")
        controlTrajFile.write("~\n")

    return avgSolveTime, maxSolveTime, totalSolveTime, num_iters, feasible, closedLoopControlTraj, closedLoopPredictedStates, closedLoopStates

def controlLoopLTV(plot, useGP, onlineCovar, writeTraces, T, N, simRes, car_control_model, car_sim_model, stateTrajRef0, controlTrajRef0, stateCovarTrajRef0, x0, y0, steerAng0, v0, carAng0, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y, stateTrajFile = [], controlTrajFile = []):  
    car_sim_model.setInitialState(x0, y0, steerAng0, v0, carAng0)

    stateVector = np.array([x0, y0, steerAng0, v0, carAng0])
    closedLoopStates = [stateVector]
    sx_openloops = []
    sy_openloops = []

    closedLoopControlTraj = []
    closedLoopPredictedStates = [stateVector]

    stateTrajRef = stateTrajRef0
    controlTrajRef = controlTrajRef0
    stateCovarTrajRef = stateCovarTrajRef0
    
    totalSolveTime = 0
    maxSolveTime = 0

    num_iters = 0

    for i in range(N):
        if stateVector[0] > obstacle_polygon_x[0] and stateVector[0] < obstacle_polygon_x[1] and stateVector[1] > obstacle_polygon_y[0] and stateVector[1] < obstacle_polygon_y[1]:
            print("starting state infeasible. Exiting")
            break

        stateTraj = []
        controlTraj = []
        feasible = False
        solveTime, feasible, stateTraj, controlTraj, stateCovarTraj = solveLTV(False, onlineCovar, T-i*(T/N), N-i, car_control_model, stateTrajRef, controlTrajRef, stateCovarTrajRef, stateVector[0], stateVector[1], stateVector[2], stateVector[3], stateVector[4], goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y)
        
        num_iters += 1 
        totalSolveTime += solveTime
        if solveTime > maxSolveTime:
            maxSolveTime = solveTime

        if feasible == False:
            print("Optimization failed. Exiting")
            break

        sx_openloops += [stateTraj[:, 0]]
        sy_openloops += [stateTraj[:, 1]]

        vSteerAng = controlTraj[0, 0]
        accel = controlTraj[0, 1]
        closedLoopControlTraj += [controlTraj[0, :]]
        car_sim_model.simNext(True, T/N, simRes, vSteerAng, accel)

        closedLoopPredictedStates += [stateTraj[1, :]]

        print("current state: ")
        print(stateVector)
        print("control input: ")
        print(controlTraj[0, :])
        print("predicted state traj: ")
        print(stateTraj[1, :])

        nextStateVector = car_sim_model.getSingleTrackStates()
        print("simulated state traj" )
        print(nextStateVector)

        closedLoopStates += [nextStateVector]
        stateVector = nextStateVector

        stateTrajRef = np.concatenate((np.array([stateVector]), stateTraj[2:, :]), axis=0)
        controlTrajRef = controlTraj[1:, :]
        stateCovarTrajRef = np.concatenate((np.array([[1e-9]*(stateCovarTrajRef.shape[1])]), stateCovarTraj[2:, :]))
       

    closedLoopStates = np.vstack(closedLoopStates)
    if plot == True:
        plotTraj(closedLoopStates[:, 0], closedLoopStates[:, 1], sx_openloops, sy_openloops, goal_A_polygon_x=goal_A_polygon_x, goal_A_polygon_y=goal_A_polygon_y, obstacle_polygon_x=obstacle_polygon_x, obstacle_polygon_y=obstacle_polygon_y)
    
    if len(closedLoopControlTraj) > 0:
        closedLoopControlTraj = np.vstack(closedLoopControlTraj)
    else:
        closedLoopControlTraj = np.array([])

    closedLoopPredictedStates = np.vstack(closedLoopPredictedStates)

    if num_iters > 0:
        avgSolveTime = totalSolveTime/num_iters
    else:
        avgSolveTime = -1

    if writeTraces:
        for i in range(closedLoopStates.shape[0]):
            stateTrajFile.write(str(closedLoopStates[i, 0]) + "," + str(closedLoopStates[i, 1]) + "," + str(closedLoopStates[i, 2]) + "," + str(closedLoopStates[i, 3]) + "," + str(closedLoopStates[i, 4]) + "\n")
        stateTrajFile.write("~\n")
        for i in range(closedLoopControlTraj.shape[0]):
            controlTrajFile.write(str(closedLoopControlTraj[i, 0]) + "," + str(closedLoopControlTraj[i, 1]) + "\n")
        controlTrajFile.write("~\n")

    return avgSolveTime, maxSolveTime, totalSolveTime, num_iters, feasible, closedLoopControlTraj, closedLoopPredictedStates, closedLoopStates

def controlLoopSetpoint(plot, T, N, simRes, car_control_model, car_sim_model, x0, y0, steerAng0, v0, carAng0, xN, yN, steerAngN, vN, carAngN):
    dT = T/N

    car_sim_model.setInitialState(x0, y0, steerAng0, v0, carAng0)

    stateVector = np.array([x0, y0, steerAng0, v0, carAng0])
    closedLoopStates = [stateVector]
    sx_openloops = []
    sy_openloops = []

    closedLoopControlTraj = []

    for i in range(N):
        stateTraj = []
        controlTraj = []
        feasible = False
        _, feasible, stateTraj, controlTraj = solveSetpointControl(False, T-i*(T/N), N-i, car_control_model, stateVector[0], stateVector[1], stateVector[2], stateVector[3], stateVector[4], xN, yN, steerAngN, vN, carAngN)
        
        if feasible == False:
            print("Optimization failed. Exiting")
            break       
        
        sx_openloops += [stateTraj[0, :]]
        sy_openloops += [stateTraj[1, :]]

        vSteerAng = controlTraj[0, 0]
        accel = controlTraj[1, 0]
        closedLoopControlTraj += [controlTraj[:, 0]]
        car_sim_model.simNext(True, T/N, simRes, vSteerAng, accel)
        
        print("current state: ")
        print(stateVector)
        print("control input: ")
        print(controlTraj[:, 0])
        print("predicted state traj: ")
        print(stateTraj[:, 1])

        nextStateVector = car_sim_model.getSingleTrackStates()
        print("simulated state traj" )
        print(nextStateVector)

        closedLoopStates += [nextStateVector]
        stateVector = nextStateVector

    closedLoopStates = np.vstack(closedLoopStates)
    if plot == True:
        plotTraj(closedLoopStates[:, 0], closedLoopStates[:, 1], sx_openloops, sy_openloops)
    
    closedLoopControlTraj = np.vstack(closedLoopControlTraj)

    return closedLoopControlTraj, closedLoopStates

def controlLoop(plot, controlCost, writeTraces, T, N, simRes, car_control_model, car_sim_model, x0, y0, steerAng0, v0, carAng0, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y, stateTrajFile = [], controlTrajFile = []):
    dT = T/N

    car_sim_model.setInitialState(x0, y0, steerAng0, v0, carAng0)

    stateVector = np.array([x0, y0, steerAng0, v0, carAng0])
    closedLoopStates = [stateVector]
    sx_openloops = []
    sy_openloops = []

    closedLoopControlTraj = []
    closedLoopPredictedStates = [stateVector]

    totalSolveTime = 0
    maxSolveTime = 0

    numIters = 0
    for i in range(N):
        stateTraj = []
        controlTraj = []
        feasible = False

        if stateVector[0] > obstacle_polygon_x[0] and stateVector[0] < obstacle_polygon_x[1] and stateVector[1] > obstacle_polygon_y[0] and stateVector[1] < obstacle_polygon_y[1]:
            print("Starting state infeasible. Exiting")
            break

        solveTime, feasible, stateTraj, controlTraj = solveSmoothedRobustnessNLP(False, controlCost, T-i*(T/N), N-i, car_control_model, stateVector[0], stateVector[1], stateVector[2], stateVector[3], stateVector[4], goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y)

        numIters += 1
        totalSolveTime += solveTime
        if solveTime > maxSolveTime:
            maxSolveTime = solveTime

        if feasible == False:
            print("Optimization failed. Exiting")
            break       
        
        sx_openloops += [stateTraj[0, :]]
        sy_openloops += [stateTraj[1, :]]

        vSteerAng = controlTraj[0, 0]
        accel = controlTraj[1, 0]
        closedLoopControlTraj += [controlTraj[:, 0]]
        car_sim_model.simNext(True, T/N, simRes, vSteerAng, accel)
        
        closedLoopPredictedStates += [stateTraj[:, 1]]

        print("current state: ")
        print(stateVector)
        print("control input: ")
        print(controlTraj[:, 0])
        print("predicted state traj: ")
        print(stateTraj[:, 1])

        nextStateVector = car_sim_model.getSingleTrackStates()
        print("simulated state traj" )
        print(nextStateVector)

        closedLoopStates += [nextStateVector]
        stateVector = nextStateVector

    closedLoopStates = np.vstack(closedLoopStates)
    if plot == True:
        plotTraj(closedLoopStates[:, 0], closedLoopStates[:, 1], sx_openloops, sy_openloops, goal_A_polygon_x=goal_A_polygon_x, goal_A_polygon_y=goal_A_polygon_y, obstacle_polygon_x=obstacle_polygon_x, obstacle_polygon_y=obstacle_polygon_y)
    
    if len(closedLoopControlTraj) > 0:
        closedLoopControlTraj = np.vstack(closedLoopControlTraj)
    else:
        closedLoopControlTraj = np.array([])

    closedLoopPredictedStates = np.vstack(closedLoopPredictedStates)

    if numIters > 0:
        avgSolveTime = totalSolveTime/numIters
    else: 
        avgSolveTime = -1

    if writeTraces:
        for i in range(closedLoopStates.shape[0]):
            stateTrajFile.write(str(closedLoopStates[i, 0]) + "," + str(closedLoopStates[i, 1]) + "," + str(closedLoopStates[i, 2]) + "," + str(closedLoopStates[i, 3]) + "," + str(closedLoopStates[i, 4]) + "\n")
        stateTrajFile.write("~\n")
        for i in range(closedLoopControlTraj.shape[0]):
            controlTrajFile.write(str(closedLoopControlTraj[i, 0]) + "," + str(closedLoopControlTraj[i, 1]) + "\n")
        controlTrajFile.write("~\n")

    return avgSolveTime, maxSolveTime, totalSolveTime, numIters, feasible, closedLoopControlTraj, closedLoopPredictedStates, closedLoopStates

def modelMismatch(car_control_model, DT, RK_steps, stateTraj, controlTraj):
    assert(stateTraj.shape[0] == controlTraj.shape[0] + 1)

    mismatch = []
    car_nominal_dynamics = car_control_model.getDiscreteDynamics(DT, RK_steps, False)
    for i in range(controlTraj.shape[0]):
        initial_state = stateTraj[i, :]
        xk = DM(initial_state)
        sysk = car_nominal_dynamics(x0=xk, u=DM(controlTraj[i, :]))
        xk = sysk['xf']
        predicted_state = xk.full().transpose()
        mismatch += [stateTraj[i+1, :] - predicted_state]
    
    if len(mismatch) > 0:
        mismatch = np.vstack(mismatch)

    return mismatch

useGP = False

sx0_list = [sx0_2, sx0_3, sx0_4, sx0_5, sx0_6]
sy0_list = [sy0_2, sy0_3, sy0_4, sy0_5, sy0_6]
steerAng0_list = [steerAng0_2, steerAng0_3, steerAng0_4, steerAng0_5, steerAng0_6]
vel0_list = [vel0_2, vel0_3, vel0_4, vel0_5, vel0_6]
Psi0_list = [Psi0_2, Psi0_3, Psi0_4, Psi0_5, Psi0_6]

g_sx_list = []
g_sy_list = []
g_delta_list = []
g_v_list = []
g_carAng_list = []
controlList = []
stateInputDataList = []

#-----begin GP training from data----#
# for i in range(1, 4): 
#     _, _, _, _, _, controlData, predictedStateData, actualStateData = controlLoop(True, True, False, T, N, simulation_resolution, car_sys, car_sim, sx0_list[i], sy0_list[i], steerAng0_list[i], vel0_list[i], Psi0_list[i], goal_x, goal_y, obstacle_x, obstacle_y)
    
#     controlList += [controlData]
#     stateInputDataList += [actualStateData[:-1, 1:]]
#     g_sx_list += [actualStateData[1:, 0] - predictedStateData[1:, 0]]
#     g_sy_list += [actualStateData[1:, 1] - predictedStateData[1:, 1]]
#     g_delta_list += [actualStateData[1:, 2] - predictedStateData[1:, 2]]
#     g_v_list += [actualStateData[1:, 3] - predictedStateData[1:, 3]]
#     g_carAng_list += [actualStateData[1:, 4] - predictedStateData[1:, 4]]

# for i in range(0, 1): 
#     _, _, stateTrajRef, controlTrajRef = solveSetpointControl(False, T, N, car_sys, sx0_list[i], sy0_list[i], steerAng0_list[i], vel0_list[i], Psi0_list[i], 60, sy0_list[i], steerAng0_list[i], vel0_list[i], Psi0_list[i])

#     _, _, _, _, _, controlData, predictedStateData, actualStateData = controlLoopNomLTV(True, False, T, N, simulation_resolution, car_sys, car_sim, stateTrajRef, controlTrajRef, sx0_list[i], sy0_list[i], steerAng0_list[i], vel0_list[i], Psi0_list[i], goal_x, goal_y, obstacle_x, obstacle_y)
        
#     mismatch = modelMismatch(car_sys, T/N, 4, actualStateData, controlData)
#     controlList += [controlData]
#     stateInputDataList += [actualStateData[:-1, 1:]]
#     test = actualStateData[1:, :] - predictedStateData[1:, :]
#     test1 = np.abs(test - mismatch)
#     g_sx_list += [mismatch[:, 0]]
#     g_sy_list += [mismatch[:, 1]]
#     g_delta_list += [mismatch[:, 2]]
#     g_v_list += [mismatch[:, 3]]
#     g_carAng_list += [mismatch[:, 4]]

# g_sx = np.hstack(g_sx_list)
# g_sy = np.hstack(g_sy_list)
# g_delta = np.hstack(g_delta_list)
# g_v = np.hstack(g_v_list)
# g_carAng = np.hstack(g_carAng_list)

# controlData = np.vstack(controlList)
# stateInputData = np.vstack(stateInputDataList)

# trainInputs = torch.tensor(np.hstack([stateInputData, controlData]))
# train_sx = torch.tensor(g_sx)
# train_sy = torch.tensor(g_sy)
# train_delta = torch.tensor(g_delta)
# train_v = torch.tensor(g_v)
# train_carAng = torch.tensor(g_carAng)
# trainOutputs = [train_sx, train_sy, train_delta, train_v, train_carAng]

# car_sys.setGPResiduals(car_residualStateDims, car_residualInputDims, numTrainingIts, [trainInputs]*5, trainOutputs)

# car_GP_x_file = open("./gp_model_data/GP_x.txt", "w")
# car_GP_y_file = open("./gp_model_data/GP_y.txt", "w")
# car_GP_v_file = open("./gp_model_data/GP_v.txt", "w")
# car_GP_carAng_file = open("./gp_model_data/GP_carAng.txt", "w")

# car_sys.writeGPResidualsToFiles(car_GP_x_file, car_GP_y_file, None, car_GP_v_file, car_GP_carAng_file)

# car_GP_x_file.close()
# car_GP_y_file.close()
# car_GP_v_file.close()
# car_GP_carAng_file.close()
#-----end GP training from data----#

#-----begin GP load from file----#
car_GP_x_file = open("./gp_model_data/GP_x.txt", "r")
car_GP_y_file = open("./gp_model_data/GP_y.txt", "r")
car_GP_v_file = open("./gp_model_data/GP_v.txt", "r")
car_GP_carAng_file = open("./gp_model_data/GP_carAng.txt", "r")

car_sys.setGPResidualsFromFile(car_residualStateDims, car_residualInputDims, car_GP_x_file, car_GP_y_file, None, car_GP_v_file, car_GP_carAng_file)

car_GP_x_file.close()
car_GP_y_file.close()
car_GP_v_file.close()
car_GP_carAng_file.close()
#-----end GP load from file----#

def testLoop(numRuns, start_x, start_y, goal_x, goal_y, obstacle_x, obstacle_y):    
    totalSolveTimeLoopSmooth = 0
    numItersLoopSmooth = 0
    for i in range(numRuns):
        avgSolveTime, maxSolveTime, totalSolvetime, numIters, feasible, _, _, stateTraj = controlLoop(False, False, True, T, N, simulation_resolution, car_sys, car_sim, start_x, start_y, steerAng0, vel0, Psi0, goal_x, goal_y, obstacle_x, obstacle_y, smoothOp_state_trace, smoothOp_control_trace)
        totalSolveTimeLoopSmooth += totalSolvetime
        numItersLoopSmooth += numIters

        print("-----Smooth stats-----")
        print("Avg time: ", str(avgSolveTime))
        print("Max time: ", str(maxSolveTime))

    return avgSolveTime, maxSolveTime, totalSolveTimeLoopSmooth, numItersLoopSmooth

def testLoopLTV(numRuns, onlineCovar, stateTrace, controlTrace, start_x, start_y, goal_x, goal_y, obstacle_x, obstacle_y):
    car_sys.resetLTVstatus()
    totalSolveTimeLoopLTV = 0
    num_itersLoopLTV = 0
    for i in range(numRuns):
        _, _, stateTrajRef, controlTrajRef = solveSetpointControl(False, T, N, car_sys, start_x, start_y, steerAng0, vel0, Psi0, 60, sy0, steerAng0, vel0, Psi0)
        
        stateCovarTrajRefSim = generateCovariancePredictions(T/N, 4, car_sys, stateTrajRef, controlTrajRef)
        avgSolveTimeLTV, maxSolveTimeLTV, totalSolveTimeLTV, num_itersLTV, feasibleLTV, _, _, stateTraj = controlLoopLTV(False, True, onlineCovar, True, T, N, simulation_resolution, car_sys, car_sim, stateTrajRef, controlTrajRef, stateCovarTrajRefSim, start_x, start_y, steerAng0, vel0, Psi0, goal_x, goal_y, obstacle_x, obstacle_y, stateTrace, controlTrace)
        
        totalSolveTimeLoopLTV += totalSolveTimeLTV
        num_itersLoopLTV += num_itersLTV
        
        print("-----LTV stats-----")
        print("x_init, y_init: ", str(start_x), ", ", str(start_y))
        print("i: ", str(i))
        print("Avg time: ", str(avgSolveTimeLTV))
        print("Max time: ", str(maxSolveTimeLTV))

    return avgSolveTimeLTV, maxSolveTimeLTV, totalSolveTimeLoopLTV, num_itersLoopLTV

def testLoopNomLTV(numRuns, start_x, start_y, goal_x, goal_y, obstacle_x, obstacle_y):
    car_sys.resetLTVstatus()
    totalSolveTimeLoopNomLTV = 0
    num_itersLoopNomLTV = 0
    for i in range(numRuns):
        _, _, stateTrajRef, controlTrajRef = solveSetpointControl(False, T, N, car_sys, start_x, start_y, steerAng0, vel0, Psi0, 60, sy0, steerAng0, vel0, Psi0)
       
        avgSolveTimeNomLTV, maxSolveTimeNomLTV, totalSolveTimeNomLTV, num_itersNomLTV, feasibleNomLTV, _, _, stateTraj = controlLoopNomLTV(False, True, T, N, simulation_resolution, car_sys, car_sim, stateTrajRef, controlTrajRef, start_x, start_y, steerAng0, vel0, Psi0, goal_x, goal_y, obstacle_x, obstacle_y, nom_state_trace, nom_control_trace)
        
        totalSolveTimeLoopNomLTV += totalSolveTimeNomLTV
        num_itersLoopNomLTV += num_itersNomLTV

        print("-----LTV stats-----")
        print("x_init, y_init: ", str(start_x), ", ", str(start_y))
        print("i: ", str(i))
        print("Avg time: ", str(avgSolveTimeNomLTV))
        print("Max time: ", str(maxSolveTimeNomLTV))

    return avgSolveTimeNomLTV, maxSolveTimeNomLTV, totalSolveTimeLoopNomLTV, num_itersLoopNomLTV 

numStartPositions = 30
numberTrials = 30


avgTimeNomLTVList = []
maxTimeNomLTVList = []

avgTimeSmoothList = []
maxTimeSmoothList = []

avgTimeLTVList = []
maxTimeLTVList = []

avgTimeLTVList_offlineCovar = []
maxTimeLTVList_offlineCovar = []

configs = [35]

start_xy_rng = np.random.default_rng(seed=0)
start_x = start_xy_rng.random(30) * 3 + 1
start_y = start_xy_rng.random(30) * 2 + 0.5
print(start_x)
print(start_y)

totalTimeNomLTV = 0
totalNumItersNomLTV = 0
for config in configs:
    goal_x = [config, 100]
    goal_y = [0, 3] 
    obstacle_x = [20, config]
    obstacle_y = [0, 3]
    for i in range(numStartPositions):
        x_init = start_x[i]
        y_init = start_y[i]

        avgTimeNomLTV, maxTimeNomLTV, oneLoopTotalTimeNomLTV, oneLoopNumItersNomLTV = testLoopNomLTV(numberTrials, x_init, y_init, goal_x, goal_y, obstacle_x, obstacle_y)

        totalTimeNomLTV += oneLoopTotalTimeNomLTV 
        totalNumItersNomLTV += oneLoopNumItersNomLTV

        avgTimeNomLTVList += [avgTimeNomLTV]
        maxTimeNomLTVList += [maxTimeNomLTV]


totalTimeSmooth = 0
totalNumItersSmooth = 0
for config in configs:
    goal_x = [config, 100]
    goal_y = [0, 3] 
    obstacle_x = [20, config]
    obstacle_y = [0, 3]

    for i in range(numStartPositions):
        x_init = start_x[i]
        y_init = start_y[i]

        avgTimeSmooth, maxTimeSmooth, oneLoopSolveTime, oneLoopNumIters = testLoop(numberTrials, x_init, y_init, goal_x, goal_y, obstacle_x, obstacle_y)

        totalTimeSmooth += oneLoopSolveTime
        totalNumItersSmooth += oneLoopNumIters

        avgTimeSmoothList += [avgTimeSmooth]
        maxTimeSmoothList += [maxTimeSmooth]

totalTimeLTV = 0
totalNumItersLTV = 0
for config in configs:
    goal_x = [config, 100]
    goal_y = [0, 3] 
    obstacle_x = [20, config]
    obstacle_y = [0, 3]
    for i in range(numStartPositions):
        x_init = start_x[i]
        y_init = start_y[i]

        avgTimeLTV, maxTimeLTV, oneLoopTotalTimeLTV, oneLoopNumItersLTV = testLoopLTV(numberTrials, True, LTVGP_state_trace, LTVGP_control_trace, x_init, y_init, goal_x, goal_y, obstacle_x, obstacle_y)

        totalTimeLTV += oneLoopTotalTimeLTV 
        totalNumItersLTV += oneLoopNumItersLTV

        avgTimeLTVList += [avgTimeLTV]
        maxTimeLTVList += [maxTimeLTV]
    

totalTimeLTV_offlineCovar = 0
totalNumItersLTV_offlineCovar = 0
for config in configs:
    goal_x = [config, 100]
    goal_y = [0, 3] 
    obstacle_x = [20, config]
    obstacle_y = [0, 3]
    for i in range(numStartPositions):
        x_init = start_x[i]
        y_init = start_y[i]

        avgTimeLTV_offlineCovar, maxTimeLTV_offlineCovar, oneLoopTotalTimeLTV_offlineCovar, oneLoopNumItersLTV_offlineCovar = testLoopLTV(numberTrials, False, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, x_init, y_init, goal_x, goal_y, obstacle_x, obstacle_y)

        totalTimeLTV_offlineCovar += oneLoopTotalTimeLTV_offlineCovar
        totalNumItersLTV_offlineCovar += oneLoopNumItersLTV_offlineCovar
        
        avgTimeLTVList_offlineCovar += [avgTimeLTV_offlineCovar]
        maxTimeLTVList_offlineCovar += [maxTimeLTV_offlineCovar]


totalAvgTimeNomLTV = -1
if totalNumItersNomLTV > 0:
    totalAvgTimeNomLTV = totalTimeNomLTV/totalNumItersNomLTV

totalAvgTimeSmooth = -1 
if totalNumItersSmooth > 0:
    totalAvgTimeSmooth = totalTimeSmooth/totalNumItersSmooth

totalAvgTimeLTV = -1
if totalNumItersLTV > 0: 
    totalAvgTimeLTV = totalTimeLTV/totalNumItersLTV

totalAvgTimeLTVofflineCovar = -1
if totalNumItersLTV_offlineCovar > 0:
    totalAvgTimeLTVofflineCovar = totalTimeLTV_offlineCovar/totalNumItersLTV_offlineCovar

print("-----Avg time stat-----")
print("Avg time Nom: ", totalAvgTimeNomLTV)
print("Avg time smooth: ", totalAvgTimeSmooth)
print("Avg time LTV: ", totalAvgTimeLTV)
print("Avg time LTV offline covar: ", totalAvgTimeLTVofflineCovar)

print("-----num Iters debug-----")
print("num iters Nom: ", totalNumItersNomLTV)
print("num iters smooth: ", totalNumItersSmooth)
print("num iters LTV: ", totalNumItersLTV)
print("num iters LTV offline covar: ", totalNumItersLTV_offlineCovar)


print("-----Max time stat-----")
print("Max time Nom: ", maxTimeNomLTVList)
print("Max time smooth: ", maxTimeSmoothList)
print("Max time LTV: ", maxTimeLTVList)
print("Max time LTV offline covar: ", maxTimeLTVList_offlineCovar)

smoothOp_state_trace.close()
smoothOp_control_trace.close()
LTVGP_state_trace.close()
LTVGP_control_trace.close()
LTVGP_state_trace_offlineCovar.close()
LTVGP_control_trace_offlineCovar.close()
nom_state_trace.close()
nom_control_trace.close()

plt.show()
