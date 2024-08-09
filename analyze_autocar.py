import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import math

from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

import single_track_model_car as stcar

# load parameters
p = parameters_vehicle2()
g = 9.81  # [m/s^2]

# control model -----------------------------------------------------------------
carlf = 1.156 # this doesn't seem to be in the framework files, but it is in the documentation
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

car_residualStateDims = [4, 4, 4, 4, 4]
car_residualInputDims = [2, 2, 2, 2, 2]

car_GP_x_file = open("./gp_model_data/GP_x.txt", "r")
car_GP_y_file = open("./gp_model_data/GP_y.txt", "r")
car_GP_v_file = open("./gp_model_data/GP_v.txt", "r")
car_GP_carAng_file = open("./gp_model_data/GP_carAng.txt", "r")

car_sys.setGPResidualsFromFile(car_residualStateDims, car_residualInputDims, car_GP_x_file, car_GP_y_file, None, car_GP_v_file, car_GP_carAng_file)

car_GP_x_file.close()
car_GP_y_file.close()
car_GP_v_file.close()
car_GP_carAng_file.close()

# Horizon 
T = 4
N = 32

goal_x = [35, 100]
goal_y = [0, 3] 
obstacle_x = [20, 35]
obstacle_y = [0, 3]
goal_min_speed = 15
goal_carAng = [-1/20*math.pi, 1/20*math.pi]

bigMx = xlim[1] - xlim[0] + 1
bigMy = ylim[1] - ylim[0] + 1
bigMv = vlim[1] - vlim[0] + 1
bigMcarAng = carAngLim[1] - carAngLim[0] + 1

def getInitialStates(stateTraceFile):
    stateTraceFile.seek(0)
    initialStates = []
    initialState = np.array([])
    prev_initialState = np.array([-100, -100, -100, -100, -100])
    for line in stateTraceFile:
        if line == "~\n":
            if initialState.size > 0 and not(np.array_equal(initialState, prev_initialState)):
                initialStates += [initialState]
                prev_initialState = initialState
            initialState = np.array([])
        else:
            if initialState.size == 0:
                lineElements = line.split(",")
                x1 = float(lineElements[0])
                x2 = float(lineElements[1])
                x3 = float(lineElements[2])
                x4 = float(lineElements[3])
                x5_processed = lineElements[4].split("\n")
                x5 = float(x5_processed[0])
                initialState = np.array([x1, x2, x3, x4, x5])

    return initialStates

def getTracesFor(init_x, init_y, stateTraceFile, controlTraceFile):
    stateTraceFile.seek(0)
    controlTraceFile.seek(0)

    stateTraces = []
    stateTrace = []
    for line in stateTraceFile:
        if line == "~\n":
            # stateTrajectory = np.array([])
            if len(stateTrace) > 0:
                stateTrajectory = np.vstack(stateTrace)
            stateTraces += [stateTrajectory]
            stateTrace = []
        else: 
            lineElements = line.split(",")
            x1 = float(lineElements[0])
            x2 = float(lineElements[1])
            x3 = float(lineElements[2])
            x4 = float(lineElements[3])
            x5_processed = lineElements[4].split("\n")
            x5 = float(x5_processed[0])
            stateTrace += [np.array([x1, x2, x3, x4, x5])] 

    controlTrace = []
    controlTraces = []
    for line in controlTraceFile:
        if line == "~\n":
            controlTrajectory = np.array([])
            if len(controlTrace) > 0:
                controlTrajectory = np.vstack(controlTrace)
            controlTraces += [controlTrajectory]
            controlTrace = []
        else: 
            lineElements = line.split(",")
            u1 = float(lineElements[0])
            u2 = float(lineElements[1].split("\n")[0])
            controlTrace += [np.array([u1, u2])] 

    targetStateTraces = []
    targetControlTraces = []
    for i in range(len(stateTraces)):
        stateTrace = stateTraces[i]
        if stateTrace.size > 0 and stateTrace[0, 0] == init_x and stateTrace[0, 1] == init_y:
            controlTrace = controlTraces[i]
            targetStateTraces += [stateTrace]
            targetControlTraces += [controlTrace]
    
    return targetStateTraces, targetControlTraces

def plotSol(N, plotControl, x_smoothOp = [], u_smoothOp = [], x_GP = [], u_GP = [], x_GP_offlineCovar = [], u_GP_offlineCovar = [], x_Nom = [], u_Nom = [], goal_A_polygon_x = [], goal_A_polygon_y = [], obstacle_polygon_x = [], obstacle_polygon_y = []):
    nom_plotIdx = 0
    smoothOp_plotIdx = 1
    GP_plotIdx = 2
    GP_offlineCovar_plotIdx = 3
    
    fig, ax = plt.subplots(4, 1, constrained_layout=True)

    if len(goal_A_polygon_x) > 0:
        goal_A_polygon_x_plot = [goal_A_polygon_x[0], goal_A_polygon_x[0], goal_A_polygon_x[1], goal_A_polygon_x[1]]
        goal_A_polygon_y_plot = [goal_A_polygon_y[1], goal_A_polygon_y[0], goal_A_polygon_y[0], goal_A_polygon_y[1]] 
        obstacle_polygon_x_plot = [obstacle_polygon_x[0], obstacle_polygon_x[0], obstacle_polygon_x[1], obstacle_polygon_x[1]]
        obstacle_polygon_y_plot = [obstacle_polygon_y[1], obstacle_polygon_y[0], obstacle_polygon_y[0], obstacle_polygon_y[1]]
        
        ax[smoothOp_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[smoothOp_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[smoothOp_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[smoothOp_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

        ax[GP_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[GP_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[GP_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[GP_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

        ax[GP_offlineCovar_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[GP_offlineCovar_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[GP_offlineCovar_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[GP_offlineCovar_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

        ax[nom_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[nom_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[nom_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[nom_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

    ax[smoothOp_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[smoothOp_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[smoothOp_plotIdx].set_xlabel('Sx (m)')
    ax[smoothOp_plotIdx].set_ylabel('Sy (m)')

    for traj in x_smoothOp:
        ax[smoothOp_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color="xkcd:windows blue")
    ax[smoothOp_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[GP_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[GP_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[GP_plotIdx].set_xlabel('Sx (m)')
    ax[GP_plotIdx].set_ylabel('Sy (m)')

    for traj in x_GP:
        ax[GP_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color="xkcd:orangish")
    ax[GP_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[GP_offlineCovar_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[GP_offlineCovar_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[GP_offlineCovar_plotIdx].set_xlabel('Sx (m)')
    ax[GP_offlineCovar_plotIdx].set_ylabel('Sy (m)')

    for traj in x_GP_offlineCovar:
        ax[GP_offlineCovar_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color="xkcd:orangish")
    ax[GP_offlineCovar_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[nom_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[nom_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[nom_plotIdx].set_xlabel('Sx (m)')
    ax[nom_plotIdx].set_ylabel('Sy (m)')

    for traj in x_Nom:
        ax[nom_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color="xkcd:windows blue")
    ax[nom_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    if plotControl:
        fig_u, ax_u = plt.subplots(2, 1, constrained_layout=True)
        vSteerAngIdx = 0
        accelIdx = 1
        ax_u[vSteerAngIdx].grid(True)
        ax_u[accelIdx].grid(True)
        ax_u[vSteerAngIdx].set_ylabel("vSteerAng (rad/s)")
        ax_u[accelIdx].set_ylabel("Acceleration (m/s^2)")
        ax_u[vSteerAngIdx].set_xlabel("k")
        ax_u[accelIdx].set_xlabel("k")
        ax_u[vSteerAngIdx].set_ylim([-0.4, 0.4])
        ax_u[accelIdx].set_ylim([-10, 10])
        
        for controlTraj in u_Nom:
            tgrid = [k for k in range(len(controlTraj))]
            nom_vSteerAng, = ax_u[vSteerAngIdx].plot(tgrid, controlTraj[:,0], '-o', alpha=0.5, linewidth=1, label="Nominal", color="steelblue")
            nom_accel, = ax_u[accelIdx].plot(tgrid, controlTraj[:, 1], '-o', alpha=0.5, linewidth=1, label="Nominal", color="steelblue")
            
        for controlTraj in u_GP:
            tgrid = [k for k in range(len(controlTraj))]
            gp_vSteerAng, = ax_u[vSteerAngIdx].plot(tgrid, controlTraj[:,0], '-o', alpha=0.5, linewidth=1, label="GP", color="purple")
            gp_accel, = ax_u[accelIdx].plot(tgrid, controlTraj[:, 1], '-o', alpha=0.5, linewidth=1, label="GP", color="purple")

        for controlTraj in u_GP_offlineCovar:
            tgrid = [k for k in range(len(controlTraj))]
            gpOfflineCovar_vSteerAng, = ax_u[vSteerAngIdx].plot(tgrid, controlTraj[:,0], '-o', alpha=0.5, linewidth=1, label="GP (offlineCovar)", color="green")
            gpOfflineCovar_accel, = ax_u[accelIdx].plot(tgrid, controlTraj[:, 1], '-o', alpha=0.5, linewidth=1, label="GP (offlineCovar)", color="green")

        ax_u[vSteerAngIdx].legend(handles=[nom_vSteerAng, gp_vSteerAng, gpOfflineCovar_vSteerAng])
        ax_u[accelIdx].legend(handles=[nom_accel, gp_accel, gpOfflineCovar_accel])

def plotAllSol(N, plotControlTraj, stateTraceFileSmoothOp, controlTraceFileSmoothOp, stateTraceFileGP, controlTraceFileGP, stateTraceFileGPofflineCovar, controlTraceFileGPofflineCovar, stateTraceFileNom, controlTraceFileNom):
    initialStatesSmoothOp = getInitialStates(stateTraceFileSmoothOp)
    initialStatesGP = getInitialStates(stateTraceFileGP)
    initialStatesNom = getInitialStates(stateTraceFileNom)
    for initialState in initialStatesSmoothOp:
        stateTracesSmoothOp, controlTracesSmoothOp = getTracesFor(initialState[0], initialState[1], stateTraceFileSmoothOp, controlTraceFileSmoothOp)
        stateTracesGP, controlTracesGP = getTracesFor(initialState[0], initialState[1], stateTraceFileGP, controlTraceFileGP)
        stateTraceGPofflineCovar, controlTraceGPofflineCovar = getTracesFor(initialState[0], initialState[1], stateTraceFileGPofflineCovar, controlTraceFileGPofflineCovar)
        stateTracesNom, controlTracesNom = getTracesFor(initialState[0], initialState[1], stateTraceFileNom, controlTraceFileNom)
        plotSol(N, plotControlTraj, stateTracesSmoothOp, controlTracesSmoothOp, stateTracesGP, controlTracesGP, stateTraceGPofflineCovar, controlTraceGPofflineCovar, stateTracesNom, controlTracesNom, goal_x, goal_y, obstacle_x, obstacle_y)

def checkSAT(stateTraj, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y):
    #This routine checks whether the trajectory satisfies the STL specification by using gurobi to try and assign integer variables, in hindsight not the most efficient way to check but it's been kept from the development process
    SAT = False

    if stateTraj[-1, -1] > goal_carAng[1] or stateTraj[-1, -1] < goal_carAng[0] or stateTraj[-1, 1] > goal_A_polygon_y[1] or stateTraj[-1, 3] < goal_min_speed:
        return SAT

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

    eventuallyAlwaysList = []
    for k in range(N+1):
        #Integer variables for STL 
        Zavoid_k = MX.sym('Zavoid_' + str(k), 3)
        w += [Zavoid_k]
        lbw += [0]*3
        ubw += [1]*3
        w0 += [0]*3
        discrete += [True]*3

        Zreach_k = MX.sym('Zreach_' + str(k), 1)
        w += [Zreach_k]
        lbw += [0]*1
        ubw += [1]*1
        w0 += [0]*1
        discrete += [True]*1

        eventuallyAlwaysList += [Zreach_k]

        # STL constraints
        # avoid
        g += [obstacle_polygon_x[0] - stateTraj[k, 0] + bigMx*(1 - Zavoid_k[0])] 
        g += [stateTraj[k, 0] - obstacle_polygon_x[1] + bigMx*(1 - Zavoid_k[1])] 
        g += [stateTraj[k, 1] - obstacle_polygon_y[1] + bigMy*(1 - Zavoid_k[2])] 
        
        g += [Zavoid_k[0] + Zavoid_k[1] + Zavoid_k[2]]
    
        lbg += [0, 0, 0, 1]
        ubg += [2*bigMx, 2*bigMx, 2*bigMy, 3]

        # reach
        for j in range(k+1):
            g += [stateTraj[k, 0] - goal_A_polygon_x[0] + bigMx*(1 - eventuallyAlwaysList[j])] 
            g += [goal_A_polygon_x[1] - stateTraj[k, 0] + bigMx*(1 - eventuallyAlwaysList[j])] 
            g += [goal_A_polygon_y[1] - stateTraj[k, 1] + bigMy*(1 - eventuallyAlwaysList[j])] 
            g += [stateTraj[k, 1] - goal_A_polygon_y[0] + bigMy*(1 - eventuallyAlwaysList[j])]
            g += [stateTraj[k, 3] - goal_min_speed + bigMv*(1 - eventuallyAlwaysList[j])]
            g += [stateTraj[k, 4] - goal_carAng[0] + bigMcarAng*(1 - eventuallyAlwaysList[j])]
            g += [goal_carAng[1] - stateTraj[k, 4] + bigMcarAng*(1 - eventuallyAlwaysList[j])]

            lbg += [0, 0, 0, 0, 0, 0, 0]
            ubg += [2*bigMx, 2*bigMx, 2*bigMy, 2*bigMy, 2*bigMv, 2*bigMcarAng, 2*bigMcarAng]

        # envelope
        g += [stateTraj[k, 0] - xlim[0]]
        g += [xlim[1] - stateTraj[k, 0]]
        g += [stateTraj[k, 1] - ylim[0]]
        g += [ylim[1] - stateTraj[k, 1]]

        lbg += [0, 0, 0, 0]
        ubg += [2*bigMx, 2*bigMx, 2*bigMy, 2*bigMy]
    
            
    eventuallyAlways = 0
    for k in range(N+1):
        eventuallyAlways += eventuallyAlwaysList[k]

    g += [eventuallyAlways]
    
    lbg += [1]
    ubg += [N+1]

    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # J = 1

    # Create an NLP solver
    qp_prob = {'f': J, 'x': w, 'g': g}
    qp_solver = qpsol('qp_solver', 'gurobi', qp_prob, {"discrete": discrete, "error_on_fail": False})

    # Solve the NLP
    sol = qp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(qp_solver.stats())

    solver_stats = qp_solver.stats()
    SAT = solver_stats['success']

    return SAT

def satLoop(stateTraceFile, goal_x, goal_y, obstacle_x, obstacle_y):
    stateTraceFile.seek(0)
    stateTrace = []
    satCount = 0
    recursiveFeasibilityCount = 0
    for line in stateTraceFile:
        if line == "~\n":
            if len(stateTrace) > N:
                recursiveFeasibilityCount += 1
                stateTrajectory = np.vstack(stateTrace)
                trajSat = checkSAT(stateTrajectory, goal_x, goal_y, obstacle_x, obstacle_y)
                if trajSat:
                    satCount += 1
            stateTrace = []
        else: 
            lineElements = line.split(",")
            x1 = float(lineElements[0])
            x2 = float(lineElements[1])
            x3 = float(lineElements[2])
            x4 = float(lineElements[3])
            x5_processed = lineElements[4].split("\n")
            x5 = float(x5_processed[0])
            stateTrace += [np.array([x1, x2, x3, x4, x5])] 

    return satCount, recursiveFeasibilityCount

def satList(N, stateTraceFile, controlTraceFile, goal_x, goal_y, obstacle_x, obstacle_y):
    stateTraceFile.seek(0)

    satList = []
    initialStates = getInitialStates(stateTraceFile)
    for initialState in initialStates:
        stateTraces, _ = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
        satCount = 0
        for stateTrace in stateTraces:
            if stateTrace.shape[0] >= N:
                traceSat = checkSAT(stateTrace, goal_x, goal_y, obstacle_x, obstacle_y)
                if traceSat:
                    satCount += 1
        
        satList += [satCount]

    return satList

def avgControlEffort(N, stateTraceFile, controlTraceFile):
    avgControlEffort = -1
    initialStates = getInitialStates(stateTraceFile)

    totalControlEffort = 0
    recursivelyFeasibleCount = 0
    bigEffortCount = 0
    bigEffortSum = 0
    for initialState in initialStates:
        stateTraces, controlTraces = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
        assert(len(stateTraces) == len(controlTraces))
        for i in range(len(stateTraces)):
            controlTrace = controlTraces[i]
            if controlTrace.shape[0] == N:
                recursivelyFeasibleCount += 1
                controlEffort = np.sum(np.square(controlTrace), axis=None)
                if controlEffort > 1:
                    bigEffortCount += 1
                    bigEffortSum += controlEffort
                totalControlEffort += controlEffort
                
    if recursivelyFeasibleCount > 0:
        avgControlEffort = totalControlEffort/recursivelyFeasibleCount

    return avgControlEffort

def avgRobustness(stateTraceFile, controlTraceFile, goal_x, goal_y, obstacle_x, obstacle_y):
    initialStates = getInitialStates(stateTraceFile)
    total_trajectory_rho = 0
    recursively_feasible_count = 0

    for initialState in initialStates:
        stateTraces, controlTraces = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
        # STL_sat_count = 0
        obstacle_x_0_rho = [0]*(N+1)
        obstacle_x_1_rho = [0]*(N+1)
        obstacle_y_1_rho = [0]*(N+1)
        obstacle_rho = [0]*(N+1)
        goal_x_0_rho = [0]*(N+1)
        goal_x_1_rho = [0]*(N+1)
        goal_y_0_rho = [0]*(N+1)
        goal_y_1_rho = [0]*(N+1)
        goal_speed_rho = [0]*(N+1)
        goal_carAng_0_rho = [0]*(N+1)
        goal_carAng_1_rho = [0]*(N+1)
        goal_rho = [0]*(N+1)
        for stateTrace in stateTraces:
            if stateTrace.shape[0] > N:
                recursively_feasible_count += 1

                for i in range(N+1):
                    sx = stateTrace[i, 0]
                    sy = stateTrace[i, 1]
                    vel = stateTrace[i, 3]
                    carAng = stateTrace[i, 4]

                    obstacle_x_0_rho[i] = obstacle_x[0] - sx
                    obstacle_x_1_rho[i] = sx - obstacle_x[1]
                    obstacle_y_1_rho[i] = sy - obstacle_y[1]
                    obstacle_rho[i] = max([obstacle_x_0_rho[i], obstacle_x_1_rho[i], obstacle_y_1_rho[i]])
                    
                    goal_x_0_rho[i] = sx - goal_x[0]
                    goal_x_1_rho[i] = goal_x[1] - sx
                    goal_y_0_rho[i] = sy - goal_y[0]
                    goal_y_1_rho[i] = goal_y[1] - sy
                    
                    goal_speed_rho[i] = vel - goal_min_speed

                    goal_carAng_0_rho[i] = carAng - goal_carAng[0]
                    goal_carAng_1_rho[i] = goal_carAng[1] - carAng
                    
                    goal_rho[i] = min([goal_x_0_rho[i], goal_x_1_rho[i], goal_y_0_rho[i], goal_y_1_rho[i], goal_speed_rho[i], goal_carAng_0_rho[i], goal_carAng_1_rho[i]])

                always_goal_rho = [0]*(N+1)
                for i in range(N+1):
                    always_goal_rho[i] = min(goal_rho[i:])
                
                always_obstacle_rho = min(obstacle_rho)
                eventually_always_goal_rho = max(always_goal_rho)

                trajectory_rho = min([always_obstacle_rho, eventually_always_goal_rho])
                print(trajectory_rho)

                total_trajectory_rho += trajectory_rho

    avg_rho = -10000
    if recursively_feasible_count > 0:
        avg_rho = total_trajectory_rho/recursively_feasible_count

    return avg_rho

smoothOp_state_trace = open("./trace_data/autocar_state_traces_smoothOp.txt", "r")
smoothOp_control_trace = open("./trace_data/autocar_control_traces_smoothOp.txt", "r")
LTVGP_state_trace = open("./trace_data/autocar_state_trace_LTVGP.txt", "r")
LTVGP_control_trace = open("./trace_data/autocar_control_trace_LTVGP.txt", "r")
LTVGP_state_trace_offlineCovar = open("./trace_data/autocar_state_trace_LTVGP_offlineCovar.txt", "r")
LTVGP_control_trace_offlineCovar = open("./trace_data/autocar_control_trace_LTVGP_offlineCovar.txt", "r")
nom_state_trace = open("./trace_data/autocar_state_trace_nom.txt", "r")
nom_control_trace = open("./trace_data/autocar_control_trace_nom.txt", "r")

plotAllSol(N, False, smoothOp_state_trace, smoothOp_control_trace, LTVGP_state_trace, LTVGP_control_trace, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, nom_state_trace, nom_control_trace)

avgRhoSmooth = avgRobustness(smoothOp_state_trace, smoothOp_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
avgRhoLTV = avgRobustness(LTVGP_state_trace, LTVGP_control_trace_offlineCovar, goal_x, goal_y, obstacle_x, obstacle_y)
avgRhoLTV_offlinecovar = avgRobustness(LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, goal_x, goal_y, obstacle_x, obstacle_y)
avgRhoNom = avgRobustness(nom_state_trace, nom_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)

avgControlEffortSmoothOp = avgControlEffort(N, smoothOp_state_trace, smoothOp_control_trace)
avgControlEffortLTV = avgControlEffort(N, LTVGP_state_trace, LTVGP_control_trace)
avgControlEffortLTVofflineCovar = avgControlEffort(N, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar)
avgControlEffortNom = avgControlEffort(N, nom_state_trace, nom_control_trace)

satSmoothOp, recursiveFeasibilitySmoothOp = satLoop(smoothOp_state_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satGP, recursiveFeasibilityGP = satLoop(LTVGP_state_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satGPofflineCovar, recursiveFeasibilityGPofflineCovar = satLoop(LTVGP_state_trace_offlineCovar, goal_x, goal_y, obstacle_x, obstacle_y)
satNom, recursiveFeasibilityNom = satLoop(nom_state_trace, goal_x, goal_y, obstacle_x, obstacle_y)

satListSmoothOp = satList(N, smoothOp_state_trace, smoothOp_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satListNom = satList(N, nom_state_trace, nom_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satListGP = satList(N, LTVGP_state_trace, LTVGP_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satListGP_offlineCovar = satList(N, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, goal_x, goal_y, obstacle_x, obstacle_y)

smoothOp_state_trace.close()
smoothOp_control_trace.close()
LTVGP_state_trace.close()
LTVGP_control_trace.close()
LTVGP_state_trace_offlineCovar.close()
LTVGP_control_trace_offlineCovar.close()
nom_state_trace.close()
nom_control_trace.close()

print("---Sat stats---")
print("Smooth: ", str(satSmoothOp))
print("GP: ", str(satGP))
print("GP offline covar: ", str(satGPofflineCovar))
print("Nom: ", str(satNom))

print("---Sat List---")
print("Smooth:, ", satListSmoothOp)
print("GP: ", satListGP)
print("GP offline covar: ", satListGP_offlineCovar)
print("Nom: ", satListNom)

print("---Recursive feasibility---")
print("Smooth: ", str(recursiveFeasibilitySmoothOp))
print("GP: ", str(recursiveFeasibilityGP))
print("GP offline covar: ", str(recursiveFeasibilityGPofflineCovar))
print("Nom: ", str(recursiveFeasibilityNom))

print("---Control effort cost---")
print("Smooth: ", str(avgControlEffortSmoothOp))
print("GP: ", str(avgControlEffortLTV))
print("GP offline covar: ", str(avgControlEffortLTVofflineCovar))
print("Nom: ", str(avgControlEffortNom))

print("---Avg Robustness---")
print("Smooth: ", str(avgRhoSmooth))
print("GP: ", str(avgRhoLTV))
print("GP offline: ", str(avgRhoLTV_offlinecovar))
print("Nom: ", str(avgRhoNom))

plt.show()