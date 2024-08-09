import numpy as np
import matplotlib.pyplot as plt
from casadi import *

import cstr_model as cstr

smoothOp_state_trace_file = open("./trace_data/cstr_30x30_state_traces_smoothOp.txt", "r")
smoothOp_control_trace_file = open("./trace_data/cstr_30x30_control_traces_smoothOp.txt", "r")
GP_state_trace_file = open("./trace_data/cstr_30x30_state_traces_GP.txt", "r")
GP_control_trace_file = open("./trace_data/cstr_30x30_control_traces_GP.txt", "r")
Nom_state_trace_file = open("./trace_data/cstr_30x30_state_traces_nom.txt", "r")
Nom_control_trace_file = open("./trace_data/cstr_30x30_control_traces_nom.txt", "r")

cstr_paramD = 0.078
cstr_paramB = 8
cstr_paramGamma = 20
cstr_paramBeta = 0.3

cstr_sys = cstr.cstrModel(cstr_paramD, cstr_paramB, cstr_paramGamma, cstr_paramBeta)
cstr_x1lim, cstr_x2lim = cstr_sys.getStateLimits()
cstr_ulim = cstr_sys.getInputLimits()

cstr_GP_x1_file = open("./gp_model_data/cstr_GP_x1.txt", "r")
cstr_GP_x2_file = open("./gp_model_data/cstr_GP_x2.txt", "r")

cstr_sys.setGPResidualsFromFile(100, cstr_GP_x1_file, cstr_GP_x2_file)

cstr_GP_x1_file.close()
cstr_GP_x2_file.close()

#Initial conditions and problem parameters
T = 6
N = 12 

bigMx1 = (cstr_x1lim[1] - cstr_x1lim[0]) + 1
bigMx2 = (cstr_x2lim[1] - cstr_x2lim[0]) + 1

C_th = 0.1
T_th = 0.5

def satLoop(stateTraceFile):
    stateTrace = []
    satCount = 0
    for line in stateTraceFile:
        if line == "~\n":
            if len(stateTrace) > N:
                stateTrajectory = np.vstack(stateTrace)
                trajSat = checkSAT(N, stateTrajectory)
                if trajSat:
                    satCount += 1
            stateTrace = []
        else: 
            lineElements = line.split(",")
            x1 = float(lineElements[0])
            x2_processed = lineElements[1].split("\n")
            x2 = float(x2_processed[0])
            stateTrace += [np.array([x1, x2])] 

    return satCount

def getInitialStates(stateTraceFile):
    stateTraceFile.seek(0)
    initialStates = []
    initialState = np.array([])
    prev_initialState = np.array([-100, -100])
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
                x2_processed = lineElements[1].split("\n")
                x2 = float(x2_processed[0])
                initialState = np.array([x1, x2])

    return initialStates

def getTracesFor(init_x, init_y, stateTraceFile, controlTraceFile):
    stateTraceFile.seek(0)
    controlTraceFile.seek(0)

    stateTraces = []
    stateTrace = []
    for line in stateTraceFile:
        if line == "~\n":
            if len(stateTrace) > 0:
                stateTrajectory = np.vstack(stateTrace)
            stateTraces += [stateTrajectory]
            stateTrace = []
        else: 
            lineElements = line.split(",")
            x1 = float(lineElements[0])
            x2_processed = lineElements[1].split("\n")
            x2 = float(x2_processed[0])
            stateTrace += [np.array([x1, x2])] 

    controlTrace = []
    controlTraces = []
    for line in controlTraceFile:
        if line == "~\n":
            controlTrajectory = np.array([])
            if len(controlTrace) > 0:
                controlTrajectory = np.hstack(controlTrace)
            controlTraces += [controlTrajectory]
            controlTrace = []
        else: 
            u = float(line.split("\n")[0])
            controlTrace += [np.array([u])] 

    targetStateTraces = []
    targetControlTraces = []
    targetPredictionTraces = []
    for i in range(len(stateTraces)):
        stateTrace = stateTraces[i]
        if stateTrace.size > 0 and stateTrace[0, 0] == init_x and stateTrace[0, 1] == init_y:
            controlTrace = controlTraces[i]
            targetStateTraces += [stateTrace]
            targetControlTraces += [controlTrace]
    
    return targetStateTraces, targetControlTraces

def plotAllSol(N, stateTraceFileSmoothOp, controlTraceFileSmoothOp, stateTraceFileGP, controlTraceFileGP, stateTraceFileNom, controlTraceFileNom):
    initialStatesSmoothOp = getInitialStates(stateTraceFileSmoothOp)
    initialStatesGP = getInitialStates(stateTraceFileGP)
    initialStatesNom = getInitialStates(stateTraceFileNom)
    for initialState in initialStatesSmoothOp:
        stateTracesSmoothOp, controlTracesSmoothOp = getTracesFor(initialState[0], initialState[1], stateTraceFileSmoothOp, controlTraceFileSmoothOp)
        stateTracesGP, controlTracesGP = getTracesFor(initialState[0], initialState[1], stateTraceFileGP, controlTraceFileGP)
        stateTracesNom, controlTracesNom = getTracesFor(initialState[0], initialState[1], stateTraceFileNom, controlTraceFileNom)
        plotSol(N, stateTracesSmoothOp, controlTracesSmoothOp, stateTracesGP, controlTracesGP, stateTracesNom, controlTracesNom)

def satList(stateTraceFile, controlTraceFile):
    initialStates = getInitialStates(stateTraceFile)
    sat_count_list = []
    for initialState in initialStates:
        stateTraces, controlTraces = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
        recursively_feasible_count = 0
        sat_count_for_initial_state = 0
        for stateTrace in stateTraces:
            if stateTrace.shape[0] > N:
                recursively_feasible_count += 1
                disjunction_sat = [0]*(N+1)
                for i in range(N+1):
                    if stateTrace[i, 0] > C_th or stateTrace[i, 1] < T_th:
                        disjunction_sat[i] = 1
                if all(disjunction_sat) == 1:
                    sat_count_for_initial_state += 1
        
        sat_count_list += [sat_count_for_initial_state]

    return sat_count_list

def plotSol(N, x_smoothOp = [], u_smoothOp = [], x_GP = [], u_GP = [], x_Nom = [], u_Nom = []):
    nom_plotIdx = 0
    smoothOp_plotIdx = 1
    GP_plotIdx = 2
    fig, ax = plt.subplots(3, 3, constrained_layout=True)

    ax[0, smoothOp_plotIdx].set_xlim(0, T)
    ax[0, smoothOp_plotIdx].set_ylim(0, 0.2)
    ax[0, 0].set_ylabel('x1')
    for traj in x_smoothOp: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[0, smoothOp_plotIdx].plot(tgrid_i, traj[:, 0], alpha=0.5, linestyle='-', linewidth=1, color='xkcd:windows blue')
    ax[0, smoothOp_plotIdx].grid(True, which='both')

    ax[0, GP_plotIdx].set_xlim(0, T)
    ax[0, GP_plotIdx].set_ylim(0, 0.2)
    for traj in x_GP: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[0, GP_plotIdx].plot(tgrid_i, traj[:, 0], alpha=0.5, linestyle='-', linewidth=1, color='xkcd:orangish')
    ax[0, GP_plotIdx].grid(True)

    ax[0, nom_plotIdx].set_xlim(0, T)
    ax[0, nom_plotIdx].set_ylim(0, 0.2)
    for traj in x_Nom: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[0, nom_plotIdx].plot(tgrid_i, traj[:, 0], alpha=0.5, linestyle='-', linewidth=1, color='xkcd:windows blue')
    ax[0, nom_plotIdx].grid(True)

    ax[1, smoothOp_plotIdx].set_xlim(0, T)
    ax[1, smoothOp_plotIdx].set_ylim(0, 1.2)
    ax[1, 0].set_ylabel('x2')
    for traj in x_smoothOp: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[1, smoothOp_plotIdx].plot(tgrid_i, traj[:, 1], alpha=0.5, linestyle='-', linewidth=1, color='xkcd:windows blue')
    ax[1, smoothOp_plotIdx].grid(True)

    ax[1, GP_plotIdx].set_xlim(0, T)
    ax[1, GP_plotIdx].set_ylim(0, 1.2)
    for traj in x_GP: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[1, GP_plotIdx].plot(tgrid_i, traj[:, 1], alpha=0.5, linestyle='-', linewidth=1, color='xkcd:orangish')
    ax[1, GP_plotIdx].grid(True)

    ax[1, nom_plotIdx].set_xlim(0, T)
    ax[1, nom_plotIdx].set_ylim(0, 1.2)
    for traj in x_Nom: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[1, nom_plotIdx].plot(tgrid_i, traj[:, 1], alpha=0.5, linestyle='-', linewidth=1, color='xkcd:windows blue')
    ax[1, nom_plotIdx].grid(True)

    ax[2, smoothOp_plotIdx].set_xlim(0, T)
    ax[2, smoothOp_plotIdx].set_ylim(-1.2, 1.2)
    ax[2, 0].set_ylabel('u')
    ax[2, 0].set_xlabel('T (mins)')
    for traj in u_smoothOp: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]            
        ax[2, smoothOp_plotIdx].plot(tgrid_i, traj[:], '-', alpha=0.5, linewidth=1, color='xkcd:windows blue')
    ax[2, smoothOp_plotIdx].grid(True)

    ax[2, GP_plotIdx].set_xlim(0, T)
    ax[2, GP_plotIdx].set_ylim(-1.2, 1.2)
    ax[2, 1].set_xlabel('T (mins)')
    for traj in u_GP:
        tgrid_i = [T/N*k for k in range(traj.shape[0])]            
        ax[2, GP_plotIdx].plot(tgrid_i, traj[:], '-', alpha=0.5, linewidth=1, color='xkcd:orangish')
    ax[2, GP_plotIdx].grid(True)

    ax[2, nom_plotIdx].set_xlim(0, T)
    ax[2, nom_plotIdx].set_ylim(-1.2, 1.2)
    ax[2, 2].set_xlabel('T (mins)')
    for traj in u_Nom: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]            
        ax[2, nom_plotIdx].plot(tgrid_i, traj[:], '-', alpha=0.5, linewidth=1, color='xkcd:windows blue')
    ax[2, nom_plotIdx].grid(True)

def checkSAT(N, stateTraj):
    SAT = False

    lbbin = [0]
    ubbin = [1]

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

    for k in range(N+1):
        # STL constraints
        ZCk = MX.sym('ZC' + str(k), 1)
        w += [ZCk]
        lbw += lbbin
        ubw += ubbin
        w0 += [0]
        discrete += [True]

        ZTk = MX.sym('ZT' + str(k), 1)
        w += [ZTk]
        lbw += lbbin
        ubw += ubbin
        w0 += [0]
        discrete += [True]

        g += [stateTraj[k, 0] - C_th + bigMx1*(1 - ZCk)]
        g += [T_th - stateTraj[k, 1] + bigMx2*(1 - ZTk)] 

        g += [ZCk + ZTk]

        lbg += [0, 0, 1]
        ubg += [bigMx1, bigMx2, 2]
        
    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    qp_prob = {'f': J, 'x': w, 'g': g}
    qp_solver = qpsol('qp_solver', 'gurobi', qp_prob, {"discrete": discrete, "error_on_fail": False})

    # Solve the NLP
    sol = qp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(qp_solver.stats())

    solver_stats = qp_solver.stats()
    SAT = solver_stats['success']

    return SAT

def avgControlEffort(N, stateTraceFile, controlTraceFile):
    avgControlEffort = -1
    initialStates = getInitialStates(stateTraceFile)

    totalControlEffort = 0
    recursivelyFeasibleCount = 0
    for initialState in initialStates:
        stateTraces, controlTraces = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
        assert(len(stateTraces) == len(controlTraces))
        for i in range(len(stateTraces)):
            controlTrace = controlTraces[i]
            if controlTrace.shape[0] == N:
                recursivelyFeasibleCount += 1
                controlEffort = np.sum(np.square(controlTrace), axis=None)
                totalControlEffort += controlEffort
    
    if recursivelyFeasibleCount > 0:
        avgControlEffort = totalControlEffort/recursivelyFeasibleCount

    return avgControlEffort

satCountNom = satLoop(Nom_state_trace_file)
satCountGP = satLoop(GP_state_trace_file)
satCountSmoothOp = satLoop(smoothOp_state_trace_file)

def predictionRMSE(stateTraceFile, controlTraceFile, statePredictionFile):
    initialStates = getInitialStates(stateTraceFile)
    rmse = [-1]*2

    predictionErrors = [np.array([0]*2)]
    numDataPoints = 0
    for initialState in initialStates:
        stateTraces, controlTraces, predictionTraces = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile, predictionTraceFile=statePredictionFile)
        assert(len(stateTraces) == len(controlTraces) == len(predictionTraces))
        for i in range(len(stateTraces)):
            stateTrace = stateTraces[i]
            predictionTrace = predictionTraces[i]
            predictionError = (stateTrace[1:, :] - predictionTrace[1:, :])**2
            numDataPoints += predictionError.shape[0]
            predictionError = np.sum(predictionError, axis=0)
            predictionErrors += [predictionError]
    
    predictionErrors = np.vstack(predictionErrors)
    predictionErrors = np.sum(predictionErrors, axis=0)
    rmse = np.sqrt(predictionErrors/numDataPoints)
    return rmse

plotAllSol(N, smoothOp_state_trace_file, smoothOp_control_trace_file, GP_state_trace_file, GP_control_trace_file, Nom_state_trace_file, Nom_control_trace_file)

satListSmoothOp = satList(smoothOp_state_trace_file, smoothOp_control_trace_file)
satListGP = satList(GP_state_trace_file, GP_control_trace_file)
satListNom = satList(Nom_state_trace_file, Nom_control_trace_file)

avgControlEffortSmoothOp = avgControlEffort(N, smoothOp_state_trace_file, smoothOp_control_trace_file)
avgControlEffortGP = avgControlEffort(N, GP_state_trace_file, GP_control_trace_file)
avgControlEffortNom = avgControlEffort(N, Nom_state_trace_file, Nom_control_trace_file)

print("--STL sat--")
print("Smooth Op: ", str(satCountSmoothOp))
print("GP: ", str(satCountGP))
print("Nom: ", str(satCountNom))
print("---STL sat list---")
print("Smooth Op: ", str(satListSmoothOp))
print("GP: ", str(satListGP))
print("Nom: ", str(satListNom))
print("--Control effort--")
print("Smooth Op: ", str(avgControlEffortSmoothOp))
print("GP: ", str(avgControlEffortGP))
print("Nom: ", str(avgControlEffortNom))

smoothOp_state_trace_file.close()
smoothOp_control_trace_file.close()
GP_state_trace_file.close()
GP_control_trace_file.close()
Nom_state_trace_file.close()
Nom_control_trace_file.close()

plt.show()