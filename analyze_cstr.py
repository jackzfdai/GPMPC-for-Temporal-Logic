import numpy as np
import matplotlib.pyplot as plt
from casadi import *

import cstr_model as cstr

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

prstlEpsilon = 0.01

smoothOp_state_trace_file = open("./trace_data/cstr_30x30_state_traces_smoothOp.txt", "r")
smoothOp_control_trace_file = open("./trace_data/cstr_30x30_control_traces_smoothOp.txt", "r")
smoothOp_solveTime_trace_file = open("./trace_data/cstr_30x30_solveTime_traces_smoothOp.txt", "r")
GP_state_trace_file = open("./trace_data/cstr_30x30_state_traces_GP_eps"+str(prstlEpsilon)+".txt", "r")
GP_control_trace_file = open("./trace_data/cstr_30x30_control_traces_GP_eps"+str(prstlEpsilon)+".txt", "r")
GP_solveTime_trace_file = open("./trace_data/cstr_30x30_solveTime_traces_GP_eps"+str(prstlEpsilon)+".txt", "r")
Nom_state_trace_file = open("./trace_data/cstr_30x30_state_traces_nom.txt", "r")
Nom_control_trace_file = open("./trace_data/cstr_30x30_control_traces_nom.txt", "r")
Nom_solveTime_trace_file = open("./trace_data/cstr_30x30_solveTime_traces_nom.txt", "r")

oracle_smoothOp_state_trace_file = open("./trace_data/cstr_30x30_state_traces_oracle_smoothOp.txt", "r")
oracle_smoothOp_control_trace_file = open("./trace_data/cstr_30x30_control_traces_oracle_smoothOp.txt", "r")
oracle_smoothOp_solveTime_trace_file = open("./trace_data/cstr_30x30_solveTime_traces_oracle_smoothOp.txt", "r")

def satLoop(stateTraceFile):
    stateTraceFile.seek(0)
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

def plotAllSol(N, stateTraceFileSmoothOp, controlTraceFileSmoothOp, stateTraceFileGP, controlTraceFileGP, stateTraceFileNom, controlTraceFileNom, stateTraceFileOracle, controlTraceFileOracle):
    initialStatesSmoothOp = getInitialStates(stateTraceFileSmoothOp)
    initialStatesGP = getInitialStates(stateTraceFileGP)
    initialStatesNom = getInitialStates(stateTraceFileNom)
    for initialState in initialStatesSmoothOp:
        stateTracesSmoothOp, controlTracesSmoothOp = getTracesFor(initialState[0], initialState[1], stateTraceFileSmoothOp, controlTraceFileSmoothOp)
        stateTracesGP, controlTracesGP = getTracesFor(initialState[0], initialState[1], stateTraceFileGP, controlTraceFileGP)
        stateTracesNom, controlTracesNom = getTracesFor(initialState[0], initialState[1], stateTraceFileNom, controlTraceFileNom)
        stateTracesOracle, controlTracesOracle = getTracesFor(initialState[0], initialState[1], stateTraceFileOracle, controlTraceFileOracle)
        plotSol(N, stateTracesSmoothOp, controlTracesSmoothOp, stateTracesGP, controlTracesGP, stateTracesNom, controlTracesNom, stateTracesOracle, controlTracesOracle)

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

def plotSol(N, x_smoothOp = [], u_smoothOp = [], x_GP = [], u_GP = [], x_Nom = [], u_Nom = [], x_Oracle = [], u_Oracle = []):
    params = {'mathtext.default': 'regular',
              'pdf.fonttype' : 42}          
    plt.rcParams.update(params)
    nom_plotIdx = 0
    smoothOp_plotIdx = 1
    GP_plotIdx = 2
    oracle_plotIdx = 3
    fig, ax = plt.subplots(3, 4, constrained_layout=True)

    nom_color = 'xkcd:amethyst'
    smoothOp_color = 'xkcd:windows blue'
    lri_color = 'xkcd:orangish'
    oracle_color = 'xkcd:blue purple'

    controller_label_x = 0
    controller_label_y = 0.2

    # ax[0, 1].set_yticklabels([])
    # ax[0, 2].set_yticklabels([])
    # ax[1, 1].set_yticklabels([])
    # ax[1, 2].set_yticklabels([])

    ax[0, smoothOp_plotIdx].set_xlim(0, T)
    ax[0, smoothOp_plotIdx].set_ylim(0, 0.2)
    ax[0, 0].set_ylabel('$Composition, x_1$')
    for traj in x_smoothOp: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[0, smoothOp_plotIdx].plot(tgrid_i, traj[:, 0], alpha=0.5, linestyle='-', linewidth=1, color=smoothOp_color)
    ax[0, smoothOp_plotIdx].grid(True, which='both')
    ax[0, smoothOp_plotIdx].text(controller_label_x, controller_label_y, 'Smooth Robust', horizontalalignment='left', verticalalignment='bottom', fontsize="x-large")

    ax[0, GP_plotIdx].set_xlim(0, T)
    ax[0, GP_plotIdx].set_ylim(0, 0.2)
    for traj in x_GP: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[0, GP_plotIdx].plot(tgrid_i, traj[:, 0], alpha=0.5, linestyle='-', linewidth=1, color=lri_color)
    ax[0, GP_plotIdx].grid(True)
    ax[0, GP_plotIdx].text(controller_label_x, controller_label_y, 'LRi ($\epsilon_{' + str(prstlEpsilon) + '0}$)', horizontalalignment='left', verticalalignment='bottom', fontsize="x-large")

    ax[0, nom_plotIdx].set_xlim(0, T)
    ax[0, nom_plotIdx].set_ylim(0, 0.2)
    for traj in x_Nom: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[0, nom_plotIdx].plot(tgrid_i, traj[:, 0], alpha=0.5, linestyle='-', linewidth=1, color=nom_color)
    ax[0, nom_plotIdx].grid(True)
    ax[0, nom_plotIdx].text(controller_label_x, controller_label_y, 'Nominal', horizontalalignment='left', verticalalignment='bottom', fontsize="x-large")

    ax[0, oracle_plotIdx].set_xlim(0, T)
    ax[0, oracle_plotIdx].set_ylim(0, 0.2)
    for traj in x_Oracle: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[0, oracle_plotIdx].plot(tgrid_i, traj[:, 0], alpha=0.5, linestyle='-', linewidth=1, color=oracle_color)
    ax[0, oracle_plotIdx].grid(True)
    ax[0, oracle_plotIdx].text(controller_label_x, controller_label_y, 'Oracle', horizontalalignment='left', verticalalignment='bottom', fontsize="x-large")

    ax[1, smoothOp_plotIdx].set_xlim(0, T)
    ax[1, smoothOp_plotIdx].set_ylim(0, 1.2)
    ax[1, 0].set_ylabel('$Temperature, x_2$')
    for traj in x_smoothOp: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[1, smoothOp_plotIdx].plot(tgrid_i, traj[:, 1], alpha=0.5, linestyle='-', linewidth=1, color=smoothOp_color)
    ax[1, smoothOp_plotIdx].grid(True)

    ax[1, GP_plotIdx].set_xlim(0, T)
    ax[1, GP_plotIdx].set_ylim(0, 1.2)
    for traj in x_GP: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[1, GP_plotIdx].plot(tgrid_i, traj[:, 1], alpha=0.5, linestyle='-', linewidth=1, color=lri_color)
    ax[1, GP_plotIdx].grid(True)

    ax[1, nom_plotIdx].set_xlim(0, T)
    ax[1, nom_plotIdx].set_ylim(0, 1.2)
    for traj in x_Nom: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[1, nom_plotIdx].plot(tgrid_i, traj[:, 1], alpha=0.5, linestyle='-', linewidth=1, color=nom_color)
    ax[1, nom_plotIdx].grid(True)

    ax[1, oracle_plotIdx].set_xlim(0, T)
    ax[1, oracle_plotIdx].set_ylim(0, 1.2)
    for traj in x_Oracle: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]
        ax[1, oracle_plotIdx].plot(tgrid_i, traj[:, 1], alpha=0.5, linestyle='-', linewidth=1, color=oracle_color)
    ax[1, oracle_plotIdx].grid(True)

    ax[2, smoothOp_plotIdx].set_xlim(0, T)
    ax[2, smoothOp_plotIdx].set_ylim(-1.2, 1.2)
    ax[2, 0].set_ylabel('Cooling Input, u')
    ax[2, smoothOp_plotIdx].set_xlabel('T (mins)')
    for traj in u_smoothOp: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]            
        ax[2, smoothOp_plotIdx].plot(tgrid_i, traj[:], '-', alpha=0.5, linewidth=1, color=smoothOp_color)
    ax[2, smoothOp_plotIdx].grid(True)

    ax[2, GP_plotIdx].set_xlim(0, T)
    ax[2, GP_plotIdx].set_ylim(-1.2, 1.2)
    ax[2, GP_plotIdx].set_xlabel('T (mins)')
    # ax[2, 1].set_yticklabels([])
    for traj in u_GP:
        tgrid_i = [T/N*k for k in range(traj.shape[0])]            
        ax[2, GP_plotIdx].plot(tgrid_i, traj[:], '-', alpha=0.5, linewidth=1, color=lri_color)
    ax[2, GP_plotIdx].grid(True)

    ax[2, nom_plotIdx].set_xlim(0, T)
    ax[2, nom_plotIdx].set_ylim(-1.2, 1.2)
    ax[2, nom_plotIdx].set_xlabel('T (mins)')
    # ax[2, 2].set_yticklabels([])
    for traj in u_Nom: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]            
        ax[2, nom_plotIdx].plot(tgrid_i, traj[:], '-', alpha=0.5, linewidth=1, color=nom_color)
    ax[2, nom_plotIdx].grid(True)

    ax[2, oracle_plotIdx].set_xlim(0, T)
    ax[2, oracle_plotIdx].set_ylim(-1.2, 1.2)
    ax[2, oracle_plotIdx].set_xlabel('T (mins)')
    # ax[2, 2].set_yticklabels([])
    for traj in u_Oracle: 
        tgrid_i = [T/N*k for k in range(traj.shape[0])]            
        ax[2, oracle_plotIdx].plot(tgrid_i, traj[:], '-', alpha=0.5, linewidth=1, color=oracle_color)
    ax[2, oracle_plotIdx].grid(True)

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
    controlEffortStd = 0
    initialStates = getInitialStates(stateTraceFile)

    controlEffortList = []
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
                controlEffortList += [controlEffort]
    
    if recursivelyFeasibleCount > 0:
        avgControlEffort = totalControlEffort/recursivelyFeasibleCount

    if len(controlEffortList) > 0:
        controlEffortList = np.vstack(controlEffortList)
        avgControlEffort = controlEffortList.mean()
        controlEffortStd = controlEffortList.std()
        controlEffortQ75, controlEffortQ25 = np.percentile(controlEffortList, [75, 25])
        controlEffortIQR = controlEffortQ75 - controlEffortQ25
        controlEffortQ95, controlEffortQ5 = np.percentile(controlEffortList, [95, 5])

    return avgControlEffort, controlEffortStd, controlEffortIQR, controlEffortQ95, controlEffortQ5

def timingStatsAll(timingTrace):
    timingAvg = -1
    timingStddev = 0
    withinThresholdCount = 0

    timingTrace.seek(0)

    solveTimes = []
    for line in timingTrace:
        if line != "~\n":
            solveTime = float(line.split("\n")[0])
            solveTimes.append(solveTime)
            if solveTime < T/N*60: 
                withinThresholdCount += 1

    solveTimes = np.array(solveTimes)
    timingAvg = solveTimes.mean()
    timingStddev = solveTimes.std()
    timingQ75, timingQ25 = np.percentile(solveTimes, [75, 25])
    timingIQR = timingQ75 - timingQ25
    timingQ95, timingQ5 = np.percentile(solveTimes, [95, 5])

    if len(solveTimes) > 0:
        withinThresholdPercent = withinThresholdCount/len(solveTimes)*100

    
    return timingAvg, timingStddev, timingIQR, withinThresholdPercent, timingQ95, timingQ5

plotAllSol(N, smoothOp_state_trace_file, smoothOp_control_trace_file, GP_state_trace_file, GP_control_trace_file, Nom_state_trace_file, Nom_control_trace_file, oracle_smoothOp_state_trace_file, oracle_smoothOp_control_trace_file)
# plt.show()

solveTimeAvgNom, solveTimeStdNom, solveTimeIqrNom, withinIntervalPercentNom, solveTime95Nom, solveTime5Nom = timingStatsAll(Nom_solveTime_trace_file)
solveTimeAvgSmoothOp, solveTimeStdSmoothOp, solveTimeIqrSmoothOp, withinIntervalPercentSmoothOp, solveTime95SmoothOp, solveTime5SmoothOp = timingStatsAll(smoothOp_solveTime_trace_file)
solveTimeAvgGP, solveTimeStdGP, solveTimeIqrGP, withinIntervalPercentGP, solveTime95GP, solveTime5GP = timingStatsAll(GP_solveTime_trace_file)
solveTimeAvgOracleSmoothOp, solveTimeStdOracleSmoothOp, solveTimeIqrOracleSmoothOp, withinIntervalPercentOracleSmoothOp, solveTime95OracleSmoothOp, solveTime5OracleSmoothOp = timingStatsAll(oracle_smoothOp_solveTime_trace_file)

satCountNom = satLoop(Nom_state_trace_file)
satCountGP = satLoop(GP_state_trace_file)
satCountSmoothOp = satLoop(smoothOp_state_trace_file)
satCountOracleSmoothOp = satLoop(oracle_smoothOp_state_trace_file)

satListSmoothOp = satList(smoothOp_state_trace_file, smoothOp_control_trace_file)
satListGP = satList(GP_state_trace_file, GP_control_trace_file)
satListNom = satList(Nom_state_trace_file, Nom_control_trace_file)
satListOracleSmoothOp = satList(oracle_smoothOp_state_trace_file, oracle_smoothOp_control_trace_file)

avgControlEffortSmoothOp, controlEffortStdSmoothOp, controlEffortIqrSmoothOp, controlEffort95SmoothOp, controlEffort5SmoothOp = avgControlEffort(N, smoothOp_state_trace_file, smoothOp_control_trace_file)
avgControlEffortGP, controlEffortStdGP, controlEffortIqrGP, controlEffort95GP, controlEffort5GP = avgControlEffort(N, GP_state_trace_file, GP_control_trace_file)
avgControlEffortNom, controlEffortStdNom, controlEffortIqrNom, controlEffort95Nom, controlEffort5Nom = avgControlEffort(N, Nom_state_trace_file, Nom_control_trace_file)
avgControlEffortOracleSmoothOp, controlEffortStdOracleSmoothOp, controlEffortIqrOracleSmoothOp, controlEffort95OracleSmoothOp, controlEffort5OracleSmoothOp = avgControlEffort(N, oracle_smoothOp_state_trace_file, oracle_smoothOp_control_trace_file)

print("--STL sat--")
print("Smooth Op: ", str(satCountSmoothOp))
print("Smooth Op (oracle): ", str(satCountOracleSmoothOp))
print("LRi (GP): ", str(satCountGP))
print("Nom: ", str(satCountNom))
print("---STL sat list---")
print("Smooth Op: ", str(satListSmoothOp))
print("Smooth Op (oracle): ", str(satListOracleSmoothOp))
print("LRi (GP): ", str(satListGP))
print("Nom: ", str(satListNom))
print("--_Control effort--_")
print("Smooth Op Avg: ", str(avgControlEffortSmoothOp), " Std: ", str(controlEffortStdSmoothOp), " IQR: ", str(controlEffortIqrSmoothOp), " 95: ", str(controlEffort95SmoothOp), " 5: ", str(controlEffort5SmoothOp))
print("Smooth Op (oracle) Avg: ", str(avgControlEffortOracleSmoothOp), " Std: ", str(controlEffortStdOracleSmoothOp), " IQR: ", str(controlEffortIqrOracleSmoothOp), " 95: ", str(controlEffort95OracleSmoothOp), " 5: ", str(controlEffort5OracleSmoothOp))
print("LRi (GP) Avg: ", str(avgControlEffortGP), " Std: ", str(controlEffortStdGP), " IQR: ", str(controlEffortIqrGP), " 95: ", str(controlEffort95GP), " 5: ", str(controlEffort5GP))
print("Nom Avg: ", str(avgControlEffortNom), " Std: ", str(controlEffortStdNom), " IQR: ", str(controlEffortIqrNom), " 95: ", str(controlEffort95Nom), " 5: ", str(controlEffort5Nom))
print("---Solve Time---")
print("Smooth Op Avg: ", str(solveTimeAvgSmoothOp), " Std: ", str(solveTimeStdSmoothOp), " IQR: ", str(solveTimeIqrSmoothOp), " WithinThreshold: ", str(withinIntervalPercentSmoothOp), " 95: ", str(solveTime95SmoothOp), " 5: ", str(solveTime5SmoothOp))
print("Smooth Op (oracle) Avg: ", str(solveTimeAvgOracleSmoothOp), " Std: ", str(solveTimeStdOracleSmoothOp), " IQR: ", str(solveTimeIqrOracleSmoothOp), " WithinThreshold: ", withinIntervalPercentOracleSmoothOp, " 95: ", str(solveTime95OracleSmoothOp), " 5: ", str(solveTime5OracleSmoothOp))
print("LRi (GP) Avg: ", str(solveTimeAvgGP), " Std: ", str(solveTimeStdGP), " IQR: ", str(solveTimeIqrGP), " WithinThreshold: ", str(withinIntervalPercentGP), " 95: ", str(solveTime95GP), " 5: ", str(solveTime5GP))
print("Nom Avg: ", str(solveTimeAvgNom), " Std: ", str(solveTimeStdNom), " IQR: ", str(solveTimeIqrNom), " WithinThreshold: ", str(withinIntervalPercentNom), " 95: ", str(solveTime95Nom), " 5: ", str(solveTime5Nom))

smoothOp_state_trace_file.close()
smoothOp_control_trace_file.close()
smoothOp_solveTime_trace_file.close()
oracle_smoothOp_state_trace_file.close()
oracle_smoothOp_control_trace_file.close()
oracle_smoothOp_solveTime_trace_file.close()
GP_state_trace_file.close()
GP_control_trace_file.close()
GP_solveTime_trace_file.close()
Nom_state_trace_file.close()
Nom_control_trace_file.close()
Nom_solveTime_trace_file.close()

plt.show()