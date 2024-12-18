import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import math

from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

import single_track_model_car as stcar

# load parameters
p = parameters_vehicle2()
g = 9.81  # [m/s^2]

# control model limits -----------------------------------------------------------------
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

# Horizon 
T = 4
N = 32

goal_x = [35, 100]
goal_y = [0, 3] 
obstacle_x = [20, 35]
obstacle_y = [0, 3]
goal_min_speed = 15
goal_carAng = [-1/20*math.pi, 1/20*math.pi]

prstlEpsilon = 0.1

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

def getTracesFor(init_x, init_y, stateTraceFile, controlTraceFile = None, solveTimeTraceFile = None):
    stateTraceFile.seek(0)
    
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
    if controlTraceFile is not None:
        controlTraceFile.seek(0)
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

    solveTimeTrace = []
    solveTimeTraces = []
    if solveTimeTraceFile is not None:
        solveTimeTraceFile.seek(0)

        for line in solveTimeTraceFile:
            if line == "~\n":
                if len(solveTimeTrace) > 0:
                    solveTimeTraces += [np.vstack(solveTimeTrace)]
                else:
                    solveTimeTraces += [np.array([])]
            else:
                solveTime = float(line.split("\n")[0])
                solveTimeTrace += [np.array([solveTime])] 

    targetStateTraces = []
    targetControlTraces = []
    targetSolveTimeTraces = []
    for i in range(len(stateTraces)):
        stateTrace = stateTraces[i]
        if stateTrace.size > 0 and stateTrace[0, 0] == init_x and stateTrace[0, 1] == init_y:
            if len(controlTraces) > 0:
                controlTrace = controlTraces[i]
                targetControlTraces += [controlTrace]
            if len(solveTimeTraces) > 0:
                solveTimeTrace = solveTimeTraces[i]
                targetSolveTimeTraces += [solveTimeTrace]
            targetStateTraces += [stateTrace]
    
    return targetStateTraces, targetControlTraces, targetSolveTimeTraces

def plotSol(N, plotControl, x_smoothOp = [], u_smoothOp = [], x_GP = [], u_GP = [], x_GP_offlineCovar = [], u_GP_offlineCovar = [], x_Nom = [], u_Nom = [], goal_A_polygon_x = [], goal_A_polygon_y = [], obstacle_polygon_x = [], obstacle_polygon_y = []):
    params = {'mathtext.default': 'regular',
              'pdf.fonttype' : 42}          
    plt.rcParams.update(params)
    nom_plotIdx = 0
    smoothOp_plotIdx = 1
    GP_plotIdx = 2
    GP_offlineCovar_plotIdx = 3
    
    fig, ax = plt.subplots(4, 1, constrained_layout=True)

    controller_label_x = 99
    controller_label_y = 4.5

    nom_color = 'xkcd:amethyst'
    smoothOp_color = 'xkcd:windows blue'
    lri_color = 'xkcd:orangish'

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
    ax[smoothOp_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[smoothOp_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[smoothOp_plotIdx].text(controller_label_x, controller_label_y, 'Smooth Robust', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_smoothOp:
        ax[smoothOp_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=smoothOp_color)
    ax[smoothOp_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[GP_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[GP_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[GP_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[GP_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[GP_plotIdx].text(controller_label_x, controller_label_y, 'LLRi', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_GP:
        ax[GP_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=lri_color)
    ax[GP_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[GP_offlineCovar_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[GP_offlineCovar_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[GP_offlineCovar_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[GP_offlineCovar_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[GP_offlineCovar_plotIdx].text(controller_label_x, controller_label_y, 'LLRi-PC', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_GP_offlineCovar:
        ax[GP_offlineCovar_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=lri_color)
    ax[GP_offlineCovar_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[nom_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[nom_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[nom_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[nom_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[nom_plotIdx].text(controller_label_x, controller_label_y, 'Nominal', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_Nom:
        ax[nom_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=nom_color)
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
        stateTracesSmoothOp, controlTracesSmoothOp, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileSmoothOp, controlTraceFileSmoothOp)
        stateTracesGP, controlTracesGP, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileGP, controlTraceFileGP)
        stateTraceGPofflineCovar, controlTraceGPofflineCovar, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileGPofflineCovar, controlTraceFileGPofflineCovar)
        stateTracesNom, controlTracesNom, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileNom, controlTraceFileNom)
        plotSol(N, plotControlTraj, stateTracesSmoothOp, controlTracesSmoothOp, stateTracesGP, controlTracesGP, stateTraceGPofflineCovar, controlTraceGPofflineCovar, stateTracesNom, controlTracesNom, goal_x, goal_y, obstacle_x, obstacle_y)

def plotSol1(N, plotControl, x_smoothOp = [], u_smoothOp = [], x_GP_offlineCovar = [], u_GP_offlineCovar = [], x_Nom = [], u_Nom = [], x_Oracle = [], u_Oracle = [], goal_A_polygon_x = [], goal_A_polygon_y = [], obstacle_polygon_x = [], obstacle_polygon_y = []):
    params = {'mathtext.default': 'regular',
              'pdf.fonttype' : 42 }          
    plt.rcParams.update(params)
    nom_plotIdx = 0
    smoothOp_plotIdx = 1
    GP_offlineCovar_plotIdx = 2
    oracle_plotIdx = 3
    
    fig, ax = plt.subplots(4, 1, constrained_layout=True)

    controller_label_x = 99
    controller_label_y = 4.5

    nom_color = 'xkcd:amethyst'
    smoothOp_color = 'xkcd:windows blue'
    lri_color = 'xkcd:orangish'
    oracle_color = 'xkcd:blue purple'

    if len(goal_A_polygon_x) > 0:
        goal_A_polygon_x_plot = [goal_A_polygon_x[0], goal_A_polygon_x[0], goal_A_polygon_x[1], goal_A_polygon_x[1]]
        goal_A_polygon_y_plot = [goal_A_polygon_y[1], goal_A_polygon_y[0], goal_A_polygon_y[0], goal_A_polygon_y[1]] 
        obstacle_polygon_x_plot = [obstacle_polygon_x[0], obstacle_polygon_x[0], obstacle_polygon_x[1], obstacle_polygon_x[1]]
        obstacle_polygon_y_plot = [obstacle_polygon_y[1], obstacle_polygon_y[0], obstacle_polygon_y[0], obstacle_polygon_y[1]]
        
        ax[smoothOp_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[smoothOp_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[smoothOp_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[smoothOp_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

        ax[GP_offlineCovar_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[GP_offlineCovar_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[GP_offlineCovar_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[GP_offlineCovar_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

        ax[nom_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[nom_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[nom_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[nom_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

        ax[oracle_plotIdx].fill(goal_A_polygon_x_plot, goal_A_polygon_y_plot, 'g', alpha=0.5)
        ax[oracle_plotIdx].plot(goal_A_polygon_x_plot + [goal_A_polygon_x[0]], goal_A_polygon_y_plot + [goal_A_polygon_y[1]], 'g')
        ax[oracle_plotIdx].fill(obstacle_polygon_x_plot, obstacle_polygon_y_plot, 'lightslategray', alpha=0.5)
        ax[oracle_plotIdx].plot(obstacle_polygon_x_plot + [obstacle_polygon_x[0]], obstacle_polygon_y_plot + [obstacle_polygon_y[1]], 'lightslategray')

    ax[smoothOp_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[smoothOp_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[smoothOp_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[smoothOp_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[smoothOp_plotIdx].text(controller_label_x, controller_label_y, 'Smooth Robust', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_smoothOp:
        ax[smoothOp_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=smoothOp_color)
    ax[smoothOp_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[GP_offlineCovar_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[GP_offlineCovar_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[GP_offlineCovar_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[GP_offlineCovar_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[GP_offlineCovar_plotIdx].text(controller_label_x, controller_label_y, 'LRi-A ($\epsilon_{' + str(prstlEpsilon) + '}$)', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_GP_offlineCovar:
        ax[GP_offlineCovar_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=lri_color)
    ax[GP_offlineCovar_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[nom_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[nom_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[nom_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[nom_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[nom_plotIdx].text(controller_label_x, controller_label_y, 'Nominal', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_Nom:
        ax[nom_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=nom_color)
    ax[nom_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

    ax[oracle_plotIdx].set_xlim(xlim[0], xlim[1])
    ax[oracle_plotIdx].set_ylim(ylim[0], ylim[1])
    ax[oracle_plotIdx].set_xlabel('$s_x\, (m)$')
    ax[oracle_plotIdx].set_ylabel('$s_y\, (m)$')
    ax[oracle_plotIdx].text(controller_label_x, controller_label_y, 'Oracle', horizontalalignment='right', verticalalignment='center', fontsize="x-large")

    for traj in x_Oracle:
        ax[oracle_plotIdx].plot(traj[:,0], traj[:,1], linestyle='-', linewidth=1, color=oracle_color)
    ax[oracle_plotIdx].scatter(traj[0,0], traj[0,1], s=120, facecolors='none', edgecolors='black')

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
            
        for controlTraj in u_GP_offlineCovar:
            tgrid = [k for k in range(len(controlTraj))]
            gpOfflineCovar_vSteerAng, = ax_u[vSteerAngIdx].plot(tgrid, controlTraj[:,0], '-o', alpha=0.5, linewidth=1, label="GP (offlineCovar)", color="green")
            gpOfflineCovar_accel, = ax_u[accelIdx].plot(tgrid, controlTraj[:, 1], '-o', alpha=0.5, linewidth=1, label="GP (offlineCovar)", color="green")

        ax_u[vSteerAngIdx].legend(handles=[nom_vSteerAng, gpOfflineCovar_vSteerAng])
        ax_u[accelIdx].legend(handles=[nom_accel, gpOfflineCovar_accel])

def plotAllSol1(N, plotControlTraj, stateTraceFileSmoothOp, controlTraceFileSmoothOp, stateTraceFileGPofflineCovar, controlTraceFileGPofflineCovar, stateTraceFileNom, controlTraceFileNom, stateTraceFileOracle, controlTraceFileOracle):
    initialStatesSmoothOp = getInitialStates(stateTraceFileSmoothOp)
    for initialState in initialStatesSmoothOp:
        stateTracesSmoothOp, controlTracesSmoothOp, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileSmoothOp, controlTraceFileSmoothOp)
        stateTraceGPofflineCovar, controlTraceGPofflineCovar, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileGPofflineCovar, controlTraceFileGPofflineCovar)
        stateTracesNom, controlTracesNom, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileNom, controlTraceFileNom)
        stateTracesOracle, controlTracesOracle, _ = getTracesFor(initialState[0], initialState[1], stateTraceFileOracle, controlTraceFileOracle)
        plotSol1(N, plotControlTraj, stateTracesSmoothOp, controlTracesSmoothOp, stateTraceGPofflineCovar, controlTraceGPofflineCovar, stateTracesNom, controlTracesNom, stateTracesOracle, controlTracesOracle, goal_x, goal_y, obstacle_x, obstacle_y)

def checkSAT(stateTraj, goal_A_polygon_x, goal_A_polygon_y, obstacle_polygon_x, obstacle_polygon_y):
    #This routine checks whether the trajectory satisfies the STL specification by using gurobi to try and assign integer variables, 
    #in hindsight not the most efficient way to check but it's been kept from the development process
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
        stateTraces, _, _ = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
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
    controlEffortStd = 0
    initialStates = getInitialStates(stateTraceFile)

    totalControlEffort = 0
    controlEffortList = []
    recursivelyFeasibleCount = 0
    bigEffortCount = 0 #used for debugging
    bigEffortSum = 0
    for initialState in initialStates:
        controlEffortListForInitialState = []
        stateTraces, controlTraces, _ = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
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
                controlEffortList += [controlEffort]
                controlEffortListForInitialState += [controlEffort]
        controlEffortListForInitialState = np.array(controlEffortListForInitialState)
        print("mean: ", str(controlEffortListForInitialState.mean()), " Std: ", str(controlEffortListForInitialState.std()))
        counts, bins = np.histogram(controlEffortListForInitialState, 10)
        print("Counts: ", str(counts))
        # fig =plt.figure()

        # plt.stairs(counts, bins)
        # plt.show()
                
    if recursivelyFeasibleCount > 0:
        avgControlEffort = totalControlEffort/recursivelyFeasibleCount

    if len(controlEffortList) > 0:
        controlEffortList = np.vstack(controlEffortList)
        avgControlEffort = controlEffortList.mean()
        controlEffortStd = controlEffortList.std()
        controlEffortQ75, controlEffortQ25 = np.percentile(controlEffortList, [75, 25])
        controlEffortIQR = controlEffortQ75 - controlEffortQ25
        controlEffortQ95, controlEffortQ5 = np.percentile(controlEffortList, [95, 5])
        controlEffortMed = np.median(controlEffortList)
        
    return avgControlEffort, controlEffortStd, controlEffortMed, controlEffortIQR, controlEffortQ95, controlEffortQ5, controlEffortList

def avgRobustness(stateTraceFile, controlTraceFile, goal_x, goal_y, obstacle_x, obstacle_y):
    initialStates = getInitialStates(stateTraceFile)
    total_trajectory_rho = 0
    recursively_feasible_count = 0

    for initialState in initialStates:
        stateTraces, controlTraces, _ = getTracesFor(initialState[0], initialState[1], stateTraceFile, controlTraceFile)
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

def timingStatsAll(timingTrace):
    timingAvg = -1
    timingStddev = 0

    timingTrace.seek(0)
    withinControlIntervalCount = 0

    solveTimes = []
    for line in timingTrace:
        if line != "~\n":
            solveTime = float(line.split("\n")[0])
            solveTimes.append(solveTime)
            if solveTime < T/N:
                withinControlIntervalCount += 1

    solveTimes = np.array(solveTimes)
    timingAvg = solveTimes.mean()
    timingStddev = solveTimes.std()
    timingQ75, timingQ25 = np.percentile(solveTimes, [75, 25])
    timingIQR = timingQ75 - timingQ25
    timingQ95, timingQ5 = np.percentile(solveTimes, [95, 5])
    timingMed = np.median(solveTimes)

    withinControlIntervalPercent = -1
    if len(solveTimes) > 0:
        withinControlIntervalPercent = withinControlIntervalCount/len(solveTimes)*100

    return timingAvg, timingStddev, timingMed, timingIQR, timingQ95, timingQ5, withinControlIntervalPercent

smoothOp_state_trace = open("./trace_data/autocar_state_traces_smoothOp.txt", "r")
smoothOp_control_trace = open("./trace_data/autocar_control_traces_smoothOp.txt", "r")
smoothOp_solveTime_trace = open("./trace_data/autocar_solveTime_traces_smoothOp.txt", "r")
oracle_smoothOp_state_trace = open("./trace_data/autocar_state_traces_oracle_smoothOp.txt", "r")
oracle_smoothOp_control_trace = open("./trace_data/autocar_control_traces_oracle_smoothOp.txt", "r")
oracle_smoothOp_solveTime_trace = open("./trace_data/autocar_solveTime_traces_oracle_smoothOp.txt", "r")
LTVGP_state_trace = open("./trace_data/autocar_state_trace_LTVGP.txt", "r")
LTVGP_control_trace = open("./trace_data/autocar_control_trace_LTVGP.txt", "r")
LTVGP_state_trace_offlineCovar = open("./trace_data/autocar_state_trace_LTVGP_offlineCovar_eps"+str(prstlEpsilon)+".txt", "r")
LTVGP_control_trace_offlineCovar = open("./trace_data/autocar_control_trace_LTVGP_offlineCovar_eps"+str(prstlEpsilon)+".txt", "r")
LTVGP_solveTime_trace_offlineCovar = open("./trace_data/autocar_solveTime_traces_LTVGP_offlineCovar_eps"+str(prstlEpsilon)+".txt", "r")
nom_state_trace = open("./trace_data/autocar_state_trace_nom.txt", "r")
nom_control_trace = open("./trace_data/autocar_control_trace_nom.txt", "r")
nom_solveTime_trace = open("./trace_data/autocar_solveTime_traces_nom.txt", "r")

# plotAllSol(N, False, smoothOp_state_trace, smoothOp_control_trace, LTVGP_state_trace, LTVGP_control_trace, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, nom_state_trace, nom_control_trace)
plotAllSol1(N, False, smoothOp_state_trace, smoothOp_control_trace, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, nom_state_trace, nom_control_trace, oracle_smoothOp_state_trace, oracle_smoothOp_control_trace)
# plt.show()

timingAvgSmooth, timingStdSmooth, timingMedSmooth, timingIqrSmooth, timing95Smooth, timing5Smooth, timingWithinThresholdPercentSmooth = timingStatsAll(smoothOp_solveTime_trace)
timingAvgOracleSmooth, timingStdOracleSmooth, timingMedOracleSmooth, timingIqrOracleSmooth, timing95OracleSmooth, timing5OracleSmooth, timingWithinThresholdPercentOracleSmooth = timingStatsAll(oracle_smoothOp_solveTime_trace)
timingAvgLTV_offlinecovar, timingStdLTV_offlinecovar, timingMedLTV_offlinecovar, timingIqrLTV_offlinecovar, timing95LTV_offlinecovar, timing5LTV_offlinecovar, timingWithinThresholdPercentLTV_offlinecovar = timingStatsAll(LTVGP_solveTime_trace_offlineCovar)
timingAvgNom, timingStdNom, timingMedNom, timingIqrNom, timing95Nom, timing5Nom, timingWithinThresholdPercentNom = timingStatsAll(nom_solveTime_trace)

avgRhoSmooth = avgRobustness(smoothOp_state_trace, smoothOp_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
avgRhoOracleSmooth = avgRobustness(oracle_smoothOp_state_trace, oracle_smoothOp_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
avgRhoLTV = avgRobustness(LTVGP_state_trace, LTVGP_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
avgRhoLTV_offlinecovar = avgRobustness(LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, goal_x, goal_y, obstacle_x, obstacle_y)
avgRhoNom = avgRobustness(nom_state_trace, nom_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)

avgControlEffortSmoothOp, controlEffortStdSmoothOp, controlEffortMedSmoothOp, controlEffortIqrSmoothOp, controlEffort95SmoothOp, controlEffort5SmoothOp, controlEffortListSmoothOp = avgControlEffort(N, smoothOp_state_trace, smoothOp_control_trace)
avgControlEffortOracleSmoothOp, controlEffortStdOracleSmoothOp, controlEffortMedOracleSmoothOp, controlEffortIqrOracleSmoothOp, controlEffort95OracleSmoothOp, controlEffort5OracleSmoothOp, controlEffortListOracleSmoothOp = avgControlEffort(N, oracle_smoothOp_state_trace, oracle_smoothOp_control_trace)
avgControlEffortLTV, controlEffortStdLTV, controlEffortMedLTV, controlEffortIqrLTV, controlEffort95LTV, controlEffort5LTV, _ = avgControlEffort(N, LTVGP_state_trace, LTVGP_control_trace)
avgControlEffortLTVofflineCovar, controlEffortStdLTVofflineCovar, controlEffortMedLTVofflinecovar, controlEffortIqrLTVofflinecovar, controlEffort95LTVofflinecovar, controlEffort5LTVofflinecovar, controlEffortListLTV_offlinecovar = avgControlEffort(N, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar)
avgControlEffortNom, controlEffortStdNom, controlEffortMedNom, controlEffortIqrNom, controlEffort95Nom, controlEffort5Nom, controlEffortListNom = avgControlEffort(N, nom_state_trace, nom_control_trace)

satSmoothOp, recursiveFeasibilitySmoothOp = satLoop(smoothOp_state_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satOracleSmoothOp, recursiveFeasibilityOracleSmoothOp = satLoop(oracle_smoothOp_state_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satGP, recursiveFeasibilityGP = satLoop(LTVGP_state_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satGPofflineCovar, recursiveFeasibilityGPofflineCovar = satLoop(LTVGP_state_trace_offlineCovar, goal_x, goal_y, obstacle_x, obstacle_y)
satNom, recursiveFeasibilityNom = satLoop(nom_state_trace, goal_x, goal_y, obstacle_x, obstacle_y)

satListSmoothOp = satList(N, smoothOp_state_trace, smoothOp_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satListOracleSmoothOp = satList(N, oracle_smoothOp_state_trace, oracle_smoothOp_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satListNom = satList(N, nom_state_trace, nom_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satListGP = satList(N, LTVGP_state_trace, LTVGP_control_trace, goal_x, goal_y, obstacle_x, obstacle_y)
satListGP_offlineCovar = satList(N, LTVGP_state_trace_offlineCovar, LTVGP_control_trace_offlineCovar, goal_x, goal_y, obstacle_x, obstacle_y)

smoothOp_state_trace.close()
smoothOp_control_trace.close()
smoothOp_solveTime_trace.close()
oracle_smoothOp_state_trace.close()
oracle_smoothOp_control_trace.close()
oracle_smoothOp_solveTime_trace.close()
LTVGP_state_trace.close()
LTVGP_control_trace.close()
LTVGP_state_trace_offlineCovar.close()
LTVGP_control_trace_offlineCovar.close()
LTVGP_solveTime_trace_offlineCovar.close()
nom_state_trace.close()
nom_control_trace.close()
nom_solveTime_trace.close()

print("---Sat stats---")
print("Smooth: ", str(satSmoothOp))
print("Smooth (oracle): ", str(satOracleSmoothOp))
print("GP: ", str(satGP))
print("GP offline covar: ", str(satGPofflineCovar))
print("Nom: ", str(satNom))

print("---Sat List---")
print("Smooth:, ", satListSmoothOp)
print("Smooth (oracle):, ", satListOracleSmoothOp)
print("GP: ", satListGP)
print("GP offline covar: ", satListGP_offlineCovar)
print("Nom: ", satListNom)

print("---Recursive feasibility---")
print("Smooth: ", str(recursiveFeasibilitySmoothOp))
print("Smooth (oracle): ", str(recursiveFeasibilityOracleSmoothOp))
print("GP: ", str(recursiveFeasibilityGP))
print("GP offline covar: ", str(recursiveFeasibilityGPofflineCovar))
print("Nom: ", str(recursiveFeasibilityNom))

print("---Control effort cost---")
print("Smooth Avg: ", str(avgControlEffortSmoothOp), " Std: ", str(controlEffortStdSmoothOp), " Med: ", str(controlEffortMedSmoothOp), " IQR: ", str(controlEffortIqrSmoothOp), " 95: ", str(controlEffort95SmoothOp), " 5: ", str(controlEffort5SmoothOp))
print("Smooth (oracle) Avg: ", str(avgControlEffortOracleSmoothOp), " Std: ", str(controlEffortStdOracleSmoothOp), " Med: ", str(controlEffortMedOracleSmoothOp), " IQR: ", str(controlEffortIqrOracleSmoothOp), " 95: ", str(controlEffort95OracleSmoothOp), " 5: ", str(controlEffort5OracleSmoothOp))
print("GP Avg: ", str(avgControlEffortLTV), " Std: ", str(controlEffortStdLTV), " Med: ", str(controlEffortMedLTV), " IQR: ", str(controlEffortIqrLTV), " 95: ", str(controlEffort95LTV), " 5: ", str(controlEffort5LTV))
print("GP offline covar Avg: ", str(avgControlEffortLTVofflineCovar), " Std: ", str(controlEffortStdLTVofflineCovar), " Med: ", str(controlEffortMedLTVofflinecovar), " IQR: ", str(controlEffortIqrLTVofflinecovar), " 95: ", str(controlEffort95LTVofflinecovar), " 5: ", str(controlEffort5LTVofflinecovar))
print("Nom Avg: ", str(avgControlEffortNom), " Std: ", str(controlEffortStdNom), " Med: ", str(controlEffortMedNom), " IQR: ", str(controlEffortIqrNom), " 95: ", str(controlEffort95Nom), " 5: ", str(controlEffort5Nom))

print("---Avg Robustness---")
print("Smooth: ", str(avgRhoSmooth))
print("Smooth (oracle): ", str(avgRhoOracleSmooth))
print("GP: ", str(avgRhoLTV))
print("GP offline: ", str(avgRhoLTV_offlinecovar))
print("Nom: ", str(avgRhoNom))

print("---Solve Time---")
print("Smooth Avg: ", str(timingAvgSmooth), " Std: ", str(timingStdSmooth), " Within Threshold: ", str(timingWithinThresholdPercentSmooth), " 95: ", str(timing95Smooth), " 5: ", str(timing5Smooth)) #" Med: ", str(timingMedSmooth), " IQR: ", str(timingIqrSmooth), " 95: ", str(timing95Smooth), " 5: ", str(timing5Smooth))
print("Smooth (oracle) Avg: ", str(timingAvgOracleSmooth), " Std: ", str(timingStdOracleSmooth), " Within Threshold: ", str(timingWithinThresholdPercentOracleSmooth), " 95: ", str(timing95OracleSmooth), " 5: ", str(timing5OracleSmooth)) #" Med: ", str(timingMedOracleSmooth), " IQR: ", str(timingIqrOracleSmooth), " 95: ", str(timing95OracleSmooth), " 5: ", str(timing5OracleSmooth))
print("GP offline covar Avg: ", str(timingAvgLTV_offlinecovar), " Std: ", str(timingStdLTV_offlinecovar), " Within Threshold: ", str(timingWithinThresholdPercentLTV_offlinecovar), " 95: ", str(timing95LTV_offlinecovar), " 5: ", str(timing5LTV_offlinecovar)) # " Med: ", str(timingMedLTV_offlinecovar), " IQR: ", str(timingIqrLTV_offlinecovar), " 95: ", str(timing95LTV_offlinecovar), " 5: ", str(timing5LTV_offlinecovar))
print("Nom Avg: ", str(timingAvgNom), " Std: ", str(timingStdNom), " Within Threshold: ", str(timingWithinThresholdPercentNom), " 95: ", str(timing95Nom), " 5: ", str(timing5Nom)) # " Med: ", str(timingMedNom), " IQR: ", str(timingIqrNom), " 95: ", str(timing95Nom), " 5: ", str(timing5Nom))

plt.show()