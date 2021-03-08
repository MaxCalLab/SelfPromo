# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:47:25 2017

@author: tefirman
"""

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

simConditions = 'SelfPromoDimer' # Name of simulation conditions
timeInc = 300 # Time increment of simulated expression traces
numTrials = 100 # Number of traces being simulated
numDays = 7 # Time length of simulated expression traces
maxn = 5 # Maximum number of mRNA allowed (FSP requirement)
maxN = 92 # Maximum number of protein allowed (FSP requirement)
maxN2 = 5 # Maximum number of dimers allowed (FSP requirement)
numIterations = 'Avg' # Length of multi-frame transition to use in ML inference (m)
#numIterations = 1 # Length of multi-frame transition to use in ML inference (m)
simNum = 10 # Replicate number, just used for naming
numFits = 10 # Number of parameter fittings

def conditionsInitGill(g,g_pro,d,p,r,k_fD,k_bD,k_fP,k_bP,exclusive,\
n_A_init,n_A2_init,n_a_init,n_alpha_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all relevant rates and conditions for the self-promotion (dimer) circuit """
    return {'g':g, 'g_pro':g_pro, 'd':d, 'p':p, 'r':r, 'k_fD':k_fD, 'k_bD':k_bD, \
    'k_fP':k_fP, 'k_bP':k_bP, 'exclusive':exclusive, 'N_A_init':n_A_init, \
    'N_A2_init':n_A2_init, 'N_a_init':n_a_init, 'N_alpha_init':n_alpha_init, \
    'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def gillespieSim_SelfPromo(conditions,n_A=[],n_A2=[],n_a=[],n_alpha=[]):
    """ Simulates self-promotion (dimer) gene expression trajectories given the provided conditions """
    """ Can continue simulations if provided in n_A, n_A2, n_a, and n_alpha """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_A2 = (conditions['N_A2_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_a = (conditions['N_a_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_alpha = (conditions['N_alpha_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        if len(n_a) < numTrial + 1:
            numA = np.copy(conditions['N_A_init'])
            numA2 = np.copy(conditions['N_A2_init'])
            numa = np.copy(conditions['N_a_init'])
            numalpha = np.copy(conditions['N_alpha_init'])
            n_A.append([np.copy(numA)])
            n_A2.append([np.copy(numA2)])
            n_a.append([np.copy(numa)])
            n_alpha.append([np.copy(numalpha)])
        else:
            numA = np.copy(n_A[numTrial][-1])
            numA2 = np.copy(n_A2[numTrial][-1])
            numa = np.copy(n_a[numTrial][-1])
            numalpha = np.copy(n_alpha[numTrial][-1])
        timeFrame = (len(n_A[numTrial]) - 1)*conditions['inc']
        incCheckpoint = len(n_A[numTrial])*conditions['inc']
        print('Trial #' + str(numTrial + 1))
        while timeFrame < float(conditions['numSteps']*conditions['inc']):
            prob = [conditions['g']*(1 - numalpha),\
            conditions['g_pro']*numalpha,\
            conditions['d']*numa,\
            conditions['p']*numa,\
            conditions['r']*numA,\
            conditions['k_fD']*numA*(numA - 1),\
            conditions['k_bD']*numA2,\
            conditions['k_fP']*(1 - numalpha)*numA2,\
            conditions['k_bP']*numalpha]
            overallRate = sum(prob)
            randNum1 = np.random.rand(1)
            timeFrame -= np.log(randNum1)/overallRate
            while timeFrame >= incCheckpoint:
                n_A[numTrial].append(np.copy(numA).tolist())
                n_A2[numTrial].append(np.copy(numA2).tolist())
                n_a[numTrial].append(np.copy(numa).tolist())
                n_alpha[numTrial].append(np.copy(numalpha).tolist())
                incCheckpoint += conditions['inc']
                if incCheckpoint%10000 == 0:
                    print('Time = ' + str(incCheckpoint) + ' seconds')
            prob = prob/overallRate
            randNum2 = np.random.rand(1)
            if randNum2 <= sum(prob[:2]):
                numa += 1
            elif randNum2 <= sum(prob[:3]):
                numa -= 1
            elif randNum2 <= sum(prob[:4]):
                numA += 1
            elif randNum2 <= sum(prob[:5]):
                numA -= 1
            elif randNum2 <= sum(prob[:6]):
                numA2 += 1
                numA -= 2
            elif randNum2 <= sum(prob[:7]):
                numA2 -= 1
                numA += 2
            elif randNum2 <= sum(prob[:8]):
                numalpha += 1
                numA2 -= 1
            else:
                numalpha -= 1
                numA2 += 1
        n_A[numTrial] = n_A[numTrial][:conditions['numSteps'] + 1]
        n_A2[numTrial] = n_A2[numTrial][:conditions['numSteps'] + 1]
        n_a[numTrial] = n_a[numTrial][:conditions['numSteps'] + 1]
        n_alpha[numTrial] = n_alpha[numTrial][:conditions['numSteps'] + 1]
    return n_A, n_A2, n_a, n_alpha

def peakVals(origHist,numFilter=0,minVal=0):
    """ Identifies the locations of the high and low states in given protein number distribution """
    """ Can also provide an averaging filter as well as a minimum probability threshold """
    simHist = np.copy(origHist)
    for numTry in range(numFilter):
        for ind in range(1,len(simHist) - 1):
            simHist[ind] = np.sum(simHist[max(ind - 1,0):min(ind + 2,len(simHist))])/\
            np.size(simHist[max(ind - 1,0):min(ind + 2,len(simHist))])
    maxInds = np.where(np.all([simHist[2:] - simHist[1:-1] < 0,\
    simHist[1:-1] - simHist[:-2] > 0,simHist[1:-1] >= minVal],axis=0))[0] + 1
    return maxInds

def entropyStats(n_A,maxInds):
    """ Calculates the relevant entropy stats given expression trajectories and low/high state locations """
    global maxN
    stateProbs = [np.zeros((maxN,maxN)) for ind in range(len(maxInds))]
    cgProbs = np.zeros((len(maxInds),len(maxInds)))
    dwellVals = [[] for ind in range(len(maxInds))]
    for numTrial in range(len(n_A)):
        cgTraj = -1*np.ones(len(n_A[numTrial]))
        for ind in range(len(maxInds)):
            cgTraj[n_A[numTrial] == maxInds[ind]] = ind
        ind1 = np.where(cgTraj >= 0)[0][0]
        inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        while len(inds) > 0:
            stateProbs[int(cgTraj[ind1])] += np.histogram2d(n_A[numTrial][ind1 + 1:ind1 + inds[0]],\
            n_A[numTrial][ind1:ind1 + inds[0] - 1],bins=np.arange(-0.5,maxN))[0]
            cgProbs[int(cgTraj[ind1 + inds[0]]),int(cgTraj[ind1])] += 1
            cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += inds[0]
            dwellVals[int(cgTraj[ind1])].append(inds[0])
            ind1 += inds[0]
            inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        stateProbs[int(cgTraj[ind1])] += np.histogram2d(n_A[numTrial][ind1 + 1:],\
        n_A[numTrial][ind1:-1],bins=np.arange(-0.5,maxN))[0]
        cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += len(n_A[numTrial]) - ind1 - 1
    totProbs = np.zeros((maxN,maxN))
    stateEntropies = []
    for ind in range(len(stateProbs)):
        totProbs += stateProbs[ind]
        stateProbs[ind] = stateProbs[ind]/np.sum(stateProbs[ind])
        stateEntropies.append(-np.nansum(stateProbs[ind]*np.log2(stateProbs[ind])))
    totProbs = totProbs/np.sum(totProbs)
    totEntropy = -np.nansum(totProbs*np.log2(totProbs))
    cgProbs = cgProbs/np.sum(cgProbs)
    macroEntropy = -np.nansum(cgProbs*np.log2(cgProbs))
    return totEntropy, stateEntropies, macroEntropy, dwellVals

def conditionsInitHill(g,g_pro,n,K,r,n_A_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all relevant rates and conditions for the Hill function circuit """
    return {'g':float(g), 'g_pro':float(g_pro), 'n':float(n), 'K':K, 'r':r, \
    'N_A_init':n_A_init, 'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def hillSim(conditions,n_A=[]):
    """ Simulates Hill function gene expression trajectories given the provided conditions """
    """ Can continue simulations if provided in n_A """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    if len(n_A) == 0:
        n_A = conditions['N_A_init']*np.ones((conditions['numTrials'],1))
        n_A = n_A.tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        if len(n_A) < numTrial + 1:
            numA = np.copy(conditions['N_A_init'])
            n_A.append([np.copy(numA)])
        else:
            numA = np.copy(n_A[numTrial][-1])
        timeFrame = (len(n_A[numTrial]) - 1)*conditions['inc']
        incCheckpoint = len(n_A[numTrial])*conditions['inc']
        print('Trial #' + str(numTrial + 1))
        while timeFrame < float(conditions['numSteps']*conditions['inc']):
            prob = [conditions['g'] + conditions['g_pro']*(numA**conditions['n'])/\
            (numA**conditions['n'] + conditions['K']),conditions['r']*numA]
            overallRate = sum(prob)
            randNum1 = np.random.rand(1)
            timeFrame -= np.log(randNum1)/overallRate
            while timeFrame >= incCheckpoint:
                n_A[numTrial].append(np.copy(numA))
                incCheckpoint += conditions['inc']
            prob = prob/overallRate
            randNum2 = np.random.rand(1)
            if randNum2 < prob[0]:
                numA += 1
            else:
                numA -= 1
        n_A[numTrial] = n_A[numTrial][:conditions['numSteps'] + 1]
    return n_A

def hillFSP(conditions):
    """ Runs a Finite State Projection (FSP) of the Hill function circuit using the provided conditions """
    global maxN
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probMatrix[probN][probN] -= (conditions['g'] + conditions['g_pro']*(probN**conditions['n'])/(probN**conditions['n'] + conditions['K']) + conditions['r']*probN)
        if probN < maxN - 1:
            probMatrix[probN + 1][probN] += conditions['g'] + conditions['g_pro']*(probN**conditions['n'])/(probN**conditions['n'] + conditions['K'])
        if probN > 0:
            probMatrix[probN - 1][probN] += conditions['r']*probN
    probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    print(probMatrix.shape)
    print(probMatrix.dtype)
    print('# of bytes: ' + str(probMatrix.nbytes))
    probInit = np.zeros(len(probMatrix))
    probInit[conditions['N_A_init']] = 1.0

    """ Calculating matrix exponential time and non-sink state probability for illustrative purposes """
    timeVals = []
    for numTry in range(100):
        startTime = datetime.datetime.now()
        probFinal = np.dot(scipy.linalg.expm(probMatrix*conditions['inc']*conditions['numSteps']),probInit)
        endTime = datetime.datetime.now()
        timeVals.append((endTime - startTime).total_seconds())
    print('Exponential Time: ' + str(round(np.average(timeVals),5)) + ' +/- ' + str(round(np.std(timeVals),5)) + ' seconds')
    print('Probability: ' + str(np.sum(probFinal[:-1])))
    """ Calculating matrix exponential time and non-sink state probability for illustrative purposes """

    return probFinal

def hillFSP_mle(reactionRates):
    """ Calculates the likelihood of the transitions provided in the global variable "probs" given the rates provided """
    """ Strange function formatting is necessary to interface with scipy.minimize """
    global probs
    global maxN
    global n
    global numIterations
    global timeInc
    global numBytes
    global stepTime
    global matPop
    if np.any(np.array(reactionRates) < 0):
        return float('Inf')
    
    startTime = datetime.datetime.now()
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probMatrix[probN][probN] -= (reactionRates[0] + reactionRates[1]*(probN**n)/(probN**n + reactionRates[2]) + reactionRates[3]*probN)
        if probN < maxN - 1:
            probMatrix[probN + 1][probN] += reactionRates[0] + reactionRates[1]*(probN**n)/(probN**n + reactionRates[2])
        if probN > 0:
            probMatrix[probN - 1][probN] += reactionRates[3]*probN
    probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    endTime = datetime.datetime.now()
    matPop.append((endTime - startTime).total_seconds())
    """ Measuring matrix population and exponentiation for illustrative purposes """
    numBytes.append(probMatrix.nbytes)
    startTime = datetime.datetime.now()
    tempMatrix = scipy.linalg.expm(probMatrix*numIterations*timeInc)
    endTime = datetime.datetime.now()
    stepTime.append((endTime - startTime).total_seconds())

    loglike = -1*np.nansum(np.log(tempMatrix)*probs.toarray())
#    print('g = ' + str(round(reactionRates[0],5)) + ', g_pro = ' + str(round(reactionRates[1],5)) + \
#    ', K = ' + str(round(reactionRates[2],1)) + ', r = ' + str(round(reactionRates[3],5)) + \
#    ', loglike = ' + str(round(loglike,1)))
    return loglike

def hillEquil_mse(reactionRates):
    """ Calculating the mean squared error of protein number distributions given the rates provided """
    """ Method not provided in papers, just used to find a starting point for rate estimation """
    global simHist_Gill
    if np.any(np.array(reactionRates) < 0):
        return float('Inf')
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probMatrix[probN][probN] -= (reactionRates[0] + reactionRates[1]*(probN**n)/(probN**n + reactionRates[2]) + reactionRates[3]*probN)
        if probN < maxN - 1:
            probMatrix[probN + 1][probN] += reactionRates[0] + reactionRates[1]*(probN**n)/(probN**n + reactionRates[2])
        if probN > 0:
            probMatrix[probN - 1][probN] += reactionRates[3]*probN
    probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    tempProb = np.dot(scipy.linalg.expm(probMatrix*1000000),np.append(simHist_Gill,0.0))
    diffErr = np.sum((tempProb[:-1] - simHist_Gill)**2.0)
#    print('g = ' + str(round(reactionRates[0],3)) + ', g_pro = ' + str(round(reactionRates[1],3)) + \
#    ', K = ' + str(round(reactionRates[2],3)) + ', r = ' + str(round(reactionRates[3],3)) + \
#    ', Diff = ' + str(round(diffErr,6)))
    return diffErr

def dwellHistHill(conditions,maxInds):
    """ Calculates the distribution of dwell times using FSP with the rates and low/high states provided """
    global maxN
    dwellProbs = []
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probMatrix[probN][probN] -= (conditions['g'] + conditions['g_pro']*(probN**conditions['n'])/(probN**conditions['n'] + conditions['K']) + conditions['r']*probN)
        if probN < maxN - 1:
            probMatrix[probN + 1][probN] += conditions['g'] + conditions['g_pro']*(probN**conditions['n'])/(probN**conditions['n'] + conditions['K'])
        if probN > 0:
            probMatrix[probN - 1][probN] += conditions['r']*probN
    probMatrix = probMatrix[:maxInds[1],:maxInds[1]]
    probMatrix = np.hstack((probMatrix,np.zeros((len(probMatrix),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    probMatrix = scipy.linalg.expm(probMatrix*conditions['inc'])
    probInit = np.zeros(len(probMatrix))
    probInit[maxInds[0]] = 1.0
    dwellCumeProb = [0.0]
    while 1 - dwellCumeProb[-1] > 0.001:
#        print(dwellCumeProb[-1])
        probInit = np.dot(probMatrix,probInit)
        dwellCumeProb.append(probInit[-1])
    dwellProbs.append(np.array(dwellCumeProb[1:]) - np.array(dwellCumeProb[:-1]))
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probMatrix[probN][probN] -= (conditions['g'] + conditions['g_pro']*(probN**conditions['n'])/(probN**conditions['n'] + conditions['K']) + conditions['r']*probN)
        if probN < maxN - 1:
            probMatrix[probN + 1][probN] += conditions['g'] + conditions['g_pro']*(probN**conditions['n'])/(probN**conditions['n'] + conditions['K'])
        if probN > 0:
            probMatrix[probN - 1][probN] += conditions['r']*probN
    probMatrix = probMatrix[maxInds[0] + 1:,maxInds[0] + 1:]
    probMatrix = np.hstack((probMatrix,np.zeros((len(probMatrix),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    probMatrix = scipy.linalg.expm(probMatrix*conditions['inc'])
    probInit = np.zeros(len(probMatrix))
    probInit[maxInds[1] - maxInds[0] - 1] = 1.0
    dwellCumeProb = [0.0]
    while 1 - dwellCumeProb[-1] > 0.001:
#        print(dwellCumeProb[-1])
        probInit = np.dot(probMatrix,probInit)
        dwellCumeProb.append(probInit[-1])
    dwellProbs.append(np.array(dwellCumeProb[1:]) - np.array(dwellCumeProb[:-1]))
    return dwellProbs

""" Initializing self-promo dimer conditions... """
#conditions_Gill = conditionsInitGill(g,g_pro,d,p,r,k_fD,k_bD,k_fP,k_bP,exclusive,\
#n_A_init,n_A2_init,n_a_init,n_alpha_init,inc,numSteps,numTrials)
conditions_Gill = conditionsInitGill(0.05,0.5,0.2,0.02,0.001,0.005,50.0,\
0.006,0.00003,False,5,0,5,0,timeInc,int((24*3600*numDays)/timeInc),numTrials)

""" Loading self-promo dimer Gillespie simulations if they already exist... """
if os.path.exists('GillespieSims_' + simConditions + '/GillespieSim' + str(simNum) + '.npz'):
    tempVars = np.load('GillespieSims_' + simConditions + '/GillespieSim' + str(simNum) + '.npz')
    n_A_Gill = tempVars['n_A_Gill'][:,range(0,24*3600*numDays + 1,timeInc)]
    del tempVars
else:
    """ Creating them if they don't... """
    conditions_Gill['numTrials'] = 5
    n_A_temp,n_A2_temp,n_a_temp,n_alpha_temp = gillespieSim_SelfPromo(conditions_Gill,[],[],[],[])
    equilProb = np.zeros((2,maxn,maxN2,maxN))
    for numTrial in range(len(n_A_temp)):
        print('Trial #' + str(numTrial + 1))
        for numStep in range(len(n_A_temp[numTrial])):
            equilProb[int(n_alpha_temp[numTrial][numStep]),\
                      int(n_a_temp[numTrial][numStep]),\
                      int(n_A2_temp[numTrial][numStep]),\
                      int(n_A_temp[numTrial][numStep])] += 1
        del numStep
    del numTrial
    equilProb = equilProb/np.sum(equilProb)
    del n_A_temp
    del n_A2_temp
    del n_a_temp
    del n_alpha_temp
    conditions_Gill['numTrials'] = numTrials
    """ Using starting conditions at relative equilibrium... """
    n_A_Gill = []
    n_A2_Gill = []
    n_a_Gill = []
    n_alpha_Gill = []
    for numTrial in range(numTrials):
        print('Trial #' + str(numTrial + 1))
        randNum = np.random.uniform(low=0,high=1)
        totProb = 0.0
        for n_alpha_init in range(2):
            for n_a_init in range(maxn):
                for n_A2_init in range(maxN2):
                    for n_A_init in range(maxN):
                        totProb += equilProb[n_alpha_init,n_a_init,n_A2_init,n_A_init]
                        if totProb > randNum:
                            break
                    if totProb > randNum:
                        break
                if totProb > randNum:
                    break
            if totProb > randNum:
                break
        n_A_Gill.append([n_A_init])
        n_A2_Gill.append([n_A2_init])
        n_a_Gill.append([n_a_init])
        n_alpha_Gill.append([n_alpha_init])
        del n_A_init
        del n_A2_init
        del n_a_init
        del n_alpha_init
        del totProb
        del randNum
    del numTrial
    conditions_Gill['inc'] = 1
    conditions_Gill['numSteps'] = int(24*3600*numDays/conditions_Gill['inc'])
    n_A_Gill,n_A2_Gill,n_a_Gill,n_alpha_Gill = gillespieSim_SelfPromo(conditions_Gill,n_A_Gill,n_A2_Gill,n_a_Gill,n_alpha_Gill)
    np.savez_compressed('GillespieSims_' + simConditions + '/GillespieSim' + str(simNum) + '.npz',\
    conditions_Gill=conditions_Gill, n_A_Gill=n_A_Gill, n_A2_Gill=n_A2_Gill, n_a_Gill=n_a_Gill, n_alpha_Gill=n_alpha_Gill)
    n_A_Gill = np.array(n_A_Gill)[:,range(0,24*3600*numDays + 1,timeInc)]
    conditions_Gill['inc'] = timeInc
    conditions_Gill['numSteps'] = int((24*3600*numDays)/timeInc)
    del n_A2_Gill
    del n_a_Gill
    del n_alpha_Gill

""" Calculating the protein number distribution and low/high state locations """
simHist_Gill = np.histogram(n_A_Gill,bins=np.arange(-0.5,maxN))[0]
simHist_Gill = simHist_Gill/sum(simHist_Gill)
maxInds_Gill = peakVals(simHist_Gill[:75],5,0.0005)

""" Calculating relevant entropy statistics and dwell times """
totEntropy_Gill,stateEntropies_Gill,macroEntropy_Gill,dwellVals_Gill = entropyStats(n_A_Gill,maxInds_Gill)
avgDwells_Gill = []
avgTotDwell_Gill = []
for ind in range(len(dwellVals_Gill)):
    avgDwells_Gill.append(np.average(dwellVals_Gill[ind])*conditions_Gill['inc'])
    avgTotDwell_Gill.extend(dwellVals_Gill[ind])
del ind
avgTotDwell_Gill = np.average(avgTotDwell_Gill)*conditions_Gill['inc']
if type(numIterations) == str:
    numIterations = int(round(avgTotDwell_Gill/conditions_Gill['inc']))

""" Counting the transition numbers for maximum likelihood fitting """
probs = scipy.sparse.csc_matrix((np.ones(np.size(n_A_Gill[:,numIterations::numIterations])),\
(n_A_Gill[:,numIterations::numIterations].reshape(np.size(n_A_Gill[:,numIterations::numIterations])),\
n_A_Gill[:,:-numIterations:numIterations].reshape(np.size(n_A_Gill[:,:-numIterations:numIterations])))),shape=(maxN + 1,maxN + 1))

""" Minimizing the negative log of likelihood (and therefore maximizing likelihood) using scipy.minimize """
""" Running multiple fittings using randomized starting points """
n = 2
startGuess = []
finalGuess = []
loglike = []
timeVals = []
numBytes = []
matPop = []
newFitTime = []
for numFit in range(numFits):
    startGuess.append((0.6 + 0.8*np.random.rand())*np.array([5.76077102e-03,1.15840911e-01,1.32892531e+03,1.59452295e-03]))
    stepTime = []
    startTime = datetime.datetime.now()
    res = minimize(hillFSP_mle,startGuess[numFit],method='nelder-mead',tol=0.1,options={'disp':True,'maxiter':500})
    endTime = datetime.datetime.now()
    """ Storing the total fitting time for illustrative purposes """
    timeVals.append((endTime - startTime).total_seconds())    
    newFitTime.append(sum(stepTime))
    finalGuess.append(res['x'])
    loglike.append(res['fun'])
del numFit
bestGuess = finalGuess[np.where(loglike == np.min(loglike))[0][0]]
print('Matrix Exp Time: ')
print( str(np.mean(stepTime)) + ' +/- ' + str(np.std(stepTime)))
print('numBytes: ' + str(np.mean(numBytes)) )

""" Initializing fitted self-promo Hill function conditions... """
conditions_Hill = conditionsInitHill(bestGuess[0],bestGuess[1],n,\
bestGuess[2],bestGuess[3],5,timeInc,int((24*3600*numDays)/timeInc),numTrials)

""" Calculating protein number distribution at relative equilibrium and locating low/high states """
conditions_Hill['numSteps'] = int(conditions_Hill['numSteps']*10)
equilProb_Hill = hillFSP(conditions_Hill)
conditions_Hill['numSteps'] = int(conditions_Hill['numSteps']/10)
maxInds_Hill = [np.where(equilProb_Hill[:25] == np.max(equilProb_Hill[:25]))[0][0],\
np.where(equilProb_Hill[25:75] == np.max(equilProb_Hill[25:75]))[0][0] + 25]

""" Running Hill function Gillespie simulations for confirmation and entropy statistics... """
n_A_Hill = hillSim(conditions_Hill,n_A_Gill[:,0].reshape((len(n_A_Gill),1)).tolist())
simHist_Hill = np.zeros(maxN + 1)
for numTrial in range(len(n_A_Hill)):
    simHist_Hill += np.histogram(n_A_Hill[numTrial],np.arange(maxN + 2))[0]
del numTrial
simHist_Hill = simHist_Hill/sum(simHist_Hill)

""" Calculating dwell times and entropy statistics... """
dwellProbs_HillFSP = dwellHistHill(conditions_Hill,maxInds_Hill)
avgLowDwell_HillFSP = np.sum(np.arange(0,conditions_Hill['inc']*len(dwellProbs_HillFSP[0]),conditions_Hill['inc'])*dwellProbs_HillFSP[0])
avgHighDwell_HillFSP = np.sum(np.arange(0,conditions_Hill['inc']*len(dwellProbs_HillFSP[1]),conditions_Hill['inc'])*dwellProbs_HillFSP[1])
totEntropy_Hill,stateEntropies_Hill,macroEntropy_Hill,dwellVals_Hill = entropyStats(n_A_Hill,maxInds_Hill)
avgLowDwell_HillSim = conditions_Hill['inc']*np.average(dwellVals_Hill[0])
avgHighDwell_HillSim = conditions_Hill['inc']*np.average(dwellVals_Hill[1])

""" Calculating effective production/degradation rates... """
g_Hill = conditions_Hill['g'] + conditions_Hill['g_pro']*(maxInds_Hill[0]**conditions_Hill['n'])/\
(maxInds_Hill[0]**conditions_Hill['n'] + conditions_Hill['K'])
g_star_Hill = conditions_Hill['g'] + conditions_Hill['g_pro']*(maxInds_Hill[1]**conditions_Hill['n'])/\
(maxInds_Hill[1]**conditions_Hill['n'] + conditions_Hill['K'])
r_Hill = conditions_Hill['r']

""" Saving fitting results as an npz file... """
np.savez_compressed('ExtractedParameters,SelfPromoHill,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz',\
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, bestGuess=bestGuess, equilProb_Hill=equilProb_Hill, \
maxInds_Hill=maxInds_Hill, g_Hill=g_Hill, g_star_Hill=g_star_Hill, \
r_Hill=r_Hill, conditions_Hill=conditions_Hill, n_A_Hill=n_A_Hill, \
stateEntropies_Hill=stateEntropies_Hill, macroEntropy_Hill=macroEntropy_Hill, \
totEntropy_Hill=totEntropy_Hill, dwellVals_Hill=dwellVals_Hill, \
avgLowDwell_HillSim=avgLowDwell_HillSim, avgHighDwell_HillSim=avgHighDwell_HillSim, \
timeVals=timeVals, numBytes=numBytes, matPop=matPop)

""" Calculating dwell times from simulations for plotting... """
dwellInds_Gill = [[] for ind in range(len(dwellVals_Gill))]
dwellProbs_Gill = [[] for ind in range(len(dwellVals_Gill))]
for ind in range(len(dwellVals_Gill)):
    dwellProbs_Gill[ind],dwellInds_Gill[ind] = np.histogram(np.array(dwellVals_Gill[ind])*conditions_Gill['inc'],np.arange(0,300000,9000))
    dwellProbs_Gill[ind] = dwellProbs_Gill[ind]/sum(dwellProbs_Gill[ind])
del ind
dwellInds_Hill = [[] for ind in range(len(dwellVals_Hill))]
dwellProbs_Hill = [[] for ind in range(len(dwellVals_Hill))]
for ind in range(len(dwellVals_Hill)):
    dwellProbs_Hill[ind],dwellInds_Hill[ind] = np.histogram(np.array(dwellVals_Hill[ind])*conditions_Hill['inc'],np.arange(0,300000,9000))
    dwellProbs_Hill[ind] = dwellProbs_Hill[ind]/sum(dwellProbs_Hill[ind])
del ind

""" Initializing plotting conditions... """
mpl.rc('font',family='Arial')
mpl.rc('font',size=12)
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
plt.rc('font',weight='bold')

""" Plotting protein number distributions... """
plt.figure()
plt.plot(range(len(simHist_Gill)),simHist_Gill,'.b',markersize=10)
plt.plot(range(len(simHist_Hill)),simHist_Hill,'.r',markersize=10)
plt.axis([0,80,0,0.08])
plt.xticks(np.arange(0,81,20))
plt.yticks(np.arange(0,0.081,0.02))
plt.xlabel('$N_A$',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
plt.grid(True)
plt.legend(['Gillespie','Hill'],fontsize=15)
plt.savefig('ProbHist_HillFitting_' + simConditions + 'Sim' + str(simNum) + '.pdf')

""" Plotting low-to-high dwell time distributions... """
plt.figure()
plt.plot(dwellInds_Gill[0][:-1],dwellProbs_Gill[0],'b')
plt.plot(dwellInds_Hill[0][:-1],dwellProbs_Hill[0],'r')
#plt.axis([0,300000,0,0.006])
#plt.xticks(np.arange(0,300001,100000))
#plt.yticks(np.arange(0,0.0061,0.002))
plt.xlabel('Low-to-High Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
plt.grid(True)
plt.legend(['Gillespie','Hill'],fontsize=15)
plt.savefig('LowDwellHist_HillFitting_' + simConditions + 'Sim' + str(simNum) + '.pdf')

""" Plotting high-to-low dwell time distributions... """
plt.figure()
plt.plot(dwellInds_Gill[1][:-1],dwellProbs_Gill[1],'b')
plt.plot(dwellInds_Hill[1][:-1],dwellProbs_Hill[1],'r')
#plt.axis([0,300000,0,0.006])
#plt.xticks(np.arange(0,300001,100000))
#plt.yticks(np.arange(0,0.0061,0.002))
plt.xlabel('High-to-Low Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
plt.grid(True)
plt.legend(['Gillespie','Hill'],fontsize=15)
plt.savefig('HighDwellHist_HillFitting_' + simConditions + 'Sim' + str(simNum) + '.pdf')








