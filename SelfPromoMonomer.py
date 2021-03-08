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

def conditionsInitMono(g,g_pro,r,k_fP,k_bP,n_A_init,n_alpha_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all relevant rates and conditions for the self-promotion (monomer) circuit """
    return {'g':g, 'g_pro':g_pro, 'r':r, 'k_fP':k_fP, 'k_bP':k_bP, \
    'N_A_init':n_A_init, 'N_alpha_init':n_alpha_init, \
    'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def gillespieSim_Monomer(conditions,n_A=[],n_alpha=[]):
    """ Simulates self-promotion (monomer) gene expression trajectories given the provided conditions """
    """ Can continue simulations if provided in n_A, n_A2, n_a, and n_alpha """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_alpha = (conditions['N_alpha_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        if len(n_A) < numTrial + 1:
            numA = np.copy(conditions['N_A_init'])
            numalpha = np.copy(conditions['N_alpha_init'])
            n_A.append([np.copy(numA)])
            n_alpha.append([np.copy(numalpha)])
        else:
            numA = np.copy(n_A[numTrial][-1])
            numalpha = np.copy(n_alpha[numTrial][-1])
        timeFrame = (len(n_A[numTrial]) - 1)*conditions['inc']
        incCheckpoint = len(n_A[numTrial])*conditions['inc']
        print('Trial #' + str(numTrial + 1))
        while timeFrame < float(conditions['numSteps']*conditions['inc']):
            prob = [conditions['g']*(1 - numalpha),\
            conditions['g_pro']*numalpha,\
            conditions['r']*numA,\
            conditions['k_fP']*(1 - numalpha)*numA,\
            conditions['k_bP']*numalpha]
            overallRate = sum(prob)
            randNum1 = np.random.rand(1)
            timeFrame -= np.log(randNum1)/overallRate
            while timeFrame >= incCheckpoint:
                n_A[numTrial].append(np.copy(numA).tolist())
                n_alpha[numTrial].append(np.copy(numalpha).tolist())
                incCheckpoint += conditions['inc']
            prob = prob/overallRate
            randNum2 = np.random.rand(1)
            if randNum2 <= sum(prob[:2]):
                numA += 1
            elif randNum2 <= sum(prob[:3]):
                numA -= 1
            elif randNum2 <= sum(prob[:4]):
                numalpha += 1
                numA -= 1
            else:
                numalpha -= 1
                numA += 1
        n_A[numTrial] = n_A[numTrial][:conditions['numSteps'] + 1]
        n_alpha[numTrial] = n_alpha[numTrial][:conditions['numSteps'] + 1]
    return n_A, n_alpha

def monomerFSP(conditions):
    """ Runs a Finite State Projection (FSP) of the self-promotion monomer circuit using the provided conditions """
    global maxN
    global numBytes
    probMatrix = np.zeros((2*(maxN + 1),2*(maxN + 1)))
    for probN in range(maxN + 1):
        for probN_alpha in range(2):
            probMatrix[probN_alpha*(maxN + 1) + probN][probN_alpha*(maxN + 1) + probN] -= \
            (conditions['g']*(1 - probN_alpha) + conditions['g_pro']*probN_alpha + conditions['r']*probN + \
            conditions['k_fP']*(1 - probN_alpha)*probN + conditions['k_bP']*probN_alpha)
            if probN <= maxN - 1:
                probMatrix[probN_alpha*(maxN + 1) + probN + 1][probN_alpha*(maxN + 1) + probN] += \
                conditions['g']*(1 - probN_alpha) + conditions['g_pro']*probN_alpha
            if probN > 0:
                probMatrix[probN_alpha*(maxN + 1) + probN - 1][probN_alpha*(maxN + 1) + probN] += \
                conditions['r']*probN
            if probN_alpha == 0 and probN > 0:
                probMatrix[(maxN + 1) + probN - 1][probN] += \
                conditions['k_fP']*(1 - probN_alpha)*probN
            if probN_alpha == 1 and probN <= maxN - 1:
                probMatrix[probN + 1][(maxN + 1) + probN] += \
                conditions['k_bP']*probN_alpha
    probMatrix = np.hstack((probMatrix,np.zeros((2*(maxN + 1),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    print(probMatrix.shape)
    print(probMatrix.dtype)
    print('# of bytes: ' + str(probMatrix.nbytes))
    numBytes.append(probMatrix.nbytes)
    probInit = np.zeros(len(probMatrix))
    probInit[conditions['N_alpha_init']*(maxN + 1) + conditions['N_A_init']] = 1.0
    
    """ Calculating matrix exponential time and non-sink state probability for illustrative purposes """
    timeVals = []
    for numTry in range(100):
        startTime = datetime.datetime.now()
        tempProb = np.dot(scipy.linalg.expm(probMatrix*conditions['inc']*conditions['numSteps']),probInit)
        endTime = datetime.datetime.now()
        timeVals.append((endTime - startTime).total_seconds())
    print('Exponential Time: ' + str(round(np.average(timeVals),5)) + ' +/- ' + str(round(np.std(timeVals),5)) + ' seconds')
    finalProb = np.zeros((2,maxN + 1))
    for probN_alpha in range(2):
        for probN in range(maxN + 1):
            finalProb[probN_alpha][probN] = tempProb[probN_alpha*(maxN + 1) + probN]
    print('Probability: ' + str(np.sum(finalProb)))
    """ Calculating matrix exponential time and non-sink state probability for illustrative purposes """

    return finalProb

def monomerFSP_mle(reactionRates):
    """ Calculates the likelihood of the transitions provided in the global variable "probs" given the rates provided """
    """ Strange function formatting is necessary to interface with scipy.minimize """
    global probs
    global maxN
    global numIterations
    global timeInc
    global stepTime
    global matPop    
    if np.any(np.array(reactionRates) < 0):
        return float('Inf')
    startTime = datetime.datetime.now()
    probMatrix = np.zeros((2*(maxN + 1),2*(maxN + 1)))
    for probN in range(maxN + 1):
        for probN_alpha in range(2):
            probMatrix[probN_alpha*(maxN + 1) + probN][probN_alpha*(maxN + 1) + probN] -= \
            (reactionRates[0]*(1 - probN_alpha) + reactionRates[1]*probN_alpha + reactionRates[2]*probN + \
            reactionRates[3]*(1 - probN_alpha)*probN + reactionRates[4]*probN_alpha)
            if probN <= maxN - 1:
                probMatrix[probN_alpha*(maxN + 1) + probN + 1][probN_alpha*(maxN + 1) + probN] += \
                reactionRates[0]*(1 - probN_alpha) + reactionRates[1]*probN_alpha
            if probN > 0:
                probMatrix[probN_alpha*(maxN + 1) + probN - 1][probN_alpha*(maxN + 1) + probN] += \
                reactionRates[2]*probN
            if probN_alpha == 0 and probN > 0:
                probMatrix[(maxN + 1) + probN - 1][probN] += \
                reactionRates[3]*(1 - probN_alpha)*probN
            if probN_alpha == 1 and probN <= maxN - 1:
                probMatrix[probN + 1][(maxN + 1) + probN] += \
                reactionRates[4]*probN_alpha
    probMatrix = np.hstack((probMatrix,np.zeros((2*(maxN + 1),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    
    equilProb = np.dot(scipy.linalg.expm(probMatrix*1000000),np.append(np.append(simHist_Gill/2,0.0),np.append(simHist_Gill/2,[0.0,0.0])))
    endTime = datetime.datetime.now()
    matPop.append((endTime - startTime).total_seconds())
    """ Measuring matrix population and exponentiation for illustrative purposes """
    startTime = datetime.datetime.now()
    tempMatrix = scipy.linalg.expm(probMatrix*numIterations*timeInc)
    endTime = datetime.datetime.now()
    stepTime.append((endTime - startTime).total_seconds())
    
    tempMatrix = tempMatrix[:maxN + 1,:] + tempMatrix[maxN + 1:-1,:]
    for startN_A in range(maxN + 1):
        tempMatrix[:,startN_A] = (equilProb[startN_A]/(equilProb[startN_A] + equilProb[maxN + 1 + startN_A]))*tempMatrix[:,startN_A] + \
        (equilProb[maxN + 1 + startN_A]/(equilProb[startN_A] + equilProb[maxN + 1 + startN_A]))*tempMatrix[:,maxN + 1 + startN_A]
    tempMatrix = tempMatrix[:,:maxN + 1]
    
    loglike = -1*np.nansum(np.log(tempMatrix)*probs.toarray())
#    print('g = ' + str(round(reactionRates[0],5)) + ', g_pro = ' + str(round(reactionRates[1],5)) + \
#    ', r = ' + str(round(reactionRates[2],5)) + ', k_fP = ' + str(round(reactionRates[3],5)) + \
#    ', k_bP = ' + str(round(reactionRates[4],5)) + ', loglike = ' + str(round(loglike,1)))
    return loglike

def monomerEquil_mse(reactionRates):
    """ Calculating the mean squared error of protein number distributions given the rates provided """
    """ Method not provided in papers, just used to find a starting point for rate estimation """
    global simHist_Gill
    if np.any(np.array(reactionRates) < 0):
        return float('Inf')
    probMatrix = np.zeros((2*(maxN + 1),2*(maxN + 1)))
    for probN in range(maxN + 1):
        for probN_alpha in range(2):
            probMatrix[probN_alpha*(maxN + 1) + probN][probN_alpha*(maxN + 1) + probN] -= \
            (reactionRates[0]*(1 - probN_alpha) + reactionRates[1]*probN_alpha + reactionRates[2]*probN + \
            reactionRates[3]*(1 - probN_alpha)*probN + reactionRates[4]*probN_alpha)
            if probN <= maxN - 1:
                probMatrix[probN_alpha*(maxN + 1) + probN + 1][probN_alpha*(maxN + 1) + probN] += \
                reactionRates[0]*(1 - probN_alpha) + reactionRates[1]*probN_alpha
            if probN > 0:
                probMatrix[probN_alpha*(maxN + 1) + probN - 1][probN_alpha*(maxN + 1) + probN] += \
                reactionRates[2]*probN
            if probN_alpha == 0 and probN > 0:
                probMatrix[(maxN + 1) + probN - 1][probN] += \
                reactionRates[3]*(1 - probN_alpha)*probN
            if probN_alpha == 1 and probN <= maxN - 1:
                probMatrix[probN + 1][(maxN + 1) + probN] += \
                reactionRates[4]*probN_alpha
    probMatrix = np.hstack((probMatrix,np.zeros((2*(maxN + 1),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    tempProb = np.dot(scipy.linalg.expm(probMatrix*1000000),np.append(np.append(simHist_Gill/2,0.0),np.append(simHist_Gill/2,[0.0,0.0])))
    tempProb = tempProb[:maxN + 1] + tempProb[maxN + 1:-1]
    diffErr = np.sum((tempProb[:-1] - simHist_Gill)**2.0)
#    print('g = ' + str(round(reactionRates[0],5)) + ', g_pro = ' + str(round(reactionRates[1],5)) + \
#    ', r = ' + str(round(reactionRates[2],5)) + ', k_fP = ' + str(round(reactionRates[3],5)) + \
#    ', k_bP = ' + str(round(reactionRates[4],5)) + ', Diff = ' + str(round(diffErr,6)))
    return diffErr

def dwellHistMonomer(conditions,maxInds):
    """ Calculates the distribution of dwell times using FSP with the rates and low/high states provided """
    global maxN
    dwellProbs = []
    probMatrix = np.zeros((2*(maxN + 1),2*(maxN + 1)))
    for probN in range(maxN + 1):
        for probN_alpha in range(2):
            probMatrix[probN_alpha*(maxN + 1) + probN][probN_alpha*(maxN + 1) + probN] -= \
            (conditions['g']*(1 - probN_alpha) + conditions['g_pro']*probN_alpha + conditions['r']*probN + \
            conditions['k_fP']*(1 - probN_alpha)*probN + conditions['k_bP']*probN_alpha)
            if probN <= maxN - 1:
                probMatrix[probN_alpha*(maxN + 1) + probN + 1][probN_alpha*(maxN + 1) + probN] += \
                conditions['g']*(1 - probN_alpha) + conditions['g_pro']*probN_alpha
            if probN > 0:
                probMatrix[probN_alpha*(maxN + 1) + probN - 1][probN_alpha*(maxN + 1) + probN] += \
                conditions['r']*probN
            if probN_alpha == 0 and probN > 0:
                probMatrix[(maxN + 1) + probN - 1][probN] += \
                conditions['k_fP']*(1 - probN_alpha)*probN
            if probN_alpha == 1 and probN <= maxN - 1:
                probMatrix[probN + 1][(maxN + 1) + probN] += \
                conditions['k_bP']*probN_alpha
    probMatrix = probMatrix[:maxInds[1],:maxInds[1]]
    probMatrix = np.hstack((probMatrix,np.zeros((len(probMatrix),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    probMatrix = scipy.linalg.expm(probMatrix*conditions['inc'])
    probInit = np.zeros(len(probMatrix))
    probInit[maxInds[0]] = 1.0
    dwellCumeProb = [0.0]
    while 1 - dwellCumeProb[-1] > 0.001:
        probInit = np.dot(probMatrix,probInit)
        dwellCumeProb.append(probInit[-1])
    dwellProbs.append(np.array(dwellCumeProb[1:]) - np.array(dwellCumeProb[:-1]))
    probMatrix = np.zeros((2*(maxN + 1),2*(maxN + 1)))
    for probN in range(maxN + 1):
        for probN_alpha in range(2):
            probMatrix[probN_alpha*(maxN + 1) + probN][probN_alpha*(maxN + 1) + probN] -= \
            (conditions['g']*(1 - probN_alpha) + conditions['g_pro']*probN_alpha + conditions['r']*probN + \
            conditions['k_fP']*(1 - probN_alpha)*probN + conditions['k_bP']*probN_alpha)
            if probN <= maxN - 1:
                probMatrix[probN_alpha*(maxN + 1) + probN + 1][probN_alpha*(maxN + 1) + probN] += \
                conditions['g']*(1 - probN_alpha) + conditions['g_pro']*probN_alpha
            if probN > 0:
                probMatrix[probN_alpha*(maxN + 1) + probN - 1][probN_alpha*(maxN + 1) + probN] += \
                conditions['r']*probN
            if probN_alpha == 0 and probN > 0:
                probMatrix[(maxN + 1) + probN - 1][probN] += \
                conditions['k_fP']*(1 - probN_alpha)*probN
            if probN_alpha == 1 and probN <= maxN - 1:
                probMatrix[probN + 1][(maxN + 1) + probN] += \
                conditions['k_bP']*probN_alpha
    probMatrix = probMatrix[maxInds[0] + 1:,maxInds[0] + 1:]
    probMatrix = np.hstack((probMatrix,np.zeros((len(probMatrix),1))))
    probMatrix = np.vstack((probMatrix,-1*np.sum(probMatrix,axis=0)))
    probMatrix = scipy.linalg.expm(probMatrix*conditions['inc'])
    probInit = np.zeros(len(probMatrix))
    probInit[maxInds[1] - maxInds[0] - 1] = 1.0
    dwellCumeProb = [0.0]
    while 1 - dwellCumeProb[-1] > 0.001:
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
            equilProb[int(n_alpha_temp[numTrial][numStep]),int(n_a_temp[numTrial][numStep]),int(n_A2_temp[numTrial][numStep]),int(n_A_temp[numTrial][numStep])] += 1
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
    n_A_Gill = []
    n_A2_Gill = []
    n_a_Gill = []
    n_alpha_Gill = []
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
startGuess = []
finalGuess = []
loglike = []
timeVals = []
numBytes = []
matPop = []
newFitTime = []
for numFit in range(numFits):
    startGuess.append((0.6 + 0.8*np.random.rand())*np.array([0.005,0.05,0.001,0.00001,0.00005]))
    stepTime = []
    startTime = datetime.datetime.now()
    res = minimize(monomerFSP_mle,startGuess[numFit],method='nelder-mead',tol=0.1,options={'disp':True,'maxiter':500})
    endTime = datetime.datetime.now()
    newFitTime.append(sum(stepTime))
    finalGuess.append(res['x'])
    loglike.append(res['fun'])
    """ Storing the total fitting time for illustrative purposes """
    timeVals.append((endTime - startTime).total_seconds())
del numFit
bestGuess = finalGuess[np.where(loglike == np.min(loglike))[0][0]]
print('Matrix Exp Time: ')
print( str(np.mean(stepTime)) + ' +/- ' + str(np.std(stepTime)))

""" Initializing fitted self-promo monomer conditions... """
#conditionsInitMono(g,g_pro,r,k_fP,k_bP,n_A_init,n_alpha_init,inc,numSteps,numTrials)
conditions_Mono = conditionsInitMono(bestGuess[0],bestGuess[1],bestGuess[2],\
bestGuess[3],bestGuess[4],5,0,timeInc,int((24*3600*numDays)/timeInc),numTrials)

""" Calculating protein number distribution at relative equilibrium and locating low/high states """
conditions_Mono['numSteps'] = int(conditions_Mono['numSteps']*10)
equilProb_Mono = monomerFSP(conditions_Mono)
conditions_Mono['numSteps'] = int(conditions_Mono['numSteps']/10)
maxInds_Mono = [np.where(np.sum(equilProb_Mono,axis=0)[:25] == np.max(np.sum(equilProb_Mono,axis=0)[:25]))[0][0],\
np.where(np.sum(equilProb_Mono,axis=0)[25:75] == np.max(np.sum(equilProb_Mono,axis=0)[25:75]))[0][0] + 25]

""" Running monomer Gillespie simulations for confirmation and entropy statistics... """
n_a_Mono = np.zeros((len(n_A_Gill),1))
n_a_Mono[n_A_Gill[:,0] >= 25] = 1
n_a_Mono = n_a_Mono.tolist()
n_A_Mono,n_a_Mono = gillespieSim_Monomer(conditions_Mono,n_A_Gill[:,0].reshape((len(n_A_Gill),1)).tolist(),n_a_Mono)
simHist_Mono = np.zeros(maxN + 1)
for numTrial in range(len(n_A_Mono)):
    simHist_Mono += np.histogram(n_A_Mono[numTrial],np.arange(maxN + 2))[0]
del numTrial
simHist_Mono = simHist_Mono/np.sum(simHist_Mono)

""" Calculating dwell times and entropy statistics... """
dwellProbs_MonoFSP = dwellHistMonomer(conditions_Mono,maxInds_Mono)
avgLowDwell_MonoFSP = np.sum(np.arange(0,conditions_Mono['inc']*len(dwellProbs_MonoFSP[0]),conditions_Mono['inc'])*dwellProbs_MonoFSP[0])
avgHighDwell_MonoFSP = np.sum(np.arange(0,conditions_Mono['inc']*len(dwellProbs_MonoFSP[1]),conditions_Mono['inc'])*dwellProbs_MonoFSP[1])
totEntropy_Mono,stateEntropies_Mono,macroEntropy_Mono,dwellVals_Mono = entropyStats(n_A_Mono,maxInds_Mono)
avgLowDwell_MonoSim = conditions_Mono['inc']*np.average(dwellVals_Mono[0])
avgHighDwell_MonoSim = conditions_Mono['inc']*np.average(dwellVals_Mono[1])

""" Saving fitting results as an npz file... """
np.savez_compressed('ExtractedParameters,SelfPromoMonomer,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz',\
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, bestGuess=bestGuess, equilProb_Mono=equilProb_Mono, \
maxInds_Mono=maxInds_Mono, conditions_Mono=conditions_Mono, n_A_Mono=n_A_Mono, \
stateEntropies_Mono=stateEntropies_Mono, macroEntropy_Mono=macroEntropy_Mono, \
totEntropy_Mono=totEntropy_Mono, dwellVals_Mono=dwellVals_Mono, \
avgLowDwell_MonoSim=avgLowDwell_MonoSim, avgHighDwell_MonoSim=avgHighDwell_MonoSim, \
timeVals=timeVals, numBytes=numBytes, matPop=matPop, stepTime=stepTime)

""" Calculating dwell times from simulations for plotting... """
dwellInds_Gill = [[] for ind in range(len(dwellVals_Gill))]
dwellProbs_Gill = [[] for ind in range(len(dwellVals_Gill))]
for ind in range(len(dwellVals_Gill)):
    dwellProbs_Gill[ind],dwellInds_Gill[ind] = np.histogram(np.array(dwellVals_Gill[ind])*conditions_Gill['inc'],np.arange(0,300000,9000))
    dwellProbs_Gill[ind] = dwellProbs_Gill[ind]/sum(dwellProbs_Gill[ind])
del ind
dwellInds_Mono = [[] for ind in range(len(dwellVals_Mono))]
dwellProbs_Mono = [[] for ind in range(len(dwellVals_Mono))]
for ind in range(len(dwellVals_Mono)):
    dwellProbs_Mono[ind],dwellInds_Mono[ind] = np.histogram(np.array(dwellVals_Mono[ind])*conditions_Mono['inc'],np.arange(0,300000,9000))
    dwellProbs_Mono[ind] = dwellProbs_Mono[ind]/sum(dwellProbs_Mono[ind])
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
plt.plot(range(len(simHist_Mono)),simHist_Mono,'.r',markersize=10)
plt.plot(range(len(np.sum(equilProb_Mono,axis=0))),np.sum(equilProb_Mono,axis=0),'.k',markersize=10)
plt.axis([0,80,0,0.08])
plt.xticks(np.arange(0,81,20))
plt.yticks(np.arange(0,0.081,0.02))
plt.xlabel('$N_A$',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
plt.grid(True)
plt.legend(['Gillespie','Monomer'],fontsize=15)
plt.savefig('ProbHist_MonomerFitting_' + simConditions + 'Sim' + str(simNum) + '.pdf')

""" Plotting low-to-high dwell time distributions... """
plt.figure()
plt.plot(dwellInds_Gill[0][:-1],dwellProbs_Gill[0],'b')
plt.plot(dwellInds_Mono[0][:-1],dwellProbs_Mono[0],'r')
#plt.axis([0,300000,0,0.006])
#plt.xticks(np.arange(0,300001,100000))
#plt.yticks(np.arange(0,0.0061,0.002))
plt.xlabel('Low-to-High Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
plt.grid(True)
plt.legend(['Gillespie','Monomer'],fontsize=15)
plt.savefig('LowDwellHist_MonomerFitting_' + simConditions + 'Sim' + str(simNum) + '.pdf')

""" Plotting high-to-low dwell time distributions... """
plt.figure()
plt.plot(dwellInds_Gill[1][:-1],dwellProbs_Gill[1],'b')
plt.plot(dwellInds_Mono[1][:-1],dwellProbs_Mono[1],'r')
#plt.axis([0,300000,0,0.006])
#plt.xticks(np.arange(0,300001,100000))
#plt.yticks(np.arange(0,0.0061,0.002))
plt.xlabel('High-to-Low Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
plt.grid(True)
plt.legend(['Gillespie','Monomer'],fontsize=15)
plt.savefig('HighDwellHist_MonomerFitting_' + simConditions + 'Sim' + str(simNum) + '.pdf')









