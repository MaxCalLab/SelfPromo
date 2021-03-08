# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:47:25 2017

@author: tefirman
"""

import numpy as np
from scipy.optimize import minimize
import scipy.sparse
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

def conditionsInitMaxCal(h_a,h_A,k_A,maxl_a,\
n_A_init,l_a_init,l_iA_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing all parameters and conditions for the MaxCal framework """
    return {'h_a':float(h_a), 'h_A':float(h_A), 'k_A':float(k_A), \
    'maxl_a':maxl_a, 'N_A_init':n_A_init, 'l_a_init':l_a_init, \
    'l_iA_init':l_iA_init, 'inc':inc, 'numSteps':numSteps, \
    'numTrials':numTrials}

def logFactorial(value):
    """ Calculates the log factorial of the value provided using Stirling's Approximation """
    if all([value > 0,abs(round(value) - value) < 0.000001,value <= 34]):
        return float(sum(np.log(range(1,int(value) + 1))))
    elif all([value > 0,abs(round(value) - value) < 0.000001,value > 34]):
        return float(value)*np.log(float(value)) - float(value) + \
        0.5*np.log(2.0*np.pi*float(value)) - 1.0/(12.0*float(value))
    elif value == 0:
        return float(0)
    else:
        return float('nan')

""" Storing log factorials for future reference so they won't need to be generated every time """
factMat = []
for ind1 in range(maxN + 1):
    factMat.append([])
    for ind2 in range(ind1 + 1):
        factMat[ind1].append(logFactorial(ind1) - logFactorial(ind2) - logFactorial(ind1 - ind2))
    del ind2
    factMat[ind1] = np.array(factMat[ind1])
del ind1

def transitionProbs(lagrangeVals,maxl_a,n_A):
    """ Calculating the probability of transitioning from one protein level to another given the provided parameters """
    global factMat
    logWeight = -float('Inf')*np.ones((n_A + 1,maxl_a + 1))
    templ_a = np.arange(maxl_a + 1)
    for templ_A in range(n_A + 1):
        logWeight[templ_A][templ_a] = factMat[n_A][templ_A] + lagrangeVals[0]*templ_a + \
        lagrangeVals[1]*templ_A + lagrangeVals[2]*templ_a*templ_A
    logWeight = np.exp(logWeight - np.max(np.max(logWeight)))
    logWeight = logWeight/sum(sum(logWeight))
    return logWeight

def maxCalSim(conditions,n_A=[],l_a=[],l_iA=[]):
    """ Simulates MaxCal gene expression trajectories given the provided conditions """
    """ Can continue simulations if provided in n_A """
    """ Needs to run serially, running in parallel produces identical traces, problem with random seed... """
    global maxN
    global factMat
    probsTot = [[] for ind in range(maxN)]
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_a = (conditions['l_a_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_iA = (conditions['l_iA_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        print('Trial #' + str(numTrial + 1))
        tempData = open('ProgressReport_SelfPromoDimerMaxCal_' + simConditions + 'Sim_' + str(simNum) + '.txt','a')
        tempData.write('Trial #' + str(numTrial + 1) + '\n')
        tempData.close()
        if len(n_A) < numTrial + 1:
            n_A.append(np.copy(conditions['N_A_init'][0]).tolist())
            l_a.append(np.copy(conditions['l_a_init'][0]).tolist())
            l_iA.append(np.copy(conditions['l_iA_init'][0]).tolist())
        for numStep in range(len(n_A[numTrial]),len(n_A[numTrial]) + conditions['numSteps']):
            randNum = np.random.rand(1)
            if len(probsTot[n_A[numTrial][numStep - 1]]) == 0:
                probsTot[n_A[numTrial][numStep - 1]] = transitionProbs([conditions['h_a'],\
                conditions['h_A'],conditions['k_A']],conditions['maxl_a'],n_A[numTrial][numStep - 1])
            probSum = 0
            l_aVal = -1
            while l_aVal < conditions['maxl_a'] and randNum > probSum:
                l_aVal += 1
                l_iAVal = n_A[numTrial][numStep - 1] + 1
                while l_iAVal > 0 and randNum > probSum:
                    l_iAVal -= 1
                    probSum += probsTot[n_A[numTrial][numStep - 1]][l_iAVal,l_aVal]
            n_A[numTrial].append(l_iAVal + l_aVal)
            l_a[numTrial].append(l_aVal)
            l_iA[numTrial].append(l_iAVal)
    return n_A, l_a, l_iA

def maxCalFSP(lagrangeVals,maxl_a,n_A_init,numSteps):
    """ Runs a Finite State Projection (FSP) of the MaxCal framework using the provided conditions """
    global maxN
    global numBytes
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probsMC = transitionProbs(lagrangeVals,maxl_a,probN).reshape((probN + 1)*(maxl_a + 1))
        finalN = np.array([np.arange(ind,ind + maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        probsMC = probsMC[finalN < maxN]
        finalN = finalN[finalN < maxN]
        for ind in range(len(finalN)):
            probMatrix[finalN[ind],probN] += probsMC[ind]
    probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
    probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
    print(probMatrix.shape)
    print(probMatrix.dtype)
    print('# of bytes: ' + str(probMatrix.nbytes))
    probInit = np.zeros(len(probMatrix))
    probInit[n_A_init] = 1.0
    numBytes.append(probMatrix.nbytes)
    
    """ Calculating matrix exponential time and non-sink state probability for illustrative purposes """
    timeVals = []
    for numTry in range(100):
        startTime = datetime.datetime.now()
        probFinal = np.dot(np.linalg.matrix_power(probMatrix,numSteps),probInit)
        endTime = datetime.datetime.now()
        timeVals.append((endTime - startTime).total_seconds())
    print('Matrix Power Time: ' + str(round(np.average(timeVals),6)) + ' +/- ' + str(round(np.std(timeVals),6)) + ' seconds')
    print('Probability: ' + str(np.sum(probFinal[:-1])))
    """ Calculating matrix exponential time and non-sink state probability for illustrative purposes """

    return probFinal

def maxCalEquil(lagrangeVals,maxl_a):
    """ Calculates the equilibrium protein number distribution via the eigenvectors of the transition matrix """
    global maxN
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probsMC = transitionProbs(lagrangeVals,maxl_a,probN).reshape((probN + 1)*(maxl_a + 1))
        finalN = np.array([np.arange(ind,ind + maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        probsMC = probsMC[finalN < maxN]
        finalN = finalN[finalN < maxN]
        for ind in range(len(finalN)):
            probMatrix[finalN[ind],probN] += probsMC[ind]
    probMatrix -= np.identity(probMatrix.shape[0])
    vals,equilProb = np.linalg.eig(probMatrix)
    equilProb = equilProb[:,np.imag(vals) == 0]
    vals = abs(np.real(vals[np.imag(vals) == 0]))
    equilProb = abs(np.real(equilProb[:,np.where(vals == min(vals))[0][0]]))
    return equilProb/sum(equilProb)

def dwellHistMaxCal(conditions,maxInds,maxTol,inc):
    """ Calculates the distribution of MaxCal dwell times with the parameters and low/high states provided """
    global maxN
    dwellProbs = []
    for startInd in range(len(maxInds)):
        probMatrix = np.zeros((maxN,maxN))
        for probN in range(maxN):
            if probN in maxInds and probN != maxInds[startInd]:
                continue
            probsMC = transitionProbs([conditions['h_a'],conditions['h_A'],\
            conditions['k_A']],conditions['maxl_a'],probN).reshape((probN + 1)*(conditions['maxl_a'] + 1))
            finalN = np.array([np.arange(ind,ind + conditions['maxl_a'] + 1) \
            for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(conditions['maxl_a'] + 1))
            probsMC = probsMC[finalN < maxN]
            finalN = finalN[finalN < maxN]
            for ind in range(len(finalN)):
                probMatrix[finalN[ind],probN] += probsMC[ind]
        probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
        probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
        probMatrix = np.linalg.matrix_power(probMatrix,inc)
        probInit = np.zeros(len(probMatrix))
        probInit[maxInds[startInd]] = 1.0
        dwellCumeProb = [0.0]
        while 1 - dwellCumeProb[-1] > maxTol:
            probInit = np.dot(probMatrix,probInit)
            dwellCumeProb.append(sum(probInit[maxInds]) + probInit[-1] - probInit[maxInds[startInd]])
            print(dwellCumeProb[-1])
        dwellProbs.append(np.array(dwellCumeProb[1:]) - np.array(dwellCumeProb[:-1]))
    return dwellProbs

def rateCalc(lagrangeVals,maxl_a,n_A):
    """ Calculating the effective production/degradation rates from l_alpha and l_A """
    """ Specific to protein level n_A provided """
    l_A = np.array([indA*np.ones(maxl_a + 1) for indA in range(n_A + 1)])
    l_a = np.array([np.arange(maxl_a + 1) for indA in range(n_A + 1)])
    probVals = transitionProbs(lagrangeVals,maxl_a,n_A)
    prodRateA = np.sum(l_a*probVals)
    if n_A > 0:
        degRateA = np.sum((n_A - l_A)*probVals)
        degRateA /= n_A
    else:
        degRateA = float('NaN')
    return prodRateA, degRateA

def feedbackCalc(lagrangeVals,maxl_a,equilProb):
    """ Calculating the effective feedback metric for the parameters provided """
    if len(equilProb) == 0:
        equilProb = maxCalEquil(lagrangeVals,maxl_a)
    avgl_a = 0.0
    avgl_A = 0.0
    avgl_a2 = 0.0
    avgl_A2 = 0.0
    avgl_al_A = 0.0
    for probN in range(1,len(equilProb)):
        probsMC = transitionProbs(lagrangeVals,maxl_a,probN).reshape((probN + 1)*(maxl_a + 1))
        l_aVals = np.array([np.arange(maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        l_AVals = np.array([ind*np.ones(maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        avgl_a += equilProb[probN]*np.sum(probsMC*l_aVals)
        avgl_A += equilProb[probN]*np.sum(probsMC*l_AVals)
        avgl_a2 += equilProb[probN]*np.sum(probsMC*l_aVals*l_aVals)
        avgl_A2 += equilProb[probN]*np.sum(probsMC*l_AVals*l_AVals)
        avgl_al_A += equilProb[probN]*np.sum(probsMC*l_aVals*l_AVals)
    stDevl_a = (avgl_a2 - avgl_a**2.0)**0.5
    stDevl_A = (avgl_A2 - avgl_A**2.0)**0.5
    feedback = (avgl_al_A - avgl_a*avgl_A)/(stDevl_a*stDevl_A)
    return feedback

def maxCal_mle(lagrangeVals):
    """ Calculates the likelihood of the transitions provided in the global variable "probs" given the parameters provided """
    """ Strange function formatting is necessary to interface with scipy.minimize """
    global probs
    global maxl_a
    global maxN
    global numIterations
    global numBytes
    global stepTime
    global matPop
    startTime = datetime.datetime.now()
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probsMC = transitionProbs(lagrangeVals,maxl_a,probN).reshape((probN + 1)*(maxl_a + 1))
        finalN = np.array([np.arange(ind,ind + maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        probsMC = probsMC[finalN < maxN]
        finalN = finalN[finalN < maxN]
        for ind in range(len(finalN)):
            probMatrix[finalN[ind],probN] += probsMC[ind]
    probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
    probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
    endTime = datetime.datetime.now()
    matPop.append((endTime - startTime).total_seconds())
    """ Measuring matrix population and exponentiation for illustrative purposes """
    startTime = datetime.datetime.now()
    tempMatrix = np.linalg.matrix_power(probMatrix,numIterations)
    endTime = datetime.datetime.now()
    stepTime.append((endTime - startTime).total_seconds())
    numBytes.append(tempMatrix.nbytes)
    loglike = -1*np.nansum(np.nansum(np.log(tempMatrix)*probs.toarray()))
#    print('h_a = ' + str(round(lagrangeVals[0],3)) + ', h_A = ' + \
#    str(round(lagrangeVals[1],3)) + ', K_A = ' + str(round(lagrangeVals[2],4)) + \
#    ', loglike = ' + str(round(loglike,1)))
    return loglike

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
    conditions_Gill['numTrials'] = 2
    n_A_temp,n_A2_temp,n_a_temp,n_alpha_temp = gillespieSim_SelfPromo(conditions_Gill,[],[],[],[])
    equilProb = np.zeros((2,maxn,maxN2,maxN))
    for numTrial in range(len(n_A_temp)):
        print('Trial #' + str(numTrial + 1))
        for numStep in range(len(n_A_temp[numTrial])):
            equilProb[int(n_alpha_temp[numTrial][numStep]),\
                      int(n_a_temp[numTrial][numStep]),\
                      int(n_A2_temp[numTrial][numStep]),\
                      int(n_A_temp[numTrial][numStep])] += 1
    equilProb = equilProb/np.sum(equilProb)
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
    minMaxl_a = 10
else:
    print(np.ceil(np.max(n_A_Gill[:,numIterations::numIterations] - n_A_Gill[:,:-numIterations:numIterations])/numIterations))
    minMaxl_a = int(np.ceil(np.max(n_A_Gill[:,numIterations::numIterations] - n_A_Gill[:,:-numIterations:numIterations])/numIterations))

""" Counting the transition numbers for maximum likelihood fitting """
probs = scipy.sparse.csc_matrix((np.ones(np.size(n_A_Gill[:,numIterations::numIterations])),\
(n_A_Gill[:,numIterations::numIterations].reshape(np.size(n_A_Gill[:,numIterations::numIterations])),\
n_A_Gill[:,:-numIterations:numIterations].reshape(np.size(n_A_Gill[:,:-numIterations:numIterations])))),shape=(maxN + 1,maxN + 1))

""" Minimizing the negative log of likelihood (and therefore maximizing likelihood) using scipy.minimize """
loglike = float('NaN')*np.ones(maxMaxl_a)
finalGuess = float('NaN')*np.ones((maxMaxl_a,3))
numBytes = []
matPop = []
for maxl_a in range(minMaxl_a,maxMaxl_a + 1):
    """ Taking a first pass across possible values of maximum l_alpha (M) """
    print('Max l_a = ' + str(maxl_a))
    if maxl_a == minMaxl_a:
        """ Starting out with a random guess """
        bestGuess = [np.random.uniform(low=-2,high=-1),np.random.uniform(low=1,high=2),0.03]
    else:
        """ Using the previous lagrange values as a starting point """
        bestGuess = finalGuess[maxl_a - 2]
    res = minimize(maxCal_mle,bestGuess,method='nelder-mead',\
    tol=0.1,options={'disp':True,'maxiter':500})
    loglike[maxl_a - 1] = res['fun']
    finalGuess[maxl_a - 1] = res['x']
""" Printing the matrix power time for illustrative purposes """
print('Matrix Power Time: ')
print( str(np.mean(stepTime)) + ' +/- ' + str(np.std(stepTime)))

""" Plotting the negative log likelihood as a function of max l_alpha for a progress report """
fig = plt.figure()
plt.plot(range(1,len(loglike) + 1),loglike)
plt.xlabel('Max l_alpha')
plt.ylabel('Negative Log Likelihood')
plt.title('Pass #1')
plt.grid(True)
plt.savefig('LogLikeVsMaxl_a,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + ',Pass1.pdf')
plt.close(fig)
del fig

""" Saving best MaxCal parameters up to this point in an npz file """
np.savez_compressed('ExtractedParameters,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess)

""" Sweeping back and forth through possible values of maximum l_alpha """
timeVals = []
numBytes = []
newFitTime = []
for numTry in range(1):
    for maxl_a in range(maxMaxl_a - 1,minMaxl_a - 1,-1):
        print('Max l_a = ' + str(maxl_a))
        bestGuess = finalGuess[maxl_a] + np.array([0.2*np.random.rand(1)[0] - 0.1,\
        0.2*np.random.rand(1)[0] - 0.1,0.02*np.random.rand(1)[0] - 0.01])
        startTime = datetime.datetime.now()
        res = minimize(maxCal_mle,bestGuess,method='nelder-mead',\
        tol=0.1,options={'disp':True,'maxiter':500})
        endTime = datetime.datetime.now()
        timeVals.append((endTime - startTime).total_seconds())
        if res['fun'] < loglike[maxl_a - 1]:
            loglike[maxl_a - 1] = res['fun']
            finalGuess[maxl_a - 1] = res['x']
    
    """ Plotting the negative log likelihood as a function of max l_alpha for a progress report """
    fig = plt.figure()
    plt.plot(range(1,len(loglike) + 1),loglike)
    plt.xlabel('Max l_alpha')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Pass #' + str((numTry + 1)*2))
    plt.grid(True)
    plt.savefig('LogLikeVsMaxl_a,SelfPromoMaxCal,' + simConditions + \
    'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
    ',Inc' + str(timeInc) + ',Iter' + str(numIterations) + ',Pass' + str((numTry + 1)*2) + '.pdf')
    plt.close(fig)
    del fig
    
    """ Saving best MaxCal parameters up to this point in an npz file """
    np.savez_compressed('ExtractedParameters,SelfPromoMaxCal,' + simConditions + \
    'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
    ',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz', \
    conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess)
    
    for maxl_a in range(minMaxl_a + 1,maxMaxl_a + 1):
        print('Max l_a = ' + str(maxl_a))
        bestGuess = finalGuess[maxl_a - 2] + np.array([0.2*np.random.rand(1)[0] - 0.1,\
        0.2*np.random.rand(1)[0] - 0.1,0.02*np.random.rand(1)[0] - 0.01])
        stepTime = []
        startTime = datetime.datetime.now()
        res = minimize(maxCal_mle,bestGuess,method='nelder-mead',\
        tol=0.1,options={'disp':True,'maxiter':500})
        endTime = datetime.datetime.now()
        timeVals.append((endTime - startTime).total_seconds())
        newFitTime.append(sum(stepTime))
        if res['fun'] < loglike[maxl_a - 1]:
            loglike[maxl_a - 1] = res['fun']
            finalGuess[maxl_a - 1] = res['x']
    
    """ Plotting the negative log likelihood as a function of max l_alpha for a progress report """
    fig = plt.figure()
    plt.plot(range(1,len(loglike) + 1),loglike)
    plt.xlabel('Max l_alpha')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Pass #' + str((numTry + 1)*2 + 1))
    plt.grid(True)
    plt.savefig('LogLikeVsMaxl_a,SelfPromoMaxCal,' + simConditions + \
    'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
    ',Inc' + str(timeInc) + ',Iter' + str(numIterations) + ',Pass' + str((numTry + 1)*2 + 1) + '.pdf')
    plt.close(fig)
    del fig
    
    """ Saving best MaxCal parameters up to this point in an npz file """
    np.savez_compressed('ExtractedParameters,SelfPromoMaxCal,' + simConditions + \
    'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
    ',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz', \
    conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess)
del numTry

""" Identifying parameters with best likelihood and calculating relevant stats """
maxl_a = int(np.where(loglike == np.nanmin(loglike))[0][0]) + 1
bestGuess = finalGuess[maxl_a - 1]
equilProb_MaxCal = maxCalEquil(bestGuess,maxl_a)
maxInds_MaxCal = peakVals(equilProb_MaxCal[:75],5,0.0005)
prodRates = float('NaN')*np.ones(len(maxInds_MaxCal))
degRates = float('NaN')*np.ones(len(maxInds_MaxCal))
for ind in range(len(maxInds_MaxCal)):
    prodRates[ind],degRates[ind] = rateCalc(bestGuess,maxl_a,maxInds_MaxCal[ind])
del ind
prodRates /= timeInc
degRates /= timeInc
feedback = feedbackCalc(bestGuess,maxl_a,equilProb_MaxCal)

""" Initializing plotting conditions... """
mpl.rc('font',family='Arial')
mpl.rc('font',size=12)
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
plt.rc('font',weight='bold')

""" Plotting protein number distributions... """
fig = plt.figure()
ax = fig.gca()
plt.plot(range(len(simHist_Gill)),simHist_Gill,'b',linewidth=2.0)
plt.plot(range(len(equilProb_MaxCal)),equilProb_MaxCal,'r',linewidth=2.0)
#plt.grid(True)
ax.text(6,0.075,'A',fontsize=30,fontweight='bold')
plt.axis([0,80,0,0.09])
plt.xticks(np.arange(0,81,20))
plt.yticks(np.arange(0.0,0.081,0.02))
plt.xlabel('# of proteins',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
#plt.legend(['Gillespie','MaxCal'],fontsize=15)
plt.tight_layout()
plt.savefig('ProteinNumberDistribution,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.pdf')

""" Saving final fitted parameters and statistics """
np.savez_compressed('ExtractedParameters,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, maxl_a=maxl_a, bestGuess=bestGuess, \
equilProb_MaxCal=equilProb_MaxCal, maxInds_MaxCal=maxInds_MaxCal, \
prodRates=prodRates, degRates=degRates, feedback=feedback, \
timeVals=timeVals)

""" Initializing fitted MaxCal conditions... """
#conditions_MaxCal = conditionsInitMaxCal(h_a,h_A,k_A,maxl_a,\
#n_A_init,l_a_init,l_iA_init,inc,numSteps,numTrials)
conditions_MaxCal = conditionsInitMaxCal(bestGuess[0],bestGuess[1],bestGuess[2],maxl_a,\
5,0,5,timeInc,int((24*3600*numDays)/timeInc),numTrials)

""" Running MaxCal simulations for confirmation and entropy statistics... """
n_A_MaxCal = []
for numTrial in range(numTrials):
    randNum = np.random.uniform(low=0,high=1)
    totProb = 0.0
    for n_A_init in range(maxN):
        totProb += equilProb_MaxCal[n_A_init]
        if totProb > randNum:
            break
    n_A_MaxCal.append([n_A_init])
    del n_A_init
    del totProb
    del randNum
del numTrial
l_a_MaxCal = np.zeros((conditions_MaxCal['numTrials'],1)).tolist()
l_iA_MaxCal = np.copy(n_A_MaxCal).tolist()
n_A_MaxCal,l_a_MaxCal,l_iA_MaxCal = maxCalSim(conditions_MaxCal,n_A_MaxCal,l_a_MaxCal,l_iA_MaxCal)
totEntropy_MaxCal,stateEntropies_MaxCal,macroEntropy_MaxCal,dwellVals_MaxCal = entropyStats(n_A_MaxCal,maxInds_MaxCal)

""" Calculating MaxCal simulated dwell times... """
avgDwells_MaxCal = []
avgTotDwell_MaxCal = []
for ind in range(len(dwellVals_MaxCal)):
    avgDwells_MaxCal.append(np.average(dwellVals_MaxCal[ind])*conditions_MaxCal['inc'])
    avgTotDwell_MaxCal.extend(dwellVals_MaxCal[ind])
del ind
avgTotDwell_MaxCal = np.average(avgTotDwell_MaxCal)*conditions_MaxCal['inc']

""" Saving final fitted parameters and statistics """
np.savez_compressed('ExtractedParameters,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, maxl_a=maxl_a, bestGuess=bestGuess, \
equilProb_MaxCal=equilProb_MaxCal, maxInds_MaxCal=maxInds_MaxCal, \
prodRates=prodRates, degRates=degRates, feedback=feedback, \
conditions_MaxCal=conditions_MaxCal, n_A_MaxCal=n_A_MaxCal, \
l_a_MaxCal=l_a_MaxCal, l_iA_MaxCal=l_iA_MaxCal, stateEntropies_MaxCal=stateEntropies_MaxCal, \
macroEntropy_MaxCal=macroEntropy_MaxCal, totEntropy_MaxCal=totEntropy_MaxCal, \
dwellVals_MaxCal=dwellVals_MaxCal, avgDwells_MaxCal=avgDwells_MaxCal, \
avgTotDwell_MaxCal=avgTotDwell_MaxCal, \
timeVals=timeVals)

""" Calculating dwell times from simulations for plotting... """
dwellProbs_MaxCal = dwellHistMaxCal(conditions_MaxCal,maxInds_MaxCal,0.001,30)
dwellInds_MaxCal = []
avgDwells_MaxCalFSP = []
for ind in range(len(dwellProbs_MaxCal)):
    dwellInds_MaxCal.append(conditions_MaxCal['inc']*np.arange(0,30*len(dwellProbs_MaxCal[ind]),30))
    avgDwells_MaxCalFSP.append(sum(dwellInds_MaxCal[ind]*dwellProbs_MaxCal[ind]))
del ind
dwellInds_Gill = [[] for ind in range(len(dwellVals_Gill))]
dwellProbs_Gill = [[] for ind in range(len(dwellVals_Gill))]
for ind in range(len(dwellVals_Gill)):
    dwellProbs_Gill[ind],dwellInds_Gill[ind] = np.histogram(np.array(dwellVals_Gill[ind])*conditions_Gill['inc'],np.arange(0,300000,9000))
    dwellProbs_Gill[ind] = dwellProbs_Gill[ind]/sum(dwellProbs_Gill[ind])
del ind
dwellInds_MaxCalSim = [[] for ind in range(len(dwellVals_MaxCal))]
dwellProbs_MaxCalSim = [[] for ind in range(len(dwellVals_MaxCal))]
for ind in range(len(dwellVals_MaxCal)):
    dwellProbs_MaxCalSim[ind],dwellInds_MaxCalSim[ind] = np.histogram(np.array(dwellVals_MaxCal[ind])*conditions_MaxCal['inc'],np.arange(0,300000,9000))
    dwellProbs_MaxCalSim[ind] = dwellProbs_MaxCalSim[ind]/sum(dwellProbs_MaxCalSim[ind])
del ind

""" Plotting low-to-high dwell time distributions... """
fig = plt.figure()
ax = fig.gca()
plt.plot(dwellInds_Gill[0][:-1],dwellProbs_Gill[0],'b',linewidth=2.0)
#plt.plot(dwellInds_MaxCal[0],dwellProbs_MaxCal[0],'r',linewidth=2.0)
plt.plot(dwellInds_MaxCalSim[0][:-1],dwellProbs_MaxCalSim[0],'r',linewidth=2.0)
#plt.grid(True)
ax.text(15000,0.115,'B',fontsize=30,fontweight='bold')
plt.axis([0,300000,0,0.14])
plt.xticks(np.arange(0,300001,100000))
plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
ax.xaxis.major.formatter._useMathText = True
plt.yticks(np.arange(0,0.14,0.04))
plt.xlabel('Low-to-High Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
#plt.legend(['Gillespie','MaxCal'],fontsize=15)
plt.tight_layout()
plt.savefig('LowToHighDwellDistribution,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.pdf')

""" Plotting high-to-low dwell time distributions... """
fig = plt.figure()
ax = fig.gca()
plt.plot(dwellInds_Gill[1][:-1],dwellProbs_Gill[1],'b',linewidth=2.0)
#plt.plot(dwellInds_MaxCal[1],dwellProbs_MaxCal[1],'r',linewidth=2.0)
plt.plot(dwellInds_MaxCalSim[1][:-1],dwellProbs_MaxCalSim[1],'r',linewidth=2.0)
#plt.grid(True)
ax.text(15000,0.115,'C',fontsize=30,fontweight='bold')
plt.axis([0,300000,0,0.14])
plt.xticks(np.arange(0,300001,100000))
plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
ax.xaxis.major.formatter._useMathText = True
plt.yticks(np.arange(0,0.14,0.04))
plt.xlabel('High-to-Low Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
#plt.legend(['Gillespie','MaxCal'],fontsize=15)
plt.tight_layout()
plt.savefig('HighToLowDwellDistribution,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.pdf')

""" Saving final fitted parameters and statistics """
np.savez_compressed('ExtractedParameters,SelfPromoMaxCal,' + simConditions + \
'Sim' + str(simNum) + ',Trials' + str(numTrials) + ',Days' + str(numDays) + \
',Inc' + str(timeInc) + ',Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, maxl_a=maxl_a, bestGuess=bestGuess, \
equilProb_MaxCal=equilProb_MaxCal, maxInds_MaxCal=maxInds_MaxCal, \
prodRates=prodRates, degRates=degRates, feedback=feedback, \
conditions_MaxCal=conditions_MaxCal, n_A_MaxCal=n_A_MaxCal, \
l_a_MaxCal=l_a_MaxCal, l_iA_MaxCal=l_iA_MaxCal, stateEntropies_MaxCal=stateEntropies_MaxCal, \
macroEntropy_MaxCal=macroEntropy_MaxCal, totEntropy_MaxCal=totEntropy_MaxCal, \
dwellVals_MaxCal=dwellVals_MaxCal, avgDwells_MaxCal=avgDwells_MaxCal, \
avgTotDwell_MaxCal=avgTotDwell_MaxCal, dwellProbs_MaxCal=dwellProbs_MaxCal, \
dwellInds_MaxCal=dwellInds_MaxCal, avgDwells_MaxCalFSP=avgDwells_MaxCalFSP, \
dwellInds_MaxCalSim=dwellInds_MaxCalSim, dwellProbs_MaxCalSim=dwellProbs_MaxCalSim, \
timeVals=timeVals, numBytes=numBytes, matPop=matPop, stepTime=stepTime)
#print('numBytes: ' + str(np.mean(numBytes)))






