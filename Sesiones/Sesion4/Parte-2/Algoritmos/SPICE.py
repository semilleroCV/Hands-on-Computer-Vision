# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:52:00 2018

@author: weihuang.xu
"""
#==============================================================================
# SPICE Sparsity Promoting Iterated Constrained Endmembers Algorithm
#       Finds Endmembers and Unmixes Input Data
# 
# Syntax: [endmembers, P] = SPICE(inputData, parameters)
# 
# Inputs:
#   inputData - double Mat - NxM matrix of M data points of
#       dimensionality N (i.e.  M pixels with N spectral bands, each pixel is
#       a column vector)
#   parameters - struct - The struct contains the following fields:
#                   1. u : Regularization Parameter for RSS and V terms
#                   2. gamma: Gamma Constant for SPT term
#                   3. changeThresh: Stopping Criteria, Change threshold
#                       for Objective Function.
#                   4. M: Initial Number of endmembers
#                   5. iterationCap: Maximum number of iterations
#                   6. endmemberPruneThreshold: Proportion threshold used
#                      to prune endmembers
#                   7. produceDisplay : Set to 1 if a progress display is
#                       wanted
#                   8. initEM: Set to nan to randomly select endmembers,
#                       otherwise NxM matrix of M endmembers with N spectral
#                       bands, Number of endmembers must equal parameters.M
# Outputs:
#   endmembers - double Mat - NxM matrix of M endmembers with N spectral
#       bands
#   P - double Mat - NxM matrix of abundances corresponding to M input
#       pixels and N endmembers
# Other m-files required: unmix, Matlab Optimization Toolbox
# 
# Author: Alina Zare
# University of Missouri, Electrical and Computer Engineering
# Email Address: azare@ufl.edu
# Created: August 2006
# Latest Revision: November 22, 2011
# This product is Copyright (c) 2013 University of Missouri and University
# of Florida
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   3. Neither the name of the University nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF MISSOURI AND
# CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED.  IN NO EVENT SHALL THE UNIVERSITY OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES,
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#==============================================================================

import numpy as np
from Algoritmos.QPP import quadprog_solve_qp

class SPICEParameters():
    
    def __init__(self):
        self.u = 0.001  #Trade-off parameter between RSS and V term
        self.gamma = 5  #Sparsity parameter
        self.M = 20  #Initial number of endmembers
        self.endmemberPruneThreshold = 1e-9
        self.changeThresh = 1e-4  #Used as the stopping criterion
        self.iterationCap = 5000 #Alternate stopping criterion
        self.produceDisplay = 1
        self.initEM = None  #This randomly selects parameters.M initial endmembers from the input data


def SPICE(inputData, parameters):
    input_params = parameters
    parameters = SPICEParameters()
    for k,v in input_params.__dict__.items():
        parameters.__dict__[k] = v

    parameters.pruningIteration = 1
    M = parameters.M
    X = inputData
    
    if parameters.initEM is None:
        # Find Random Initial Endmembers
        randIndices = np.random.permutation(inputData.shape[1])
        randIndices = randIndices[0:parameters.M]
        endmembers = inputData[:,randIndices]
        parameters.initEM = endmembers
    else:
        # Use endmembers provided
        M = parameters.initEM.shape[1]
        endmembers = parameters.initEM
    
    # N is the number of pixels, RSSreg is the current objective function total.
    N = X.shape[1]
    RSSreg = np.inf
    change = np.inf
    
    iteration = 0
    P = np.ones((N,M))*(1/M)
    lamb = N*parameters.u/((M-1)*(1-parameters.u))
    Im = np.eye(M)
    I1 = np.ones((M,1))
    
    while((change > parameters.changeThresh) and (iteration < parameters.iterationCap)):
        
        iteration = iteration + 1

        # Given Endmembers, minimize P -- Quadratic Programming Problem
        P = unmix2(X, endmembers, parameters.gamma, P)
        
        # Given P minimize Endmembers
        endmembersPrev = endmembers
        endmembers = (np.linalg.inv(P.T@P + lamb*(Im - (I1@I1.T)/M)) @ (P.T @ X.T)).T
                                    
        
        # Prune Endmembers below pruning threshold
        pruneFlag = 0
       
        pruneIndex = (P.max(0)<parameters.endmemberPruneThreshold)*1
        minmaxP = P.max(0).min()
       
        if(sum(pruneIndex) > 0):
            pruneFlag = 1
            
            endmembers = endmembers[:,np.where(pruneIndex==0)].squeeze()
            P = P[:, np.where(pruneIndex==0)].squeeze()
            M = M - sum(pruneIndex)
            lamb = N*parameters.u/((M-1)*(1-parameters.u))
            Im = np.eye(M)
            I1 = np.ones((M,1))
        
        # Calculate RSSreg (the current objective function value)
        
        sqerr = X - (endmembers @ P.T)
        sqerr = np.power(sqerr, 2) 
        RSS = sum(sum(sqerr))
        V = sum(sum(np.multiply(endmembers,endmembers),2) - (1/M)*np.multiply(sum(endmembers,2),2)/(M-1))
        SPT = M*parameters.gamma
        RSSprev = RSSreg
        RSSreg = (1-parameters.u)*(RSS/N) + parameters.u*V + SPT
        
        # Determine if Change Threshold has been reached
        change = (abs(RSSreg - RSSprev))
    
        if(parameters.produceDisplay):
            print(' ')
            print('Change in Objective Function Value: {}'.format(change))
            print('Minimum of Maximum Proportions: {}'.format(minmaxP))
            print('Number of Endmembers: {}'.format(M))
            print('Iteration: {}'.format(iteration))
            print(' ')
    
    return endmembers, P
    

#==============================================================================
# This product is Copyright (c) 2013 University of Missouri and University
# of Florida
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   3. Neither the name of the University nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF MISSOURI AND
# CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED.  IN NO EVENT SHALL THE UNIVERSITY OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES,
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#==============================================================================
## CVXOPT package is used here. Parameters H,F,L,K,Aeq,beq are corresbonding to 
## P,q,G,h,A,B, respectively. lb and ub are element-wise bound constraints which 
## are added to matrix G and h respectively.

def unmix2(data, endmembers, gammaConst=0, P=None):
    
    X = data  #endmembers should be column vectors
    M = endmembers.shape[1]  #number of endmembers
    N = X.shape[1]  #number of pixels
    
    #Equation constraint Aeq*x = beq
    #All values must sum to 1 (X1+X2+...+XM = 1)
    Aeq = np.ones((1, M))
    beq = np.ones((1, 1))
    
    #Boundary Constraints ub >= x >= lb
    #All values must be greater than 0 (0 ? X1,0 ? X2,...,0 ? XM)
    lb = 0
    ub = 1
    g_lb = np.eye(M)*-1
    g_ub = np.eye(M)
    #import pdb; pdb.set_trace()
    G = np.concatenate((g_lb,g_ub), axis=0)
    h_lb = np.ones((M, 1))*lb
    h_ub = np.ones((M, 1))*ub
    h = np.concatenate((h_lb,h_ub),axis=0)
    if P is None:
        P = np.ones((M,1))/M
    gammaVecs = np.divide(gammaConst,sum(P))
    
    H = 2 * (endmembers.T @ endmembers)
    P2 = np.zeros((N, M))
    for i in range(N):
        F = ((np.transpose(-2*X[:,i]) @ endmembers)+gammaVecs).T
        qpas_ans = quadprog_solve_qp(P=H, q=F, G=G, h=h.T, A=Aeq, b=beq.T, initvals=None)
        P2[i,:] = qpas_ans
    
    P2[P2<0] = 0
    
    return P2  
        
    
