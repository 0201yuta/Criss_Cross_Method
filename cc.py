import numpy as np
from numpy import linalg as LA
import os.path
import csv
import matplotlib.pyplot as plt
import os
import time

def Pivot(tableau, b, n): 
    bn = tableau[b + 1, n + 1]
    v1 = tableau[:, n + 1] / bn
    v2 = tableau[b + 1] / bn
    v2[n + 1] = 1 / bn
    tableau = tableau - np.outer(v1, tableau[b + 1])
    tableau[:, n + 1] = -v1
    tableau[b + 1] = v2
    return tableau

def Practical(tableau, A, b, error): 
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]  
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Basis = np.arange(row) #基底解の添え字を上の行から順に格納
    NonBasis = np.arange(row, row + col) #非基底解の添え字を左の列から順に格納
    alpha = np.array([np.dot(b, a) / np.power(np.linalg.norm(a, ord=2), 1) for a in A.T]) #添え字順にalphaを格納
    alpha_Index = np.argsort(alpha)
    count = 0
    obj_his = [- tableau[0, 0]]
    B_his = [I.size]
    NB_his = [J.size]  
    while I.size + J.size != 0 and count < 5000:
        IJ_Index = np.union1d(np.array([Basis[x] for x in I]), np.array([NonBasis[x] for x in J]))
        IJ_Index = IJ_Index.astype(np.int64)
        k = np.amin(np.intersect1d(alpha_Index, IJ_Index))
        if k in Basis:
            k_store = np.where(Basis == k)[0][0]
            S = np.where(tableau[k_store + 1, 1:] < - np.power(10., -error))[0]
            if S.size != 0:
                S_Index = np.array([NonBasis[x] for x in S])
                min_alpha = np.amin(np.array([alpha[x] for x in S_Index])) 
                j = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], S_Index))
                j_store = np.where(NonBasis == j)[0][0]
                tableau = Pivot(tableau, k_store, j_store)
                Basis[k_store] = j
                NonBasis[j_store] = k
            else:
                return "no feasible solution", tableau, count, obj_his, B_his, NB_his
        else:
            k_store = np.where(NonBasis == k)[0][0]              
            T = np.where(tableau[1:, k_store + 1] > np.power(10., -error))[0]
            if T.size !=  0:
                T_Index = np.array([Basis[x] for x in T])
                min_alpha = np.amin(np.array([alpha[x] for x in T_Index])) 
                i = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], T_Index))
                i_store = np.where(Basis == i)[0][0]
                tableau = Pivot(tableau, i_store, k_store)
                Basis[i_store] = k
                NonBasis[k_store] = i
            else:
                return "no feasible solution", tableau, count, obj_his, B_his, NB_his
        I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]
        J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
        count = count + 1  
        obj_his.append(- tableau[0, 0])
        B_his.append(I.size)
        NB_his.append(J.size)  
    if count < 5000:        
        return "optimal solution", tableau, count, obj_his, B_his, NB_his
    else:    
        return "no feasible solution", tableau, count, obj_his, B_his, NB_his