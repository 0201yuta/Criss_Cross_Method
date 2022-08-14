import numpy as np
from numpy import linalg as LA
import os.path

def Pivot(tableau, b, n): 
    bn = tableau[b + 1, n + 1]
    v1 = tableau[:, n + 1] / bn
    v2 = tableau[b + 1] / bn
    v2[n + 1] = 1 / bn
    tableau = tableau - np.outer(v1, tableau[b + 1])
    tableau[:, n + 1] = -v1
    tableau[b + 1] = v2
    return tableau

def CCMethod(tableau): #algorithm1を実装したCrissCrossMethod
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < 0)[0] + col
    J = np.where(tableau[0, 1:] < 0)[0]
    Index = np.array(list(range(0, row + col)))
    Index = np.hstack([Index[-row:],Index[:-row]])
    count = 0
    while I.size + J.size != 0:
        I_index = np.array([Index[x] for x in I])
        J_index = np.array([Index[y] for y in J])
        if I_index.size != 0:
            k_I = np.amin(I_index) #Iの中で最も添え字が小さいものを選ぶ
        else:
            k_I = row + col    
        if J_index.size != 0:       
            k_J = np.amin(J_index) #Jの中で最も添え字が小さいものを選ぶ
        else:
            k_J = row + col        
        if k_I < k_J: #step3を実行
            k = np.where(Index == k_I)[0][0] - col #k_Iが何行目に格納されているかを表す式
            S = np.where(tableau[k + 1, 1:] < 0)[0] 
            if S.size != 0:
                j = np.where(Index == np.amin(np.array([Index[x] for x in S])))[0][0]
                tableau = Pivot(tableau, k, j)
                Index[k + col], Index[j] = Index[j], Index[k + col]
            else:
                return "no feasible solution",count    
        else: #step4を実行 
            k = np.where(Index == k_J)[0][0]
            T = np.where(tableau[1:, k + 1] > 0)[0] + col
            if T.size != 0:
                i = np.where(Index == np.amin(np.array([Index[x] for x in T])))[0][0]
                tableau = Pivot(tableau, i - col, k)
                Index[k], Index[i] = Index[i], Index[k]
            else:
                return "no feasible solution",count                       
        I = np.where(tableau[1:, 0] < 0)[0] + col 
        J = np.where(tableau[0, 1:] < 0)[0]  
        count = count + 1
    return tableau,count    

#基底解と非基底解を別のリストで保存、alphaは添え字順に保存することで可読性を上げる
def Practical(tableau, A, b, error): 
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]  
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Basis = np.arange(row) #基底解の添え字を上の行から順に格納
    NonBasis = np.arange(row, row + col) #非基底解の添え字を左の列から順に格納
    alpha = np.array([np.dot(b, a) / np.power(np.linalg.norm(a, ord=2), 1) for a in A.T]) #添え字順にalphaを格納
    #alpha = np.arange(row + col)
    count = 0
    while I.size + J.size != 0:
        IJ_Index = np.union1d(np.array([Basis[x] for x in I]), np.array([NonBasis[x] for x in J]))
        IJ_Index = IJ_Index.astype(np.int64)
        min_alpha = np.amin(np.array([alpha[x] for x in IJ_Index]))
        k = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], IJ_Index))
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
                return "no feasible solution",count 
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
                return "no feasible solution",count
        I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]
        J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
        count = count + 1       
    return tableau, Basis, count
 
def solve_mps(path1, path2, error):
    tableau = np.loadtxt(path1, delimiter = ',')
    Ab = np.loadtxt(path2, delimiter = ',')
    ans_new = Practical(tableau, Ab[:, 0:-1], Ab[:, -1], error)
    ans_old = CCMethod(tableau)
    return ans_new[0][0][0], ans_new[2], ans_old[0][0][0], ans_old[1]


#sample = np.array([[191.5, 5.5, -0.5, 4.5, 0], [-5.75, -0.25, 0.25, -0.25, 0], [40.75, 1.25, -0.25, 1.25, 0], [-6.25, 0, 0, -1.25, -0.25], [11.25, 0, 0, 0.25, 0.25]])
#A = np.array([[1,1,1,1,1,0,0,0], [5,1,0,0,0,1,0,0], [0,0,-1,-1,0,0,1,0], [0,0,1,5,0,0,0,8]])
#b = np.array([40,12,-5,50])

root = r'C:\Users\pc\Documents\MATLAB'
path = r'\sc50b\sc50b'
print(solve_mps(root + path + r'.txt', root + path + r'_Ab.txt', 6))



