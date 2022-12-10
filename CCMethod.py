import numpy as np
from numpy import linalg as LA
import os.path
import csv
import matplotlib.pyplot as plt
import os

def Pivot(tableau, b, n): 
    bn = tableau[b + 1, n + 1]
    v1 = tableau[:, n + 1] / bn
    v2 = tableau[b + 1] / bn
    v2[n + 1] = 1 / bn
    tableau = tableau - np.outer(v1, tableau[b + 1])
    tableau[:, n + 1] = -v1
    tableau[b + 1] = v2
    return tableau

def CCMethod(tableau, error): #algorithm1を実装したCrissCrossMethod
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0] + col
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Index = np.array(list(range(0, row + col)))
    Index = np.hstack([Index[-row:],Index[:-row]])
    count = 0
    obj_his = [- tableau[0, 0]]
    B_his = [I.size]
    NB_his = [J.size]     
    while I.size + J.size != 0 and count < 5000:
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
            S = np.where(tableau[k + 1, 1:] < - np.power(10., -error))[0] 
            if S.size != 0:
                j = np.where(Index == np.amin(np.array([Index[x] for x in S])))[0][0]
                tableau = Pivot(tableau, k, j)
                Index[k + col], Index[j] = Index[j], Index[k + col]
            else:
                return "no feasible solution", tableau, count, obj_his, B_his, NB_his
        else: #step4を実行 
            k = np.where(Index == k_J)[0][0]
            T = np.where(tableau[1:, k + 1] > np.power(10., -error))[0] + col
            if T.size != 0:
                i = np.where(Index == np.amin(np.array([Index[x] for x in T])))[0][0]
                tableau = Pivot(tableau, i - col, k)
                Index[k], Index[i] = Index[i], Index[k]
            else:
                return "no feasible solution", tableau, count, obj_his, B_his, NB_his                       
        I = np.where(tableau[1:, 0] < 0)[0] + col 
        J = np.where(tableau[0, 1:] < 0)[0]  
        count = count + 1
        obj_his.append(- tableau[0, 0])
        B_his.append(I.size)
        NB_his.append(J.size)             
    if count < 5000:
        return "optimal solution", tableau, count, obj_his, B_his, NB_his
    else:
        return "no feasible solution", tableau, count, obj_his, B_his, NB_his

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
    obj_his = [- tableau[0, 0]]
    B_his = [I.size]
    NB_his = [J.size]   
    while I.size + J.size != 0 and count < 5000:
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

#最大係数規則とalgorithm2をランダムで選んで実行する新しいalgorithm
#今回は目的関数がminなので目的関数の非基底変数の係数が最も小さいものを選ぶ
#終了条件はCrissCrossMethodのものをそのまま使う
def random_method(tableau, A, b, error):
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]  
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Basis = np.arange(row) #基底解の添え字を上の行から順に格納
    NonBasis = np.arange(row, row + col) #非基底解の添え字を左の列から順に格納
    alpha = np.array([np.dot(b, a) / np.power(np.linalg.norm(a, ord=2), 1) for a in A.T]) #添え字順にalphaを格納
    count = 0
    cc_count = 0
    rule = 0
    while I.size + J.size != 0 and count < 5000:
        if np.random.rand() < 0.5 or J.size == 0: #CCpivotを実行
            IJ_Index = np.union1d(np.array([Basis[x] for x in I]), np.array([NonBasis[x] for x in J]))
            IJ_Index = IJ_Index.astype(np.int64)
            min_alpha = np.amin(np.array([alpha[x] for x in IJ_Index]))
            k = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], IJ_Index))
            cc_count = cc_count + 1
        else:
            k = NonBasis[np.argmin(tableau[0, 1:])]       
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
                return "no feasible solution", tableau, count, cc_count 
        else:
            k_store = np.where(NonBasis == k)[0][0]              
            T = np.where(tableau[1:, k_store + 1] > np.power(10., -error))[0]
            if T.size !=  0:
                if rule == 0: #基底変数を好きな規則で選ぶ、0はαの優先規則                  
                    T_Index = np.array([Basis[x] for x in T])
                    min_alpha = np.amin(np.array([alpha[x] for x in T_Index])) 
                    i = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], T_Index))
                elif rule == 1: #1は最小添字規則
                    i = np.min(np.array([Basis[x] for x in T]))
                    rule = 0
                i_store = np.where(Basis == i)[0][0]
                tableau = Pivot(tableau, i_store, k_store)  
                Basis[i_store] = k 
                NonBasis[k_store] = i
            else:
                return "no feasible solution", tableau, count, cc_count
        I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]
        J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
        count = count + 1  
    if count < 5000:        
        return "optimal solution", tableau, count, cc_count
    else:    
        return "no feasible solution", tableau, count, cc_count       


def CCandGI(tableau, A, b, error, balance): 
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]  
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Basis = np.arange(row) #基底解の添え字を上の行から順に格納
    NonBasis = np.arange(row, row + col) #非基底解の添え字を左の列から順に格納
    alpha = np.array([np.dot(b, a) / np.power(np.linalg.norm(a, ord=2), 1) for a in A.T]) #添え字順にalphaを格納
    count = 0
    obj_his = [- tableau[0, 0]]
    B_his = [I.size]
    NB_his = [J.size]  
    while I.size + J.size != 0 and count < 5000:
        IJ_Index = np.union1d(np.array([Basis[x] for x in I]), np.array([NonBasis[x] for x in J]))
        IJ_Index = IJ_Index.astype(np.int64)
        min_alpha = np.amin(np.array([alpha[x] for x in IJ_Index]))
        k = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], IJ_Index))
        if k in Basis:
            k_store = np.where(Basis == k)[0][0]
            S = np.where(tableau[k_store + 1, 1:] < - np.power(10., -error))[0]
            if S.size != 0:
                S_Index = np.array([NonBasis[x] for x in S])
                S_alpha = np.array([alpha[x] for x in S_Index])
                S_Index = S_Index[np.argsort(S_alpha)]
                S = S[np.argsort(S_alpha)]
                S_Index = S_Index[:int(np.ceil(S.size * balance))]
                S = S[:int(np.ceil(S.size * balance))]
                GI = np.argmin(np.array([tableau[k_store + 1, 0] * tableau[0, x + 1] / tableau[k_store + 1, x + 1] for x in S]))
                j = S_Index[GI]
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
                T_alpha = np.array([alpha[x] for x in T_Index])
                T_Index = T_Index[np.argsort(T_alpha)]
                T = T[np.argsort(T_alpha)]
                T_Index = T_Index[:int(np.ceil(T.size * balance))]     
                T = T[:int(np.ceil(T.size * balance))]  
                GI = np.argmin(np.array([tableau[y + 1, 0] * tableau[0, k_store + 1] / tableau[y + 1, k_store + 1] for y in T]))
                i = T_Index[GI]                         
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

def CCandGInew(tableau, A, b, error, balance): 
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]  
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Basis = np.arange(row) #基底解の添え字を上の行から順に格納
    NonBasis = np.arange(row, row + col) #非基底解の添え字を左の列から順に格納
    alpha = np.array([np.dot(b, a) / np.power(np.linalg.norm(a, ord=2), 1) for a in A.T]) #添え字順にalphaを格納
    count = 0
    while I.size + J.size != 0 and count < 5000:
        IJ_Index = np.union1d(np.array([Basis[x] for x in I]), np.array([NonBasis[x] for x in J]))
        IJ_Index = IJ_Index.astype(np.int64)
        min_alpha = np.amin(np.array([alpha[x] for x in IJ_Index]))
        k = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], IJ_Index))
        if k in Basis:
            k_store = np.where(Basis == k)[0][0]
            S = np.where(tableau[k_store + 1, 1:] < - np.power(10., -error))[0]
            if S.size != 0:
                if J.size != 0:
                    S_Index = np.array([NonBasis[x] for x in S])
                    S_alpha = np.array([alpha[x] for x in S_Index])
                    S_Index = S_Index[np.argsort(S_alpha)]
                    S = S[np.argsort(S_alpha)]
                    S_Index = S_Index[:int(np.ceil(S.size * balance))]
                    S = S[:int(np.ceil(S.size * balance))]
                    GI = np.argmin(np.array([tableau[k_store + 1, 0] * tableau[0, x + 1] / tableau[k_store + 1, x + 1] for x in S]))
                    j = S_Index[GI]
                else: #なるべく実行可能解に戻す作業を行う
                    min_cr = row * col
                    for s in S:
                        ks = tableau[k_store + 1, s + 1]
                        c = tableau[:, 0] - tableau[k_store + 1, 0] / ks * tableau[:, s + 1]
                        r = tableau[0, :] - tableau[0, s + 1] / ks * tableau[k_store + 1, :]
                        cr = row * np.count_nonzero(c[1:] <  - np.power(10., -error)) + np.count_nonzero(r[1:] <  - np.power(10., -error))
                        if min_cr > cr or (min_cr == cr and min_change < c[0]):
                            j = NonBasis[s]
                            min_cr = cr 
                            min_change = c[0]
                        else:
                            pass
                j_store = np.where(NonBasis == j)[0][0]
                tableau = Pivot(tableau, k_store, j_store)
                Basis[k_store] = j
                NonBasis[j_store] = k    
            else:
                return "no feasible solution", tableau, count 
        else:
            k_store = np.where(NonBasis == k)[0][0]              
            T = np.where(tableau[1:, k_store + 1] > np.power(10., -error))[0]
            if T.size !=  0:
                T_Index = np.array([Basis[x] for x in T])
                T_alpha = np.array([alpha[x] for x in T_Index])
                T_Index = T_Index[np.argsort(T_alpha)]
                T = T[np.argsort(T_alpha)]
                T_Index = T_Index[:int(np.ceil(T.size * balance))]     
                T = T[:int(np.ceil(T.size * balance))]  
                GI = np.argmin(np.array([tableau[y + 1, 0] * tableau[0, k_store + 1] / tableau[y + 1, k_store + 1] for y in T]))
                i = T_Index[GI]                         
                i_store = np.where(Basis == i)[0][0]
                tableau = Pivot(tableau, i_store, k_store)
                Basis[i_store] = k
                NonBasis[k_store] = i
            else:
                return "no feasible solution", tableau, count
        I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]
        J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
        count = count + 1  
    if count < 5000:        
        return "optimal solution", tableau, count
    else:    
        return "no feasible solution", tableau, count        

def CCandGInew2(tableau, A, b, error, balance, max_back):     
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]  
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Basis = np.arange(row) #基底解の添え字を上の行から順に格納
    NonBasis = np.arange(row, row + col) #非基底解の添え字を左の列から順に格納
    alpha = np.array([np.dot(b, a) / np.power(np.linalg.norm(a, ord=2), 1) for a in A.T]) #添え字順にalphaを格納
    count = 0
    back = 0
    obj_his = [- tableau[0, 0]]
    B_his = [I.size]
    NB_his = [J.size]    
    while I.size + J.size != 0 and count < 5000:
        if J.size != 0 or back == max_back:
            IJ_Index = np.union1d(np.array([Basis[x] for x in I]), np.array([NonBasis[x] for x in J]))
            IJ_Index = IJ_Index.astype(np.int64)
            min_alpha = np.amin(np.array([alpha[x] for x in IJ_Index]))
            k = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], IJ_Index))
            if k in Basis:
                k_store = np.where(Basis == k)[0][0]
                S = np.where(tableau[k_store + 1, 1:] < - np.power(10., -error))[0]
                if S.size != 0:
                    S_Index = np.array([NonBasis[x] for x in S])
                    S_alpha = np.array([alpha[x] for x in S_Index])
                    S_Index = S_Index[np.argsort(S_alpha)]
                    S = S[np.argsort(S_alpha)]
                    S_Index = S_Index[:int(np.ceil(S.size * balance))]
                    S = S[:int(np.ceil(S.size * balance))]
                    GI = np.argmin(np.array([tableau[k_store + 1, 0] * tableau[0, x + 1] / tableau[k_store + 1, x + 1] for x in S]))
                    j = S_Index[GI]
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
                    T_alpha = np.array([alpha[x] for x in T_Index])
                    T_Index = T_Index[np.argsort(T_alpha)]
                    T = T[np.argsort(T_alpha)]
                    T_Index = T_Index[:int(np.ceil(T.size * balance))]     
                    T = T[:int(np.ceil(T.size * balance))]  
                    GI = np.argmin(np.array([tableau[y + 1, 0] * tableau[0, k_store + 1] / tableau[y + 1, k_store + 1] for y in T]))
                    i = T_Index[GI]                         
                    i_store = np.where(Basis == i)[0][0]
                    tableau = Pivot(tableau, i_store, k_store)
                    Basis[i_store] = k
                    NonBasis[k_store] = i
                else:
                    return "no feasible solution", tableau, count, obj_his, B_his, NB_his
        else: #実行可能解に大きく近づける作業を何回か行う
            min_cr = row * col
            obj = 1
            for k_s in I: 
                S = np.where(tableau[k_s + 1, 1:] < - np.power(10., -error))[0]                
                for j_s in S:
                    kj = tableau[k_s + 1, j_s + 1]
                    c = tableau[:, 0] - tableau[k_s + 1, 0] / kj * tableau[:, j_s + 1]
                    r = tableau[0, :] - tableau[0, j_s + 1] / kj * tableau[k_s + 1, :]
                    cr = row * np.count_nonzero(r[1:] <  - np.power(10., -error)) + np.count_nonzero(c[1:] <  - np.power(10., -error)) 
                    if min_cr > cr or (min_cr == cr and obj > c[0]):
                        j = NonBasis[j_s]
                        j_store = j_s
                        k = Basis[k_s]
                        k_store = k_s
                        min_cr = cr 
                        obj = c[0]
            tableau = Pivot(tableau, k_store, j_store)
            Basis[k_store] = j
            NonBasis[j_store] = k 
            back = back + 1
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

def CCGI(tableau, error): 
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]  
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Basis = np.arange(row) #基底解の添え字を上の行から順に格納
    NonBasis = np.arange(row, row + col) #非基底解の添え字を左の列から順に格納
    count = 0
    while I.size + J.size != 0 and count < 5000:
        min_ch = 1000000
        for k_s in I:
            S = np.where(tableau[k_s + 1, 1:] < - np.power(10., -error))[0]
            if S.size != 0:
                x = np.argmin(np.array([tableau[k_s + 1, 0] * tableau[0, x + 1] / tableau[k_s + 1, x + 1] for x in S]))
                j_s = S[x]
                ch = tableau[k_s + 1, 0] * tableau[0, j_s + 1] / tableau[k_s + 1, j_s + 1]
                if min_ch > ch:
                    min_ch = ch
                    b = k_s
                    n = j_s
            else:
                return "no feasible solution", tableau, count     
        for k_s in J:                  
            T = np.where(tableau[1:, k_s + 1] > np.power(10., -error))[0]
            if T.size !=  0:
                y = np.argmin(np.array([tableau[y + 1, 0] * tableau[0, k_s + 1] / tableau[y + 1, k_s + 1] for y in T]))
                i_s = T[y]
                ch = tableau[i_s + 1, 0] * tableau[0, k_s + 1] / tableau[i_s + 1, k_s + 1]
                if min_ch > ch:
                    min_ch = ch
                    b = i_s
                    n = k_s             
            else:
                return "no feasible solution", tableau, count      
        tableau = Pivot(tableau, b, n)    
        Basis[b],NonBasis[n] = NonBasis[n],Basis[b]              
        I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]
        J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
        count = count + 1  
    if count < 5000:        
        return "optimal solution", tableau, count
    else:    
        return "no feasible solution", tableau, count


def solve_alg(path, error):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_alg1 = CCMethod(tableau, error)
    ans_alg2 = Practical(tableau, Ab[:, 0:-1], Ab[:, -1], error)
    e_old = np.abs((-ans_alg1[1][0][0] - opt) / opt) <= np.power(10., -3)
    e_new = np.abs((-ans_alg2[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_old, ans_alg1[2], e_new, ans_alg2[2]]

def solve_imp(path, error, balance):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_GI = CCandGI(tableau, Ab[:, 0:-1], Ab[:, -1], error, balance)
    e_GI = np.abs((-ans_GI[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_GI, ans_GI[2]]

def solve_GInew(path, error, balance):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_GInew = CCandGInew(tableau, Ab[:, 0:-1], Ab[:, -1], error, balance)
    e_GInew = np.abs((-ans_GInew[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_GInew, ans_GInew[2]]

def solve_GInew2(path, error, balance, max_back):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_GInew = CCandGInew2(tableau, Ab[:, 0:-1], Ab[:, -1], error, balance, max_back)
    e_GInew = np.abs((-ans_GInew[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_GInew, ans_GInew[2]]

def solve_CCGI(path, error):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_CCGI = CCGI(tableau, error)
    e_CCGI = np.abs((-ans_CCGI[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_CCGI, ans_CCGI[2]]

def solve_rnd(path, error):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_rnd = random_method(tableau, Ab[:, 0:-1], Ab[:, -1], error)
    e_rnd = np.abs((-ans_rnd[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_rnd, ans_rnd[2], ans_rnd[3]]

def test_CC(root, error):   
    result_table = []
    #files = os.listdir(root)
    #dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        result.extend(solve_alg(path, error))
        result_table.append(result)
    return result_table       

def test_imp(root, error, balance):   
    result_table = []
    files = os.listdir(root)
    #dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]    
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        result.extend(solve_imp(path, error, balance))
        result_table.append(result)
    return result_table 

def test_GInew(root, error, balance):   
    result_table = []
    files = os.listdir(root)
    #dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]    
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        result.extend(solve_GInew(path, error, balance))
        result_table.append(result)
    return result_table 

def test_GInew2(root, error, balance, max_back):   
    result_table = []
    files = os.listdir(root)
    #dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]    
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        result.extend(solve_GInew2(path, error, balance, max_back))
        result_table.append(result)
    return result_table 

def test_CCGI(root, error):   
    result_table = []
    files = os.listdir(root)
    #dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]    
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        result.extend(solve_CCGI(path, error))
        result_table.append(result)
    return result_table 

def test_rnd(root, error, iteration):   
    result_table = []
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        success = 0
        total_count = 0
        total_cc = 0
        for x in range(iteration):
            [e, count, cc_count] = solve_rnd(path, error)#正誤判定と反復回数、CCMethodが呼び出された回数を出力
            if e:
                success = success + 1
                total_count = total_count + count
                total_cc = total_cc + cc_count
        result.extend([success, total_count, total_cc])                        
        result_table.append(result)
    return result_table 

def mpssize(root):   
    result_table = []
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        tableau = np.loadtxt(path + r'.txt', delimiter = ',')
        result.extend(tableau.shape)
        result_table.append(result)
    return result_table     

def analyze_alg1(path, error):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_alg1 = CCMethod(tableau, error)
    return [ans_alg1[3], ans_alg1[4], ans_alg1[5]] 

def analyze_alg2(path, error):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_alg2 = Practical(tableau, Ab[:, 0:-1], Ab[:, -1], error)    
    return [ans_alg2[3], ans_alg2[4], ans_alg2[5]] 

def analyze_CCGI1(path, error):
    balance = 1
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_imp = CCandGI(tableau, Ab[:, 0:-1], Ab[:, -1], error, balance)
    return [ans_imp[3], ans_imp[4], ans_imp[5]]     

def analyze_CCGI2(path, error):
    balance = 1
    max_back = 3
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_GInew = CCandGInew2(tableau, Ab[:, 0:-1], Ab[:, -1], error, balance, max_back)
    return [ans_GInew[3], ans_GInew[4], ans_GInew[5]]   

def make_figure(root, func, error):
    files = os.listdir(root)
    save_root = r'C:\Users\pc\Desktop\college\research\Criss_Cross_Method'
    fname = func.__name__
    fname = fname.replace('analyze_', r'\A')
    fname = fname.replace('A', '')
    save_root = save_root + fname
    #dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]    
    dirlist = ['afiro', 'blend', 'sc50a', 'sc50b', 'scagr7', 'seba', 'stocfor1']
    os.mkdir(save_root)
    for name in dirlist:
        P = r'\a\a'
        P2 = r'\a'
        path = root + P.replace('a', name)
        save_path = save_root + P2.replace('a', name)
        [obj, b, nb] = func(path, error)
        ite = list(range(len(obj)))
        plt.plot(ite, b, color = "r", label = "Basis")
        plt.plot(ite, nb, color = "b", label = "NonBasis")
        plt.grid(which = "both", axis="y")
        plt.xlabel("number of iteration", size = "large", color = "black")
        plt.ylabel("number of negative components", size = "large", color = "black")
        plt.legend(loc="center", bbox_to_anchor=(0.5, 1.05), ncol=2) 
        plt.savefig(save_path + '_bnb')
        plt.clf()
        plt.plot(ite, obj, color = "k", label = "GICC2")
        plt.plot(ite, [obj[-1]] * len(obj), color = "y", label = "optimal value")
        plt.xlabel("number of iteration", size = "large", color = "black")
        plt.ylabel("objective value", size = "large", color = "black")
        plt.grid(which = "both", axis="y")
        plt.legend(loc="center", bbox_to_anchor=(0.5, 1.05), ncol=2) 
        plt.savefig(save_path + '_obj')
        plt.clf()
    return "finish" 

#sample = np.array([[191.5, 5.5, -0.5, 4.5, 0], [-5.75, -0.25, 0.25, -0.25, 0], [40.75, 1.25, -0.25, 1.25, 0], [-6.25, 0, 0, -1.25, -0.25], [11.25, 0, 0, 0.25, 0.25]])
#A = np.array([[1,1,1,1,1,0,0,0], [5,1,0,0,0,1,0,0], [0,0,-1,-1,0,0,1,0], [0,0,1,5,0,0,0,8]])
#b = np.array([40,12,-5,50])

root = r'C:\Users\pc\Documents\MATLAB'
name = 'sc50a'
path = r'\a\a'
path = root + path.replace('a', name)

print(make_figure(root, analyze_CCGI2, 5))

[obj, b, nb] = analyze_CCGI1(path, 5)
ite = list(range(len(obj)))
plt.plot(ite, b, color = "r", label = "Basis")
plt.plot(ite, nb, color = "b", label = "NonBasis")
plt.grid(which = "both", axis="y")
plt.legend(loc="center", bbox_to_anchor=(0.5, 1.05), ncol=2) 
plt.savefig("fa")
plt.clf()
plt.plot(ite, obj, color = "k", label = "objective value")
plt.plot(ite, [obj[-1]] * len(obj), color = "y", label = "optimal value")
plt.legend(loc="center", bbox_to_anchor=(0.5, 1.05), ncol=2) 
plt.savefig("fafa")

#print(solve_alg(path, 5))
#test_result = mpssize(root)
#test_result = test_rnd(root, 5, 10)
#test_result = test_CCGI(root, 5)
#test_result = test_CC(root, 5)
#test_result = test_imp(root, 5, 0.5)
#with open('test_CCGI.csv', 'w') as f:
#    writer = csv.writer(f, lineterminator='\n')
#    writer.writerows(test_result)
#    f.close()    

#print(test_result)
#print(test_CC(root, 5))
#おそらく成功したmpsfileの一覧
#afiro blend sc50a sc50b scagr7 seba stocfor1
#sc105 adlittle
#最大係数規則メインで動かない時だけCCを使ったケースだとほとんどうまく動かない
#おそらく今回のテストケースと最大係数規則は相性が悪い