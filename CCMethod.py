import numpy as np
from numpy import linalg as LA
import os.path
import csv

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
                return "no feasible solution", tableau, count    
        else: #step4を実行 
            k = np.where(Index == k_J)[0][0]
            T = np.where(tableau[1:, k + 1] > np.power(10., -error))[0] + col
            if T.size != 0:
                i = np.where(Index == np.amin(np.array([Index[x] for x in T])))[0][0]
                tableau = Pivot(tableau, i - col, k)
                Index[k], Index[i] = Index[i], Index[k]
            else:
                return "no feasible solution", tableau, count                       
        I = np.where(tableau[1:, 0] < 0)[0] + col 
        J = np.where(tableau[0, 1:] < 0)[0]  
        count = count + 1
    if count < 5000:
        return "optimal solution", tableau, count    
    else:
        return "no feasible solution", tableau, count

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
                return "no feasible solution", tableau, count 
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
                return "no feasible solution", tableau, count
        I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]
        J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
        count = count + 1  
    if count < 5000:        
        return "optimal solution", tableau, count
    else:    
        return "no feasible solution", tableau, count

#最大係数規則とalgorithm2をランダムで選んで実行する新しいalgorithm
#今回は目的関数がminなので目的関数の非基底変数の係数が最も小さいものを選ぶ
#終了条件はCrissCrossMethodのものをそのまま使う
def random_method(tableau, A, b, error, rule):
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
    while I.size + J.size != 0 and count < 5000:
        if np.random.rand() < 0.5 or J.size == 0: #CCpivotを実行
            IJ_Index = np.union1d(np.array([Basis[x] for x in I]), np.array([NonBasis[x] for x in J]))
            IJ_Index = IJ_Index.astype(np.int64)
            min_alpha = np.amin(np.array([alpha[x] for x in IJ_Index]))
            k = np.amin(np.intersect1d(np.where(alpha == min_alpha)[0], IJ_Index))
            cc_count = cc_count + 1
        elif rule <= 1: #αの優先規則か最小添字規則への分岐
            k = NonBasis[np.argmin(tableau[0, 1:])]        
        #else: #最良改善規則への分岐
        #    k = 
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
                #else: #最良改善規則  
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
        #まず非基底変数からkを最大係数規則や最良改善規則で求める
        #次に基底変数側からiを最小添え字、αを用いた優先規則のいずれかで求める
        #最後にk,iに対してpivot関数を使用

#最良改善規則とalgorithm2をランダムで実行する新しいalgorithm
#今回は目的関数がminなのでb/aの値が最も小さいものを選ぶ
#def random_method2(tableau, A, b, error):

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
 
def solve_alg(path, error):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_alg1 = CCMethod(tableau, error)
    ans_alg2 = Practical(tableau, Ab[:, 0:-1], Ab[:, -1], error)
    e_old = np.abs((-ans_alg1[1][0][0] - opt) / opt) <= np.power(10., -3)
    e_new = np.abs((-ans_alg2[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_old, ans_alg1[2], e_new, ans_alg2[2]]

def solve_imp(path, error, rule, balance):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_rnd = random_method(tableau, Ab[:, 0:-1], Ab[:, -1], error, rule)
    ans_GI = CCandGI(tableau, Ab[:, 0:-1], Ab[:, -1], error, balance)
    e_rnd = np.abs((-ans_rnd[1][0][0] - opt) / opt) <= np.power(10., -3)
    e_GI = np.abs((-ans_GI[1][0][0] - opt) / opt) <= np.power(10., -3)
    return [e_rnd, ans_rnd[2], ans_rnd[3], e_GI, ans_GI[2]]

def test_CC(root, error):   
    result_table = []
    files = os.listdir(root)
    dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        result = [name]
        result.extend(solve_alg(path, error))
        result_table.append(result)
    return result_table   

def makedata(root):   
    error_table = []
    files = os.listdir(root)
    dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
        T = [name, opt]
        error_table.append(T)
    return error_table       


#sample = np.array([[191.5, 5.5, -0.5, 4.5, 0], [-5.75, -0.25, 0.25, -0.25, 0], [40.75, 1.25, -0.25, 1.25, 0], [-6.25, 0, 0, -1.25, -0.25], [11.25, 0, 0, 0.25, 0.25]])
#A = np.array([[1,1,1,1,1,0,0,0], [5,1,0,0,0,1,0,0], [0,0,-1,-1,0,0,1,0], [0,0,1,5,0,0,0,8]])
#b = np.array([40,12,-5,50])

root = r'C:\Users\pc\Documents\MATLAB'
name = 'afiro'
path = r'\a\a'
path = root + path.replace('a', name)

test_result = test_CC(root, 5)
with open('test_alg.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(test_result)
    f.close()    

print(test_result)

tableau = np.loadtxt(path + r'.txt', delimiter = ',')
Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
#ans0 = random_method(tableau, Ab[:, 0:-1], Ab[:, -1], 4, 0)
#ans1 = random_method(tableau, Ab[:, 0:-1], Ab[:, -1], 4, 1)
#ans_GI = CCandGI(tableau, Ab[:, 0:-1], Ab[:, -1], 4, 1)
#print(ans0[1][0][0], ans0[3], ans0[4])
#print(ans1[1][0][0], ans1[3], ans1[4])
#print(ans_GI[1][0][0], ans_GI[3])
#print(solve_mps(path, 4))


sample1 = np.array([[0,-4,-3], [6,2,3], [3,-3,2], [5,0,2], [4,2,1]])
A1 = np.array([[2,3,1,0,0,0], [-3,2,0,1,0,0], [0,2,0,0,1,0], [2,1,0,0,0,1]])
b1 = np.array([6,3,5,4])

sample2 = np.array([[0,-3,-2],[72,4,1],[48,2,2],[48,1,3]])
A2 = np.array([[4,1,1,0,0],[2,2,0,1,0],[1,3,0,0,1]])
b2 = np.array([72,48,48])

sample3 = np.array([[750,-17,29],[-12,2,1],[30,-5,10],[76,4,2]])
A3 = np.array([[2,-1,4,1,0],[1,2,2,0,1],[-2,0,1,0,0]])
b3 = np.array([10,8,4])

#print(Practical(sample1, A1, b1, 4))
#print(Practical(sample2, A2, b2, 4))
#print(Practical(sample3, A3, b3, 4))
