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

def CCMethod(tableau, error): #algorithm1を実装したCrissCrossMethod
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < - np.power(10., -error))[0] + col
    J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
    Index = np.array(list(range(0, row + col)))
    Index = np.hstack([Index[-row:],Index[:-row]])
    count = 0
    while I.size + J.size != 0 and count < 10000:
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
    if count < 10000:
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
    while I.size + J.size != 0 and count < 10000:
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
                return "no feasible solution", tableau, Basis, count 
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
                return "no feasible solution", tableau, Basis, count
        I = np.where(tableau[1:, 0] < - np.power(10., -error))[0]
        J = np.where(tableau[0, 1:] < - np.power(10., -error))[0]
        count = count + 1  
    if count < 10000:        
        return "optimal solution", tableau, Basis, count
    else:    
        return "no feasible solution", tableau, Basis, count
 
def solve_mps(path, error):
    tableau = np.loadtxt(path + r'.txt', delimiter = ',')
    Ab = np.loadtxt(path + r'_Ab.txt', delimiter = ',')
    opt = np.loadtxt(path + r'_opt.txt', delimiter = ',')[0]
    ans_new = Practical(tableau, Ab[:, 0:-1], Ab[:, -1], error)
    ans_old = CCMethod(tableau, error)
    return ans_new[0], ans_new[1][0][0], ans_new[3], ans_old[0], ans_old[1][0][0], ans_old[2], opt

def test_CC(root, error):   
    error_table = []
    files = os.listdir(root)
    dirlist = [dir for dir in files if os.path.isdir(os.path.join(root, dir))]
    for name in dirlist:
        P = r'\a\a'
        path = root + P.replace('a', name)
        sol = solve_mps(path, error)
        e_new = np.abs(sol[1] - sol[6]) / sol[6] <= np.power(10., -4)
        e_old = np.abs(sol[4] - sol[6]) / sol[6] <= np.power(10., -4)
        T = [name, e_new, e_old]
        error_table.append(T)
    return error_table    


#sample = np.array([[191.5, 5.5, -0.5, 4.5, 0], [-5.75, -0.25, 0.25, -0.25, 0], [40.75, 1.25, -0.25, 1.25, 0], [-6.25, 0, 0, -1.25, -0.25], [11.25, 0, 0, 0.25, 0.25]])
#A = np.array([[1,1,1,1,1,0,0,0], [5,1,0,0,0,1,0,0], [0,0,-1,-1,0,0,1,0], [0,0,1,5,0,0,0,8]])
#b = np.array([40,12,-5,50])

root = r'C:\Users\pc\Documents\MATLAB'
name = 'afiro'
path = r'\a\a'
path = root + path.replace('a', name)
#print(solve_mps(path, 6))

test_result = test_CC(root, 8)
with open('test.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(test_result)
    f.close()    

print(test_result)




