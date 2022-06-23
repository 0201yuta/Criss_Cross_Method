import numpy as np
from numpy import linalg as LA

#CCPivotを行う関数、入力はsimplex tableauとb,n
#bを非基底に、nを基底変数にpivotしてtableauを出力
#b,nはx_b,x_nがindexの何番目に格納されているかを表している
def Pivot(tableau, b, n): 
    bn = tableau[b + 1][n + 1]
    v1 = tableau[:, n + 1] / bn
    v2 = tableau[b + 1] / bn
    v2[n + 1] = 1 / bn
    tableau = tableau - np.outer(v1, tableau[b + 1])
    tableau[:, n + 1] = -v1
    tableau[b + 1] = v2
    print(tableau)
    return tableau

def CCMethod(tableau): #algorithm1を実装したCrissCrossMethod
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < 0)[0] + col
    J = np.where(tableau[0, 1:] < 0)[0]
    Index = np.array(list(range(0, row + col)))
    while I.size + J.size != 0:
        print(Index)
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
                return "no feasible solution"    
        else: #step4を実行 
            k = np.where(Index == k_J)[0][0]
            T = np.where(tableau[1:, k + 1] > 0)[0] + col
            if T.size != 0:
                i = np.where(Index == np.amin(np.array([Index[x] for x in T])))[0][0]
                tableau = Pivot(tableau, i - col, k)
                Index[k], Index[i] = Index[i], Index[k]
            else:
                return "no feasible solution"                        
        I = np.where(tableau[1:, 0] < 0)[0] + col 
        J = np.where(tableau[0, 1:] < 0)[0]  
    return tableau   


def NewCCMethod(tableau): #algorithm2を実装したCrissCrossMethod、index選択においてalphaを導入すればよい
    row, col = tableau.shape
    row = row - 1
    col = col - 1
    I = np.where(tableau[1:, 0] < 0)[0] + col
    J = np.where(tableau[0, 1:] < 0)[0]
    Index = np.array(list(range(0, row + col)))
    while I.size + J.size != 0:
        print(Index)
        b = tableau[1:, 0]
        tab_norm = np.array([np.linalg.norm(tableau[1:, x]) for x in range(1, col + 1)])
        alpha = np.r_[np.divide(np.array(np.dot(b, tableau[1:, 1:])), tab_norm), b] #添え字ではなく格納順
        candidate = np.where(alpha == np.amin(np.array([alpha[x] for x in np.r_[J, I]])))[0]
        k = np.where(Index == np.amin(np.array([Index[x] for x in candidate])))[0][0]
        print(alpha)
        if k >= col: #step3を実行
            k = k - col #k_Iが何行目に格納されているかを表す式
            S = np.where(tableau[k + 1, 1:] < 0)[0] 
            if S.size != 0:
                candidate = np.where(alpha == np.amin(np.array([alpha[x] for x in S])))[0]
                j = np.where(Index == np.amin(np.array([Index[x] for x in candidate])))[0][0]
                tableau = Pivot(tableau, k, j)
                Index[k + col], Index[j] = Index[j], Index[k + col]
                alpha[k + col], alpha[j] = alpha[j], alpha[k + col]
            else:
                return "no feasible solution"    
        else: #step4を実行 
            T = np.where(tableau[1:, k + 1] > 0)[0] + col
            if T.size != 0:
                candidate = np.where(alpha == np.amin(np.array([alpha[x] for x in T])))[0]
                i = np.where(Index == np.amin(np.array([Index[x] for x in candidate])))[0][0]
                tableau = Pivot(tableau, i - col, k)
                Index[k], Index[i] = Index[i], Index[k]
                alpha[k], alpha[i] = alpha[i], alpha[k]
            else:
                return "no feasible solution"                        
        I = np.where(tableau[1:, 0] < 0)[0] + col 
        J = np.where(tableau[0, 1:] < 0)[0]
    return tableau


sample1 = np.array([[0, -3, 4], [-2, -1, -2], [-4, -3, -1], [1, 1, -1], [3, 1, 1]])
sample2 = np.array([[0, -1, -1, -1], [11, 5, 4, 1], [15, 2, -1, 3], [6, -1, 2, -1]])
#print(CCMethod(sample1))
print(NewCCMethod(sample1))
