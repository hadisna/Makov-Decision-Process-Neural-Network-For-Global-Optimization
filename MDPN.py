import numpy as np
import math
import random
import matplotlib.pyplot as plt


def ChangeDomain(temp_e, umax, umin, Size, Codel, n, Po, temp_e3, temp_e2):
    v_max = math.floor((n * Codel - 1) / Codel)
    umax1 = np.array([[0.0] for i in range(v_max + 1)])
    umin1 = np.array([[0.0] for i in range(v_max + 1)])

    for j in range(0, n * Codel, Codel):
        xx = temp_e[:, j]
        xx = xx.reshape((-1, 1))
        xx = np.array(sorted(xx))
        dd = np.diff(np.concatenate([xx, [max(xx) + 1]], axis=0), axis=0)
        tmp = np.concatenate([[[1]], dd], axis=0)
        tmp3 = np.where(tmp > 0)[0]
        count = np.diff(tmp3, axis=0)
        count = count.reshape((-1, 1))
        yy = np.concatenate([xx[np.where(dd > 0)[0]], count], axis=1)
        yy = yy.T
        JJ = yy.shape
        y0 = 0.0
        y1 = 0.0
        y2 = 0.0
        y3 = 0.0
        y4 = 0.0
        for i in range(JJ[1]):
            if yy[0, i] == 0:
                y0 = yy[1, i] / Size
            elif yy[0, i] == 1:
                y1 = yy[1, i] / Size
            elif yy[0, i] == 2:
                y2 = yy[1, i] / Size
            elif yy[0, i] == 3:
                y3 = yy[1, i] / Size
            elif yy[0, i] == 4:
                y4 = yy[1, i] / Size
        v = math.floor(j / Codel)
        umax1[v] = umax[v]
        umin1[v] = umin[v]
        if y0 > Po and temp_e[0, j] == 0:
            umax[v] = (umax1[v] - umin1[v]) / 5 + umin1[v]
            umin[v] = umin1[v]
        if y1 > Po and temp_e[0, j] == 1:
            umax[v] = (umax1[v] - umin1[v]) * 2 / 5 + umin1[v]
            umin[v] = (umax1[v] - umin1[v]) / 5 + umin1[v]
        if y2 > Po and temp_e[0, j] == 2:
            umax[v] = (umax1[v] - umin1[v]) * 3 / 5 + umin1[v]
            umin[v] = (umax1[v] - umin1[v]) * 2 / 5 + umin1[v]
        if y3 > Po and temp_e[0, j] == 3:
            umax[v] = (umax1[v] - umin1[v]) * 4 / 5 + umin1[v]
            umin[v] = (umax1[v] - umin1[v]) * 3 / 5 + umin1[v]
        if y4 > Po and temp_e[0, j] == 4:
            umin[v] = (umax1[v] - umin1[v]) * 4 / 5 + umin1[v]
            umax[v] = umax1[v]
        if umax[v] != umax1[v] or umin[v] != umin1[v]:
            QBack = temp_e[:, (j + 1):(v + 1) * Codel]
            temp_e[:, j:((v + 1) * Codel - 1)] = QBack
            temp_e[:, (v + 1) * Codel - 1] = np.squeeze(np.round(4 * np.random.rand(Size, 1)), axis=(1,))
            temp_e[0, (v + 1) * Codel - 1] = 0
            TE1 = temp_e3
            QBack1 = TE1[:, (j + 1):(v + 1) * Codel]
            temp_e2[:, j:((v + 1) * Codel - 1)] = QBack1
            temp_e2[:, (v + 1) * Codel - 1] = np.squeeze(np.round(4 * np.random.rand(Size, 1)), axis=(1,))
    RR = [0 for i in range(Size)]
    for i in range(Size):
        for j in range(n * Codel):
            if temp_e[i, j] == temp_e[0, j] and j % (Codel-1) != 0:
                RR[i] = RR[i] + (math.pow(5, (Codel - ((j + 1) % Codel)))) / constant
            elif temp_e[i, j] == temp_e[0, j] and j % (Codel-1) == 0:
                RR[i] = RR[i] + 1 / constant
    for i in range(Size):
        RR[i] = RR[i] + 1 / (i + 1)

    OderRR = sorted(RR, reverse=True)
    IndexRR = sorted(range(len(RR)), key=lambda x: RR[x], reverse=True)
    TER = np.array([temp_e[IndexRR[i], :] for i in range(Size)])
    temp_e = TER.copy()
    temp_e1 = temp_e.copy()

    return temp_e, umax, umin, temp_e1, temp_e2

def ackley(fval, n):

    obj = 0.0
    obj1 = 0.0
    obj2 = 0.0
    for i in range(n):
        obj1 = obj1+fval[i]**2
        obj2 = obj2+math.cos(2*math.pi*fval[i])
    obj = -20*math.exp(-0.2*math.sqrt(obj1/n))-math.exp(obj2/n)+20+math.e

    return obj

def rosenbrock(fval, n):

    obj = 0
    for i in range(n-1):
        obj = obj+100*pow(fval[i+1]-pow((fval[i]), 2), 2)+pow((1-fval[i]), 2)

    return obj

def griewank(fval, n):

    obj = 0.0
    objt = 0.0
    objd = 1.0
    for i in range(n):
        objt = fval[i]**2.0/2.0+objt
        objd = math.cos(fval[i]/math.sqrt(i+1))*objd

    obj = objt-objd+1

    return obj

def rastrigin(fval, n):

    obj = 0.0
    for i in range(n):
        obj = obj+fval[i]**2-10*math.cos(2*math.pi*fval[i])

    obj=obj+10*n

    return obj

def evaluation(m, n, Codel, uma, umi):

    y = [0.0 for i in range(n)]
    x = [0.0 for i in range(n)]
    r = [0.0 for i in range(n)]

    for v in range(n):
        y[v] = 0.0
        mm[v] = m[[i for i in range(Codel * v, (v + 1) * Codel)]]
        for i in range(Codel):
            y[v] = y[v] + mm[v][i] * pow(5, Codel - (i + 1))
        x[v] = (uma[v] - umi[v]) * y[v] / (5 ** Codel) + umi[v]
        r[v] = x[v]
    Fi = eval(objectfunction)(r, n)
    return Fi

objectfunction = str(input("please enter the function you want\n"))
print("objectfunction is ", objectfunction)
Size = int(input("Please enter population size you want(could be empty):default 100\n"))
print("population size is ", Size)
G = int(input("Please enter iteration times you want (could be empty):default 100\n"))
print("iteration times is ", G)
Codel = int(input("Please enter String length of each variables(could be empty):default 3\n"))
print("length of each variables is ", Codel)
umaxo = int(input('please enter upper bound of variables you want(could be empty):default 10^6\n'))
print("upper bound of variables is ", umaxo)
umino = int(input('please enter lower bound of variable you want(could be empty):default -10^6\n'))
print("lower bound of variable", umino)
n = int(input("Please enter number of variables(must enter):example 2\n"))
print("number of variables is ", n)
Po = float(input("Please enter Probability measure to control the Process:example 0.5\n"))
print("Probability measure to control the Process is:", Po)

mm = [[] for i in range(n)]
mmb = [[] for i in range(n)]
bfi = [None for _ in range(G)]
BS = [None for _ in range(G)]

Q = list([])
EX = list([])
EX1 = list([])

for i in range(5):
    Q.append(np.round(4 * np.random.rand(Size, n * Codel)))
    EX.append(np.round(4 * np.random.rand(Size, n * Codel)))
    EX1.append(np.round(4 * np.random.rand(Size, n * Codel)))

umax = [umaxo for i in range(n)]
umin = [umino for i in range(n)]

constant = 0
for i in range(Codel):
    constant = constant + n * (5 ^ i)

ax = []
ay = []
plt.ion()

for k in range(G):

    F = [0 for s in range(15 * Size)]

    for i in range(Size):
        for j in range(Codel * n):
            for b in range(4):
                Q[b + 1][i, j] = (Q[b][i, j] + 1) % 5
                EX[b + 1][i, j] = (EX[b][i, j] + 1) % 5
                EX1[b + 1][i, j] = (EX1[b][i, j] + 1) % 5

    E = np.concatenate((Q, EX, EX1), axis=0)
    E = E.reshape(15 * Size, n * Codel)

    for s in range(15 * Size):
        m = E[s, :]
        F[s] = evaluation(m, n, Codel, umax, umin)

    # Ji = [1.0 / value for value in F]
    # BestJ = max(Ji)
    fi = F
    Oderfi = sorted(fi)
    Indexfi = sorted(range(len(fi)), key=lambda x: fi[x])
    Oderfi1 = sorted(fi, reverse=True)
    Indexfi1 = sorted(range(len(fi)), key=lambda x: fi[x], reverse=True)
    Bestfitness = Oderfi[0]
    TempE = np.array([list(E[Indexfi[i]]) for i in range(Size)])
    TempE2 = np.array([list(E[Indexfi1[i]]) for i in range(Size)])
    TempE3 = np.array([list(E[Indexfi[i]]) for i in range(9 * Size, 10 * Size)])
    TempE3 = np.flipud(TempE3)
    BestS = TempE[0, :]
    bfi[k] = Bestfitness
    BS[k] = BestS

    # print('Bestfitness', Bestfitness, BestS, umax, umin)

    TempE, umax, umin, TempE1, TempE2 = ChangeDomain(TempE, umax, umin, Size, Codel, n, Po, TempE3, TempE2)

    m = TempE[0, :]
    F1 = evaluation(m, n, Codel, umax, umin)

    # print('Bestfitness2', F1, m, umax, umin)
    #
    # print('TempE', TempE, TempE1, TempE2)

    for i in range(Size - 2):
        for j in range(n * Codel):
            if TempE[i, j] == TempE[i + 1, j] and TempE[i, j] != 4:
                TempE[i + 1, j] = TempE[i + 1, j] + 1
            elif TempE[i, j] == TempE[i + 1, j] and TempE[i, j] == 4:
                TempE[i + 1, j] = 0
            elif TempE[i, j] == TempE[i + 2, j] and TempE[i, j] != 4:
                TempE[i + 2, j] = TempE[i + 2, j] + 1
            elif TempE[i, j] == TempE[i + 2, j] and TempE[i, j] == 4:
                TempE[i + 2, j] = 0

            if TempE1[i, j] == TempE1[i + 1, j] and TempE1[i, j] != 0:
                TempE1[i + 1, j] = TempE1[i + 1, j] - 1
            elif TempE1[i, j] == TempE1[i + 1, j] and TempE1[i, j] == 0:
                TempE1[i + 1, j] = 4
            elif TempE1[i, j] == TempE1[i + 2, j] and TempE1[i, j] != 0:
                TempE1[i + 2, j] = TempE1[i + 2, j] - 1
            elif TempE1[i, j] == TempE1[i + 2, j] and TempE1[i, j] == 0:
                TempE1[i + 2, j] = 4

            if TempE2[i, j] == TempE2[i + 1, j] and TempE2[i, j] != 4:
                TempE2[i + 1, j] = TempE2[i + 1, j] + 1
            elif TempE2[i, j] == TempE2[i + 1, j] and TempE2[i, j] == 4:
                TempE2[i + 1, j] = 0
            elif TempE2[i, j] == TempE2[i + 2, j] and TempE2[i, j] != 4:
                TempE2[i + 2, j] = TempE2[i + 2, j] + 1
            elif TempE2[i, j] == TempE2[i + 2, j] and TempE2[i, j] == 4:
                TempE2[i + 2, j] = 0

    # print('TempE2', TempE, TempE1, TempE2)

    for i in range(Size - 1):
        for j in range(n * Codel):
            if TempE2[0, j] == TempE[i+1, j]:
                TempE[i+1, j] == TempE[0, j]
            if TempE2[0, j] == TempE1[i+1, j]:
                TempE1[i + 1, j] = TempE1[0, j]
            if TempE[0, j] == TempE2[i + 1, j]:
                TempE2[i + 1, j] = TempE2[0, j]

    TempE1[0, :] = np.round(4 * np.random.rand(1, n * Codel))

    for i in range(Size):
        for j in range(Codel * n):
            Q[0][i, j] = TempE[i, j]
            EX[0][i, j] = TempE1[i, j]
            EX1[0][i, j] = TempE2[i, j]

    yb = [0 for i in range(n)]
    variables = [0 for i in range(n)]

    for v in range(n):
        yb[v] = 0
        mmb[v] = m[[i for i in range(Codel * v, (v + 1) * Codel)]]
        for i in range(Codel):
            yb[v] = yb[v] + mm[v][i] * pow(5, Codel - (i + 1))
        variables[v] = (umax[v] - umin[v]) * yb[v] / (5 ** Codel) + umin[v]
    Fist = eval(objectfunction)(variables, n)
    print('Bestfitness3', k, Fist, variables, m, umax, umin)
    ax.append(k)
    ay.append(Fist)
    plt.clf()
    if len(ax) > 20:
        plt.plot(ax[-20:-1], ay[-20:-1])
    else:
        plt.plot(ax, ay)
    plt.pause(0.1)
    plt.ioff()





