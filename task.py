from cProfile import label
from queue import Queue
from random import random
from threading import Thread, Lock
import numpy as np
import pandas as pd
from sympy import false
import matplotlib.pyplot as plt


def generateCoefficient():
    V = 0
    n = 12
    for _ in range(0, n):
        V += random()
    mV = n/2
    sigmaV = np.sqrt(n/12)
    z = (V - mV)/sigmaV
    return z


def calculateNet(L, t, a, randCoefficient, h, taw):
    if taw < h**2 / (2 * a) == false:
        raise ValueError('Сетка с данными параметрами неусточива')
    n = int(t/taw) + 1
    m = int(L/h) + 1

    U = np.zeros((n, m))

    # Краевые условия
    for i in range(1, n):
        U[i, 0] = randCoefficient * np.cos(i * taw)
        U[i, -1] = 0

    # Начальное условие
    for i in range(1, m):
        U[0, i] = 0

    for j in range(1, n):
        for i in range(1, m - 1):
            U[j, i] = (a * taw/h**2) * (U[j - 1][i - 1] - 2 *
                                        U[j - 1][i] + U[j - 1][i + 1]) + U[j - 1][i]

    U = pd.DataFrame(U)
    return U


def checkBorders(border: float, L, t, a, randCoefficient, h, taw):
    global netQueue
    U = calculateNet(L, t, a, randCoefficient, h, taw)
    netQueue.put(U)
    tau, x = U.shape
    for i in range(0, tau):
        for j in range(0, x):
            if np.abs(U[j][i]) >= border:
                # print(U[j][i])
                return
    mutex.acquire()
    global k
    k += 1
    mutex.release()


def printOneGraph(queueNet, border, h):
    axis = np.arange(0, 1.1, h)
    topBorder = []
    bottomBorder = []
    for _ in axis:
        topBorder.append(border)
        bottomBorder.append(-border)
    U = queueNet.get()
    tau, x = U.shape
    for i in range(0, tau):
        if i == 1 or i == 2 or i == 5 or i == 10 or i == 20 or i == 50 or i == 100 or i == 200 or i == 500 or i == 1000:
            res = np.array(U.iloc[[i]])
            res = np.concatenate(res)
            plt.plot(axis, res, label=f't={i * 0.001}')
    plt.legend(loc=1)
    plt.xlabel('Значение координаты X')
    plt.ylabel('Значение функции U')
    plt.plot(axis, topBorder, axis, bottomBorder)
    plt.show()


def printGraph(queueNet, border, h):
    axis = np.arange(0, 1.1, h)
    topBorder = []
    bottomBorder = []
    for _ in axis:
        topBorder.append(border)
        bottomBorder.append(-border)
    n = 0
    while not queueNet.empty():
        n += 1
        U = queueNet.get()
        tau, x = U.shape
        for i in range(0, tau):
            res = np.array(U.iloc[[i]])
            res = np.concatenate(res)
            plt.plot(axis, res)
    plt.xlabel('Значение координаты X')
    plt.ylabel('Значение функции U')
    plt.plot(axis, topBorder, axis, bottomBorder)
    plt.show()


k = 0
mutex = Lock()
netQueue = Queue()


def startExperiment(iterations, startingBorder, borderStep, desiredProbability, L, t, a, h, taw):
    global k, netQueue
    threadQueue = []
    border = startingBorder
    borderIterations = 0
    while k/iterations < desiredProbability:
        netQueue.queue.clear()
        k = 0
        for i in range(0, iterations + 1):
            th = Thread(target=checkBorders, args=(
                border, L, t, a, generateCoefficient(), h, taw))
            threadQueue.append(th)
            th.start()
        for i in threadQueue:
            i.join()
        print(
            f'Полученные результаты вероятность не выхода за границу: {k/iterations} \n Желаемая вероятность: {desiredProbability} \n Установленная граница: {border} \n Вероятность достигнута? {"Да" if k/iterations >= desiredProbability else "НЕТ"}')
        border += borderStep
        borderIterations += 1
    border -= borderStep  # лишний на выходе из алгоритма
    print(
        f'Граница с вероятностью невыхода {desiredProbability * 100}% {border} \nПотребовалось итераций {borderIterations} со стартовой границы {startingBorder} и шагом в {borderStep}')
    # printOneGraph(netQueue, border, h)
    printGraph(netQueue, border, h)
