# -*- coding:utf-8 -*-
__author__ = 'Li Peng-cheng'
__date__ = '2022-07-11'
'''
The script is set for supplying some tool function.
'''
import os
import time
import pickle
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import argrelextrema
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from math import log
import copy
import networkx as nx
# import matplotx
import matplotlib as mpl
from math import log2

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def FiguresDisply(data, dataName=None, Savefigure = False, figureName='Figure'):

    # with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
    plt.figure(figsize=(10, 2.5))
    # plt.tick_params(labelsize=20)
    # plt.locator_params('y', nbins=6, size=20)
    plt.ylabel("{}".format('Forecasting VS True'), fontsize=18)
    plt.xlabel("Point", fontsize=18)

    plt.yticks(size=18)

    Fnumbers = 1
    lineStype = ['-','-.','--',':']
    # for yy, label in zip(data, dataName):
    #     plt.plot(np.arange(1, len(data[0]) + 1, 1), yy, label=label,alpha=0.7)

    for i in range(len(data)):

        if dataName:
            plt.plot(np.arange(1, len(data[i]) + 1, 1), data[i], label='{}'.format(dataName[i]),linestyle=lineStype[i])
        else:
            plt.plot(np.arange(1, len(data[i]) + 1, 1), data[i],label='Data {}'.format(Fnumbers),linestyle=lineStype[i])
    #                 Fnumbers+=1
    # matplotx.ylabel_top("voltage [V]")  # move ylabel to the top, rotate
    # matplotx.line_labels(fontproperties = 'Times New Roman',fontsize=20)  # line labels to the right
    xticks = np.arange(1, len(data[0]) + 1, int(len(data[0])/8))
    # if abs(xticks[-1]-len(data[0]))>10:
    #     xticks = np.append(xticks,len(data[0]))
    if xticks[-1] !=len(data[0]) :
        xticks[-1] = len(data[0]) #np.append(xticks,len(data[0]))
    if abs(xticks[-1]-xticks[-2])>1.5*abs(xticks[-2]-xticks[-3])-1:
        xticks = np.insert(xticks, -1, int((xticks[-2]+xticks[-1])/2))
    plt.xticks(xticks, fontproperties='Times New Roman', size=18)
    plt.xlim(1, len(data[0]) + 1)
    plt.legend(fontsize=12)
    if Savefigure:
        plt.savefig('./figures/{}-Results.png'.format(figureName),dpi=600,bbox_inches='tight')
    plt.show()

def FiguresDisply_Comparison(data, dataName=None, Savefigure = False, figureName='Figure'):
    # mpl.rcParams['font.family'] = 'Times New Roman'
    # mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        plt.figure(figsize=(10, 5))
        # plt.tick_params(labelsize=20)
        plt.locator_params('y', nbins=6)
        plt.ylabel("{}".format('Forecasting VS True'), fontsize=20)
        plt.xlabel("Point", fontsize=20)
        plt.yticks(fontproperties='Times New Roman', size=20)
        Fnumbers = 1
        initionalP = len(data[2]) - len(data[0])
        lastP = initionalP + len(data[0])
        # for yy, label in zip(data, dataName):
        plt.plot(np.arange(1, len(data[2]) + 1, 1), data[2], label=dataName[2],linestyle='-')
        plt.plot(np.arange(initionalP, lastP, 1), data[0], label=dataName[0],linestyle='-.',alpha=0.5)
        plt.plot(np.arange(initionalP, lastP, 1), data[1], label=dataName[1],linestyle='--',alpha=0.5)

        xticks = np.arange(1, len(data[-1]) + 1, int(len(data[-1])/8))
        if xticks[-1] !=len(data[-1]) :
            xticks[-1] = len(data[-1]) #np.append(xticks,len(data[0]))
        if abs(xticks[-1]-xticks[-2])>1.5*abs(xticks[-2]-xticks[-3])-1:
            xticks = np.insert(xticks, -1, int((xticks[-2]+xticks[-1])/2))
        plt.xticks(xticks, size=20)
        plt.xlim(1, len(data[2]) + 1)
        plt.axvline(x=initionalP, color='#C00000')
        plt.ylim(min(data[2]), max(data[2]))
        plt.legend(fontsize=15)
        if Savefigure:
            plt.savefig('./figures/{}_AllTrue.png'.format(figureName),dpi=600,bbox_inches='tight')
        plt.show()

def MMEM(data1, data2):

    tr = []

    for i in range(len(data1)):

        if (data1[i] + data2[i]) > 0:

            tr.append(np.abs(data1[i] - data2[i]) / (data1[i] + data2[i]))

        else:

            tr.append(0)

    return np.mean(tr)  # *((np.max(tr)-np.min(tr)))#(np.max(tr)-np.min(tr))/np.std(tr)

def GnerateGF(edgs,name):

    G = lanl_graph(edgs)
        # use graphviz to find radial layout
    plt.figure(figsize=(2.5, 2.5))
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", root=0)
        # draw nodes, coloring by rtt ping time
    options = {"with_labels": False, "alpha": 0.5, "node_size": 15}
    nx.draw(G, pos, node_color=[10*v for v in range(len(G))], **options)
        # adjust the plot limits
    xmax = 1.02 * max(xx for xx, yy in pos.values())
    ymax = 1.02 * max(yy for xx, yy in pos.values())
    plt.title("FT Construction",size=16)
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.savefig('./figures/{}-Construction.png'.format(name),dpi=600,bbox_inches='tight')
    plt.show()

def lanl_graph(edgs):
    """Return the lanl internet view graph from lanl.edges"""

    G = nx.Graph()

    time = {}
    time[0] = 0  # assign 0 to center node
    for i in range(len(edgs)):
        head=edgs[i][0]
        tail = edgs[i][1]
        rtt = edgs[i][2]
        G.add_edge(int(head), int(tail))
        time[int(head)] = float(rtt)

    # get largest component and assign ping times to G0time dictionary
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    G0 = G.subgraph(Gcc)
    G0.rtt = {}
#     print(time)
    for n in G0:
        try:
            G0.rtt[n] = time[n]
        except:
            G0.rtt[n] = 1

    return G0

def calculatesimilarseries(data1, data2):

    tr = []

    for i in range(len(data1)):
        tr.append(data1[i] - data2[i])

    return tr


def SimilarError(d1, d2):

    R = []

    for k in range(len(d1)):
        R.append((d1[k] - d2[k]))

    return np.mean(R)


def te(data1, data2):

    a = []
    b = []
    M2 = np.mean(data2)
    M1 = np.mean(data1)
    M3 = (M2 + M1) / 2

    for k in range(len(data1)):

        ta = np.square(data1[k] - data2[k])
        pp = (data1[k] + data2[k]) / 2
        tb1 = np.square(pp - M3)
        a.append(ta)
        b.append(tb1)  # (tb1+tb2)/2

    return np.sum(a) / np.sum(b)  # 1 - np.exp(np.mean(results))

def calculateMBs(mb, targeD, HDD):

    r2s = []
    # TRS = calculatesimilarseries(targeD, HDD)

    for k in mb:

        # MRS = calculatesimilarseries(mb[k], HDD)
        c2 = Tfunction(targeD, k)
        res = c2
        r2s.append(res)

    return np.mean(r2s)

def meragePredictData_WEIGHT(data1, data2, weight_ALL):

    new = []
    # print('weight_ALL',weight_ALL)
    for i in range(len(data1)):

        new.append(((weight_ALL) * data1[i]) + ((1 - weight_ALL) * data2[i]))

    return new  # new

def MaxStdFV2(data):

    alls_P = []
    alls_N = []
    alls_P.append(data[0])  # juump81

    for i in range(len(data)):

        r2s = Tfunction(data[0], data[i])

        if r2s <= 0.6 and r2s >= 0.3:

            alls_P.append(data[i])

        elif r2s >= 0.6 and r2s <= 0.9:

            alls_N.append(data[i])

    return alls_P, alls_N

def Tfunction(x, y):

    r2s = te(x, y) 

    return 1 - (1 / (1 + r2s))

def meragePredictData_MEAN(data):

    OneData = copy.deepcopy(data[0])

    for k in range(1, len(data)):

        for i in range(len(OneData)):
            OneData[i] = (OneData[i] + data[k][i]) / 2

    return OneData

def mergeSameSeries(data, stand, FS):

    FFV_F, PFV_F, ME_F = clusterFV(data, list(np.random.random(len(data))), stand, FS)
    # print('mergeFV',len(FFV_F))
    meragneFVALLF = []
    meragneLens = []

    for k in FFV_F:

        if len(k) > 0:
            meragneLens.append(len(k))
            merageFV = meragePredictData_FF(k)
            meragneFVALLF.append(merageFV)
            # print(merageFV[0])

    return meragneFVALLF, meragneLens

def mergeSameSeriesDE(data, stand, FS, Des):
    # print('OringalLengh',len(data))
    FFV_F, PFV_F, ME_F = clusterFV(data, list(np.random.random(len(data))), stand, FS)
    # print('mergeFV',len(FFV_F))
    meragneFVALLF = []
    meragneLens = []

    for k in FFV_F:

        if len(k) > 0:
            # merageFV = meragePredictData_1(k)
            # meragneFVALLF.append(merageFV)
            meragneFVALLF.append(k[0])
            # print(merageFV[0])
            meragneLens.append(len(k))

    return meragneFVALLF, PFV_F

def meragePredictData_FF(data):

    if len(data) > 1:

        lenge = []
        pdata = copy.deepcopy(data)

        for k in data:
            lenge.append(len(k))

        maxindex = lenge.index(min(lenge))
        maxdata = copy.deepcopy(pdata[maxindex])

        del lenge[maxindex]
        del pdata[maxindex]

        for i in range(len(pdata)):

            currentLen = len(pdata[i][:len(maxdata)])

            for c in range(currentLen):
                pd = sigmoid1(pdata[i][c], maxdata[c])
                # print(pd)

                maxdata[c] = pd
                # print(1)
            # print(lenge)
            # maxdata = [maxdata]

        return maxdata

    elif len(data) == 1:

        return data[0]

    elif len(data) == 0:

        print('The number of data must be greater than 2')


def sigmoid1(data_1, data_2):

    m = (data_1 + data_2) / 2
    re = data_1 * data_1 + data_1 * data_2 + data_2 + data_2 * data_2 + data_1
    wCos = np.square(np.cos((re)))
    wSin = np.square(np.sin((re)))
    scos = ((wCos * data_1)) + ((1 - wCos) * (data_2))
    ssin = ((wSin * data_2)) + ((1 - wSin) * (data_1))
    finalR = (scos + ssin) / 2

    if np.isnan(finalR):
        print(scos, ssin, wCos, wSin, data_1, data_2)

    return (scos + ssin + m) / 3

def clusterFV(FVdata, PVdata, M, Stand):

    sameForecastingValue = []
    samePV = []
    AllforecastingValuesCopy = copy.copy(FVdata)
    AllPVdata = copy.copy(PVdata)
    #     print(len(AllforecastingValuesCopy),len(AllPVdata))
    allR2 = []

    while len(AllforecastingValuesCopy) > 0:

        currentFV = AllforecastingValuesCopy[0]
        currentPV = AllPVdata[0]
        Tarray = [currentFV]
        TarrayPV = [currentPV]
        TpositionDelete = [0]
        R2set = []

        for n in range(1,len(AllforecastingValuesCopy)):
            a = currentFV[:Stand]
            b = AllforecastingValuesCopy[n][:Stand]
            # mmem = MMEM(a,b)
            # bf = abs(a[int(Stand / 2)] - (b[int(Stand / 2)]))
            # ff = abs(a[-1] - (b[-1]))
            # R2 = 0.7*(Tfunction(a, b)+0.3*mmem) #- log(1 + bf + ff)
            R2 = Tfunction(a, b) #0.7*(Tfunction(a, b)+0.3*mmem) #- log(1 + bf + ff)

            R2set.append(R2)

            if R2 <= M:  # if mse < 0.01 and mse>0:mse > 0 and mse <= 0.05

                Tarray.append(AllforecastingValuesCopy[n])
                TarrayPV.append(AllPVdata[n])
                TpositionDelete.append(n)

        sameForecastingValue.append(Tarray)
        samePV.append(np.mean(TarrayPV))
        allR2.append(np.mean(R2set))

        for k in TarrayPV:
            #print('k',k,AllPVdata.index(k),AllPVdata[AllPVdata.index(k)])
            del AllforecastingValuesCopy[AllPVdata.index(k)]
            del AllPVdata[AllPVdata.index(k)]

    return sameForecastingValue, samePV, allR2

def MaxMinF(data):

    cc = MinMaxScaler(feature_range=(0, 1))
    MMdata = cc.fit_transform(np.array(data).reshape(-1, 1))

    return MMdata

def separateFunction(data, distanceMaxMin):

        maxpoint = argrelextrema(data, np.greater)[0].tolist()
        minpoint = argrelextrema(data, np.less)[0].tolist()

        finalseparatePoints = maxpoint + minpoint
        finalseparatePoints.insert(0, 0)
        finalseparatePoints = sorted(finalseparatePoints)

        if finalseparatePoints[-1] != len(data) - 1:
            finalseparatePoints.append(len(data) - 1)

        State = np.ones(len(finalseparatePoints))

        nes = 0
        for k in range(1, len(finalseparatePoints)):

            if finalseparatePoints[k] - nes >= distanceMaxMin:
                State[k] = 0
                nes = finalseparatePoints[k]

        Tfin = []

        for n in range(len(State)):

            if State[n] == 0:
                Tfin.append(finalseparatePoints[n])

        finalseparatePoints = Tfin
        finalseparatePoints.insert(0, 0)

        if finalseparatePoints[-1] != len(data) - 1:
            finalseparatePoints.append(len(data) - 1)

        return finalseparatePoints

# def SoomtingData(so_Variable):

#     sommting_oriagn = so_Variable
#     soomint_over_1 = so_Variable
#     soomting_over_difference = []

#     for c in range(len(sommting_oriagn)):

#         so_f_f_f = []

#         for a in range(len(sommting_oriagn)):
#             so_f_f_f.append(sommting_oriagn[c] - sommting_oriagn[a])

#         soomting_over_difference.append(np.sum(so_f_f_f))

#     for h in range(600):

#         for k in range(len(sommting_oriagn) - 4):

#             m = np.mean(sommting_oriagn[k + 1:k + 4])
#             fluctuation = []

#             for g in range(3):

#                 fluctuation.append(sommting_oriagn[g + k + 2] - sommting_oriagn[g + k + 1])

#             fluctuation_over = np.round_(np.sqrt(np.square((np.sum(fluctuation)))), 7)
#             soomint_over_1[k + 1] = sommting_oriagn[k + 1] - ((fluctuation_over) * (sommting_oriagn[k + 1] - m))

#     fig = plt.figure(figsize=(20, 5))
#     ax = fig.add_subplot(222)
#     plt.ylabel("Smoothed Data", fontsize=8)
#     plt.locator_params('y', nbins=10)
#     plt.plot(range(len(soomint_over_1)), soomint_over_1, color="#009969", label='Data', alpha=0.7)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_color("#e1e1e1")
#     ax.spines['left'].set_color("#e1e1e1")
#     plt.tick_params(labelsize=6, color="none")
#     plt.show()

#     return soomint_over_1
def SoomtingData(so_Variable):
    
    sommting_oriagn =  so_Variable
    soomint_over_1 =  so_Variable
    soomting_over = []
    soomting_over_Std = []
    soomting_over_difference = []
    Baseline = 3

    for c in range(len(sommting_oriagn)):
        
        so_f_f_f = []
        
        for a in range(len(sommting_oriagn)):
            
            so_f_f_f.append(sommting_oriagn[c]-sommting_oriagn[a])
#             print(so_f_f_f)
        soomting_over_difference.append(np.sum(so_f_f_f))
    
    for h in range(600):
        
        for k in range(len(sommting_oriagn)-Baseline):
            
            # s = np.std(so_Variable[k+1:k+4])
            m = np.mean(sommting_oriagn[k:k+Baseline])
            # j = (0.5*np.sin(m))+(0.1*np.cos(m))
            # c = np.mean(j)
            fluctuation = []
            
            for g in range(Baseline-1):
                
                fluctuation.append(sommting_oriagn[g+k+2]-sommting_oriagn[g+k+1])

#             f = (so_Variable[k+1] - so_Variable[k+2]) + (so_Variable[k+2] - so_Variable[k+3])+ (so_Variable[k+3] - so_Variable[k+4])

            fluctuation_over = np.round_(np.sqrt(np.square((np.sum(fluctuation)))),7)
#             fluctuation_over = np.round_(np.sum(np.sqrt(np.square(fluctuation))),7)
            soomint_over_1[k+1] = sommting_oriagn[k+1] - ((fluctuation_over)*(sommting_oriagn[k+1] - m))
#             soomint_over_1[k+1] = so_Variable[k+1] - ((so_Variable[k+1] - m))
    
    fig = plt.figure() #figsize=(20, 5)
    ax = fig.add_subplot(222)
    plt.ylabel("Wind Power(W/S)", fontsize = 8)

    plt.locator_params('y', nbins = 10)
#     plt.plot(range(len(so_Variable)),so_Variable,':',color = "#ff4864",label = 'Actural WP',)
    plt.plot(range(len(soomint_over_1)),soomint_over_1,color = "#009969", label = 'Forecasting WP by LSTM',alpha = 0.7)    
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color("#e1e1e1")
    ax.spines['left'].set_color("#e1e1e1")

    # plt.legend(loc='upper right',fontsize = 6,framealpha = 1)
    # plt.xlim(0,1000)
    plt.tick_params(labelsize=6,color = "none")
    # plt.xlim(100,150)
#     plt.savefig('images_out/50_soomting.pdf', dpi=600,bbox_inches='tight') 
    plt.show
    

    return soomint_over_1

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def DisplayseparationSeries(data, separationP, figurename='None'):

    datas = data
    length = len(datas)
    x = np.linspace(1, length, length)
    y = datas.reshape(1, length)[0]
    dydx = x  # first derivative

    print(len(x), len(y))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    # fig, axs = plt.subplots(1,1)

    fig = plt.figure(figsize=(8, 2.5))  # figsize=(20, 10)
    ax = fig.add_subplot(111)

    linerColor = []
    for a in range(len(separationP)):
        linerColor.append(randomcolor())

    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):

        cmap = ListedColormap(linerColor)
        norm = BoundaryNorm(separationP, cmap.N)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line)
        plt.title("Split Data",size=16)
        ax.set_xlim(x.min(), x.max())
        # ax.set_xlim(x.max() - 200, x.max())
        ax.set_ylim(0, 1)
        # ax.plot(marker='o', label='Wind Speed')
        plt.savefig('./figures/{}-SplitData.png'.format(figurename),dpi=600,bbox_inches='tight')
        plt.show()



def sortByPoint(data, point):

    bunderSorted = sorted(point)
    NewbunderData = []

    for k in bunderSorted:
        NewbunderData.append(data[point.index(k)])

    return NewbunderData

def calculateGruopDifferences(data):

    r2s = []
    r2sindex = []

    for i in range(len(data)):

        Alls = []

        for k in range(len(data)):

            # c1 = r2_score(mb[k],targeD)
            # c2 = Tfunction(data[i], data[k])
            c2 = MAE(data[i],data[k])
            # r2 = r2_score(targeD,mb[k])
            # c = c2*c1
            if c2 > 0:

                res = c2  # (log(2-(r2))+c2)
                Alls.append(res)

        r2s.append(np.mean(Alls))

    return r2s

# def FiguresDisply(data,dataName=None,figureName='Figure'):
#
#     plt.figure(figsize=(20, 10))
#     plt.tick_params(labelsize=20)
#     plt.ylabel("{}".format(figureName), fontsize=20)
#     plt.locator_params('y', nbins=5)
#     Fnumbers = 1
#
#     for i in range(len(data)):
#
#         if dataName:
#             plt.plot(range(1,len(data[i])+1), data[i], color=randomcolor(), label='{}'.format(dataName[i]))
#         else:
#             plt.plot(range(1,len(data[i])+1), data[i], color=randomcolor(), label='Data {}'.format(Fnumbers))
#             Fnumbers+=1
#
#     plt.legend()
#     plt.show()


def RSE(pred, true):
    if type(pred) == list:
        pred = np.array(pred)
    if type(true) == list:
        true = np.array(true)
    pred = np.array(pred).reshape(len(pred),pred[0])
    true = np.array(true)
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    if type(pred) == list:
        pred = np.array(pred)
    if type(true) == list:
        true = np.array(true)
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    if type(pred) == list:
        pred = np.array(pred)
    if type(true) == list:
        true = np.array(true)
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    if type(pred) == list:
        pred = np.array(pred)
    if type(true) == list:
        true = np.array(true)
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    if type(pred) == list:
        pred = np.array(pred)
    if type(true) == list:
        true = np.array(true)
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    if type(pred) == list:
        pred = np.array(pred)
    if type(true) == list:
        true = np.array(true)
    d = np.abs((pred - true) / true)
    d = np.where(np.isinf(d), 0, d)
    r = np.mean(d)
    return r


def MSPE(pred, true):
    if type(pred) == list:
        pred = np.array(pred)
    if type(true) == list:
        true = np.array(true)
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    vr2 = log(2-R2(pred, true))

    return mae, mse, rmse, mape, mspe, r2, vr2

def returnErrorPoints(x,y):
    
    MAE_ = []
    MSE_ = []
    MAPE_ = []
    R2_ = []
    VR2_ = []
    
    for k in range(len(y)):
        
        MAE_.append(MAE(y[k], x[k]))

        MSE_.append(MSE(y[k], x[k]))

        MAPE_.append(MAPE(y[k], x[k]))
        
        Tr2 = R2(y[k], x[k])
        R2_.append(Tr2)
        VR2_.append(log(2 -Tr2))
    
    MMAE = np.mean(MAE_)
    MMSE = np.mean(MSE_)
    MMAPE = np.mean(MAPE_)
    MR2 = np.mean(R2_)
    MVR2 = np.mean(VR2_)
    
#     if disp == True:

#         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#         if name:
#             print('The result by {}'.format(name))
#         print('MAE:{}'.format(MMAE),'|'
#               'MSE:{}'.format(MMSE),'|'
#               'MAPE:{}'.format(MMAPE),'|'
#               'R2:{}'.format(MR2),'|'
#               'VR2:{}'.format(MVR2))
#         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return MAE_, MSE_, VR2_  
    
    
def metrics_2(y, y_hat, disp,name):
    # assert y.shape == y_hat.shape  # Tensor y and Tensor y_hat must have the same shape
    # y = y.cpu()
    # y_hat = y_hat.cpu()
    # mape
    _mae= MAE(y, y_hat)

    # smape
    _mse = MSE(y, y_hat)

    # rmse
    _mape = MAPE(y, y_hat)

    # R2
    _r2 = R2s(y, y_hat)
    _vr2 = log(2 -_r2)

    if disp == True:

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if name:
            print('The result by {}'.format(name))
        print('MAE:{}'.format(_mae),'|'
              'MSE:{}'.format(_mse),'|'
              'MAPE:{}'.format(_mape),'|'
              'R2:{}'.format(_r2),'|'
              'VR2:{}'.format(_vr2))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return _mae,_mse,_mape,_r2,_vr2

def metrics_3(y, y_hat, disp, name=None):
    # assert y.shape == y_hat.shape  # Tensor y and Tensor y_hat must have the same shape
    # y = y.cpu()
    # y_hat = y_hat.cpu()
    # mape
    MAE_ = []
    MSE_ = []
    MAPE_ = []
    R2_ = []
    VR2_ = []
    
    for k in range(len(y)):
        
        MAE_.append(MAE(y_hat,  y[k]))

        MSE_.append(MSE(y_hat,  y[k]))

        MAPE_.append(MAPE(y_hat,  y[k]))
        
        Tr2 = R2(y_hat,  y[k])
        R2_.append(Tr2)
        VR2_.append(log(2 -Tr2))
    
    MMAE = np.mean(MAE_)
    MMSE = np.mean(MSE_)
    MMAPE = np.mean(MAPE_)
    MR2 = np.mean(R2_)
    MVR2 = np.mean(VR2_)
    
    if disp == True:

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if name:
            print('The result by {}'.format(name))
        print('MAE:{}'.format(MMAE),'|'
              'MSE:{}'.format(MMSE),'|'
              'MAPE:{}'.format(MMAPE),'|'
              'R2:{}'.format(MR2),'|'
              'VR2:{}'.format(MVR2))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return MMAE, MMSE, MMAPE, MR2, MVR2

def R2(Y,Y_hat):

    # Y = np.array(Y).reshape(len(Y),Y[0])
    # Y_hat = np.array(Y_hat)
    return r2_score(Y,Y_hat)


def R2s(Y, Y_hat):
    # Y = np.array(Y).reshape(len(Y),Y[0])
    # Y_hat = np.array(Y_hat)
    r2s = []
    v2s = []
    for k in range(len(Y)):
        rr = r2_score(Y[k], Y_hat[k])
        r2s.append(rr)
        v2s.append(log(2 -rr))
        
    return np.mean(r2s)

def gauss(x):

    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))

def get_kde(x, data_array, bandwidth=0.1):

    N = len(data_array)
    res = 0
    if len(data_array) == 0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x - data_array[i]) / bandwidth)
    res /= (N * bandwidth)
    return res

def BPreAndTrue(FV, TD, FS, display):

    r2Set = []
    maeSet = []
    mseSet = []

    for i in FV:
        a = i[:FS]
        b = TD[:len(a)]
        mse = MSE(a, b)
        mae = MAE(a, b)
        r2 = r2_score(a, b)
        r2Set.append(r2)
        mseSet.append(mse)
        maeSet.append(mae)
    # Br2Index =
    TheBestscoreR2 = r2Set.index(max(r2Set))
    TheBestscoreMSE = mseSet.index(min(mseSet))

    if display == True:
        print('Number of forecasting results', len(FV))
        print('!!!', max(r2Set), min(mseSet), TheBestscoreR2, TheBestscoreMSE, 'mean', np.mean(r2Set), np.mean(mseSet))
        fingure_4_comparison(FV[TheBestscoreR2], TD, [], [])

    return TheBestscoreR2, r2Set, maeSet, mseSet


def calculateUniques(data):

    results = []

    for k in range(len(data)):

        tr2 = []

        for n in range(len(data)):

            if k != n:

                r2s = Tfunction(data[k], data[n])
                tr2.append(r2s)

        results.append(np.mean(tr2) - np.min(tr2))
        
    return results

def cacluationError_BestFV(FV, TD, FS, display,name=None, Savefigure=False, oringaldata=[], retrunPre=False):

    r2Set = []
    maeSet = []
    mseSet = []
    mapeSet = []
    inverseSSFV = []

    if len(oringaldata)>0:

        xx = MinMaxScaler(feature_range=(0.1,1))
        xx.fit(oringaldata.reshape(-1, 1))

    for i in FV:

        if len(oringaldata)>0:
            fianlforacastingdata = np.array(i[:FS], float).reshape(-1, 1)
            finalPredictedValues_orgianlScalar = xx.inverse_transform(fianlforacastingdata).reshape(-1,)
            TrueInvers = TD.reshape(-1, 1)
            TrueInvers = xx.inverse_transform(TrueInvers).reshape(-1,)

        else:

            finalPredictedValues_orgianlScalar = i[:FS]
            TrueInvers = TD
        mse = MSE(TrueInvers, finalPredictedValues_orgianlScalar)
        mae = MAE(TrueInvers, finalPredictedValues_orgianlScalar)
        vr2 = log(2 - r2_score(TrueInvers, finalPredictedValues_orgianlScalar))
        mape = MAPE(finalPredictedValues_orgianlScalar,TrueInvers)

        r2Set.append(vr2)
        mseSet.append(mse)
        maeSet.append(mae)
        mapeSet.append(mape)

        inverseSSFV.append(finalPredictedValues_orgianlScalar)
    Mr2 = min(r2Set)
    Mmse = min(mseSet)
    Mmae = min(maeSet)
    Mmape = min(mapeSet)
    TheBestscoreR2 = r2Set.index(Mr2)
    TheBestscoreMSE = mseSet.index(Mmse)
    TheBestscoreMAE = maeSet.index(Mmae)

    if display == True:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if name:
            print('The results by {}'.format(name))
        print('Number of forecasting results:', len(FV))
        print('The best result:', 'VR2',Mr2,'|', "MSE",Mmse,'|',"MAE",Mmae,'|',"MAPE",Mmape,"|",'Indexs:',TheBestscoreR2,TheBestscoreMSE,TheBestscoreMAE) #"""'Index:', TheBestscoreR2, TheBestscoreMSE, 'Mean:',np.mean(r2Set), np.mean(mseSet)
        if Savefigure:
            FiguresDisply([inverseSSFV[TheBestscoreR2],TrueInvers],
                          ['The best result by FT model','TrueData'], Savefigure, 'Forecasting VS TrueData')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    if retrunPre:
        return TheBestscoreR2,Mr2,Mmae,Mmse,Mmape,inverseSSFV[TheBestscoreR2],TrueInvers
    else:
        return TheBestscoreR2,Mr2,Mmae,Mmse,Mmape #inverseSSFV[TheBestscoreR2], inverseSSFV[TheBestscoreMSE], inverseSSFV[TheBestscoreMAE],
# def Tfunction(x,y):
# def Tfunction(x,y):
def fingure_4_comparison(data1,data2,data3,data4):

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    # plt.xlabel("Seconds")
    plt.ylabel("{}".format('Forecasting Results'), fontsize = 20)
    plt.xlabel("Points", fontsize = 20)

    # plt.xlabel("S", fontsize = 8)
    # ture_value_1 = power_data_test_one
    plt.locator_params('y', nbins = 10)
    plt.locator_params('x', nbins = 10)

    if len(data3)>1:

        plt.plot(range(len(data2)),data2,ls='--',color = "#01a1b1",label = 'Actural Data',linewidth=0.8,alpha = 1)
        plt.plot(range(len(data1)),data1,'b',color = "#ff4864", label = '{}'.format('Data 2'),alpha = 0.7,linewidth=.8)
        plt.plot(range(len(data3)),data3,'b',color = "#ffb502", label = '{}'.format('Data 3'),alpha = 0.7,linewidth=.8)
        plt.plot(range(len(data4)),data4,'b',color = "#ff4102", label = '{}'.format('Data 4'),alpha = 0.7,linewidth=.8)


    else:

        plt.plot(range(len(data2)),data2,ls='--',color = "#01a1b1",label = 'Actural Data',linewidth=0.8,alpha = 1)
        plt.plot(range(len(data1)),data1,'b',color = "#ff4864", label = '{}'.format('The ADBNN'),alpha = 0.7,linewidth=.8)

    # plt.legend(loc='lower right')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['right'].set_color("#e1e1e1")
    ax.spines['top'].set_color("#e1e1e1")
    ax.spines['bottom'].set_color("#e1e1e1")
    ax.spines['left'].set_color("#e1e1e1")

    plt.legend(loc='lower right',fontsize = 10,framealpha = 1)
    # plt.xlim(640,700)
    # plt.ylim(0,0.63)

    plt.tick_params(labelsize=8,color = "none")

    # plt.savefig('images/{}_prediciton.png'.format(self.ModelName), dpi=1000)
    plt.show()

if __name__ == '__main__':
    pass