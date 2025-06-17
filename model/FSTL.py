import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from utils.Modeltools import meragePredictData_FF
from utils.Modeltools import BPreAndTrue
from utils.Modeltools import mergeSameSeriesDE
from utils.Modeltools import separateFunction
from utils.Modeltools import get_kde
from utils.Modeltools import mergeSameSeries
from utils.Modeltools import Tfunction
from utils.Modeltools import meragePredictData_MEAN
from utils.Modeltools import MaxStdFV2
from utils.Modeltools import meragePredictData_WEIGHT
from utils.Modeltools import calculatesimilarseries
from utils.Modeltools import calculateMBs
from utils.Modeltools import calculateGruopDifferences
from utils.Modeltools import sortByPoint
from utils.Modeltools import FiguresDisply
from utils.Modeltools import calculateUniques
from utils.Modeltools import cacluationError_BestFV
from IPython.core.pylabtools import figsize # import figsize
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from math import sqrt
from math import isnan
from math import log2
from math import log
from math import log10
from scipy.signal import argrelextrema
import time
import sys
from collections import Counter
sys.setrecursionlimit(100)
import copy
import math

import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100

class FSTSFModel():

    def __init__(self, Smootheddata, originalData_, args):

        self.data = Smootheddata
        self.originalData = originalData_
        self.forecastingStep = args.Pre_len #forecastingStep
        self.regularData = args.RDY #regularData
        self.DisplayResults = args.Display #DisplayResults
        self.distanceMaxMin = args.SNumber #distanceMaxMin
        self.Noice = args.Noice #distanceMaxMin
        self.forecastingResultsByFT = []
        self.forecastingResultsByKDE = []
        self.TestCharacters = []
        self.TrainDataLastR2 = 0
        self.construction_FN = None
        self.args = args
        self.name = '{}-{}-{}-{}'.format(args.Data,args.Pre_len,args.SNumber,args.Treetype)
        self.TreeType = args.Treetype
        self.thresoldNoice = (len(self.originalData)-self.forecastingStep)/self.forecastingStep
        self.OTD = []
        self.OOTD = []


    def fusionUntilEnd(self,data):

        results = [data[:3], data]
        results_SFR = [data[:3], data]
        N = 0.2
        
        while len(results[-1]) != len(results[-2]):

            FD, _ = mergeSameSeries(results[-1], N, self.forecastingStep)
            results.append(FD)
        
        return results[-1]

    def BackwardFiltering(self, TrueData=[], oringalTrainData=[]):
        
        O_FL = len(self.forecastingResultsByFT)
        self.OTD = TrueData
        self.OOTD = oringalTrainData
        KDE_FL = len(self.forecastingResultsByKDE)
        
        
        if KDE_FL>0:
            UniqueFVs, UniqueDes = self.calculateUniqueDes(self.forecastingResultsByKDE, self.TestCharacters, 0)
        else:
            print('KDE~~Err')
            charateries_ = self.calculateCharateries(self.forecastingResultsByFT, self.forecastingStep, 0.1)
            UniqueFVs, UniqueDes = self.calculateUniqueDes(self.forecastingResultsByFT, charateries_, 0)
    
        allCalss, totalD = self.calculateR2FVtoHDLast(UniqueFVs, self.data, UniqueDes, self.TrainDataLastR2)
        LM_FL = len(totalD)

        allMHMH  = []
        allMHMHs = []
        LatsMaxH = []
        if len(totalD)>5:
        
            for k in allCalss:

                MaxH = self.fusionUntilEnd(k)
                LatsMaxH.append(MaxH)
                Tmeans = calculateUniques(MaxH)
                FinalFvs = sortByPoint(MaxH, Tmeans)
                allMHMHs.append(FinalFvs)
                BASEDL = len(FinalFvs)
                basedline = int(len(FinalFvs) * 0.5)
                
                if basedline<2:
                    basedline = 2
                FinalFvs = FinalFvs[:basedline]
                
                for j in FinalFvs:

                    allMHMH.append(j)

        else:
            
            allMHMH = totalD
            allMHMHs = [totalD]
            
        MHMH_FL = len(allMHMH)    

        BM_1,FinalForecastingResults_1,BM_2,FinalForecastingResults_2,BM_3,FinalForecastingResults_3 = self.bilateralMatching(self.originalData, self.forecastingStep, allMHMH, TrueData)

        if self.args.Display:
            
            print('The number of K:{},runing the GET:'.format(O_FL ))
            _,GET_r2,GET_mae,GET_mse = cacluationError_BestFV(self.forecastingResultsByFT,TrueData,self.args.Pre_len,display=self.args.Display,name='The GET Model', oringaldata=oringalTrainData)
            
            print('1 runing the PS method')
            _,GSPS_r2,GSPS_mae,GSPS_mse = cacluationError_BestFV(UniqueFVs, TrueData, self.args.Pre_len, display=self.args.Display, name='The PS method',oringaldata=oringalTrainData)
            print("Number",self.changePresent((len(UniqueFVs)),O_FL),'VR2',self.changePresent(GSPS_r2,GET_r2),
                  'MAE',self.changePresent(GSPS_mae,GET_mae),'MSE',self.changePresent(GSPS_mse,GET_mse))
            print('-------------------------------------------------------------------------------------------------------------------------------')
            
            print('2 runing the PBS method')
            _,PBS_r2,PBS_mae,PBS_mse = cacluationError_BestFV(totalD, TrueData, self.args.Pre_len, display=self.args.Display, name='The PBS method',oringaldata=oringalTrainData)
            
            print("Number",self.changePresent((len(totalD)),(len(UniqueFVs))),'VR2',self.changePresent(GSPS_r2,PBS_r2),
                  'MAE',self.changePresent(GSPS_mae,PBS_mae),'MSE',self.changePresent(GSPS_mse,PBS_mse))
            print('-------------------------------------------------------------------------------------------------------------------------------')

            print('3 runing the RASS method')
            
            _,PASS_r2,PRASS_mae,PASS_mse = cacluationError_BestFV(allMHMH, TrueData, self.args.Pre_len, display=self.args.Display, name='The RASS method', oringaldata=oringalTrainData)
            print("Number",self.changePresent((len(allMHMH)),(len(totalD))),'VR2',self.changePresent(PASS_r2,PBS_r2),
                  'MAE',self.changePresent(PRASS_mae,PBS_mae),'MSE',self.changePresent(PASS_mse,PBS_mse))
            print('-------------------------------------------------------------------------------------------------------------------------------')
            
            print('4 runing the GR and NS method','number:',MHMH_FL)

        return BM_1,FinalForecastingResults_1,BM_2,FinalForecastingResults_2,BM_3,FinalForecastingResults_3,allMHMH,UniqueFVs,allCalss,LatsMaxH,allMHMH
    
    def changePresent(self,A,B):
        
        return abs((1-(A/B))*100)
    
    def CalculatingFRbnHistorical(self,FV,HD):
        
        allFV = FV
        HSS_1 = HD
        
        errors = []
        for k in allFV:
            r2_mb = mean_squared_error(k, HSS_1)
            errors.append(r2_mb)
        NallFV = sortByPoint(allFV, errors)

        errorsT = []
        for k in allFV:
            r2_mb = Tfunction(k, HSS_1)
            errorsT.append(r2_mb)
        NallFVT = sortByPoint(allFV, errorsT)

        TwoFVF = NallFV[:int(len(NallFV) * 0.5)] + NallFVT[:int(len(NallFVT) * 0.5)]
        TwoFVL = NallFV[-int(len(NallFV) * 0.5):] + NallFVT[-int(len(NallFVT) * 0.5):]
        
        if len(FV) > 10:
            TwoFVF, _ = mergeSameSeries(TwoFVF, 0.2, self.forecastingStep)  # [FF,LL]
            TwoFVL, _ = mergeSameSeries(TwoFVL, 0.2, self.forecastingStep)  # [FF,LL]
        
        BAF = int(len(TwoFVF) * 0.3)
        if BAF<2:
            BAF=2
            
        BAL = int(len(TwoFVL) * 0.3)
        if BAL<2:
            BAL=2
            
        UNIAF = TwoFVF

        UNIAL = TwoFVL  # sortByPoint(AL,UniALIndex)

        AddtionalFAF = meragePredictData_FF(UNIAF[:BAF])  
        AddtionalFAL = meragePredictData_FF(UNIAL[:BAL])  

        F_AF = UNIAF[0]  
        L_AF = UNIAF[-1] 

        F_AL = UNIAL[0]  
        L_AL = UNIAL[-1]  

        AAA = [F_AF, F_AL, L_AF, AddtionalFAF, AddtionalFAL]
        MfinalFV_tr = meragePredictData_FF(UNIAF)
        # print(len(F_AF),len(F_AL),len(MfinalFV_tr))
        
        MAA = meragePredictData_MEAN(AAA)
        FFLL = MAA
        
        return AAA, FFLL, UNIAF, AddtionalFAF, UNIAL, AddtionalFAL

    def bilateralMatching(self, OrignalTrainData_1, forecastingstpes, allFV, TrueData):  # jump81

        HSS_1 = OrignalTrainData_1[-forecastingstpes:]

        if self.Noice == False and self.thresoldNoice>5:

            weight_ALL = (np.exp(-((forecastingstpes) / len(OrignalTrainData_1))) * np.exp(-1000 / ((4 * len(OrignalTrainData_1)))))
            
            if self.DisplayResults == 1:
                
                print('weight_ALL', weight_ALL)
                print('The historical data can be used',len(OrignalTrainData_1)/self.forecastingStep)
                
            NewallMBs_ = self.historicalDataSegment(OrignalTrainData_1, forecastingstpes)
            HfinalStoredData_1, HfinalFV_hr, HfinalStoredData_2, HfinalFV_r2, HfinalStoredData_3, HfinalFV_tr = self.bidirectionalmatching(
                NewallMBs_, allFV, 0, TrueData, OrignalTrainData_1, weight_ALL)

            AllfinalStoredData_1 = HfinalStoredData_1
            AllfinalStoredData_2 = HfinalStoredData_2
            AllfinalStoredData_3 = HfinalStoredData_3
            MfinalFV_hr = HfinalFV_hr
            MfinalFV_r2 = HfinalFV_r2
            MfinalFV_tr = HfinalFV_tr

            return AllfinalStoredData_1, MfinalFV_hr, AllfinalStoredData_2, MfinalFV_r2, AllfinalStoredData_3, MfinalFV_tr

        else:
            
            HfinalStoredData_1, HfinalFV_hr, HfinalStoredData_2, HfinalFV_r2, HfinalStoredData_3, HfinalFV_tr = self.CalculatingFRbnHistorical(allFV,HSS_1)
            
            AllfinalStoredData_1 = HfinalStoredData_1
            AllfinalStoredData_2 = HfinalStoredData_2
            AllfinalStoredData_3 = HfinalStoredData_3
            MfinalFV_hr = HfinalFV_hr
            MfinalFV_r2 = HfinalFV_r2
            MfinalFV_tr = HfinalFV_tr
            
            return AllfinalStoredData_1, MfinalFV_hr, AllfinalStoredData_2, MfinalFV_r2, AllfinalStoredData_3, MfinalFV_tr

    def bidirectionalmatching(self, a, b, type_ab, TrueData, OrignalTrainData_1, weight_ALL):  # jump91
        
        mb = b  # mb_1
        allFV = a
        corretionResults = []
        HSS_1 = OrignalTrainData_1[-self.forecastingStep:]
        originalData = []
        
        for k in range(len(allFV)):

            r2s = []
            r2sindex = []

            for n in range(len(mb)):
                ki = Tfunction(mb[n], allFV[k])
                rre = ki  # ki
                r2s.append(rre)
                r2sindex.append(n)

            Types_ = np.min(r2s)
            indexr2 = r2sindex[r2s.index(Types_)]

            if type_ab == 0:
                
                megd = meragePredictData_WEIGHT(allFV[k], mb[indexr2], weight_ALL)
                originalData.append(mb[indexr2])
            else:
                
                megd = meragePredictData_WEIGHT(mb[indexr2], allFV[k], weight_ALL)
                
            corretionResults.append(megd)
        
        HfinalStoredData_1, HfinalFV_hr, HfinalStoredData_2, HfinalFV_r2, HfinalStoredData_3, HfinalFV_tr = self.CalculatingFRbnHistorical(corretionResults,HSS_1)
        
        return HfinalStoredData_1, HfinalFV_hr, HfinalStoredData_2, HfinalFV_r2, HfinalStoredData_3, HfinalFV_tr

    def explorHistroicalData(self, data, forecastingstpes, basedline, gapTP):

        results = []
        bunderData = []
        cp = data[-forecastingstpes]
        historicalD = data  # list(reversed(data))

        for k in range(len(historicalD) - 1 - forecastingstpes - forecastingstpes):

            lower = 1 + k
            intergredData = historicalD[lower:lower + (forecastingstpes * 2)]
            corredID = intergredData + (cp - intergredData[0])
            HDatas = corredID[:forecastingstpes]
            HSS = historicalD[-forecastingstpes:]
            Tsocre = Tfunction(HDatas, HSS)
            twopoint = abs(corredID[forecastingstpes] - HSS[-1])

            if Tsocre <= basedline and twopoint < gapTP:  

                FDs = corredID[forecastingstpes:]
                results.append(Tfunction(FDs, HSS))  
                bunderData.append(FDs)

        return bunderData, results

    def historicalDataSegment(self, historicalD, forecastingD):

        Lenthrosld = len(historicalD) - 1 - self.forecastingStep - self.forecastingStep
        if self.DisplayResults:
            print('Lenthrosld',Lenthrosld)
        TBD, DPS = self.explorHistroicalData(historicalD, self.forecastingStep, 0.8, 0.05)
        HSS_1 = historicalD[-self.forecastingStep:] #TBD[DPS.index(min(DPS))] #

        mb_ = TBD  # TBD
        cpmb_ = []
        CPP = []

        for k in mb_:
            newff = k  # +(cp-k[0])
            cpmb_.append(newff)
            r2socre = Tfunction(newff, HSS_1)
            CPP.append(r2socre)

        bunderSorted = sorted(CPP)
        NewbunderData = []
        for k in bunderSorted:
                NewbunderData.append(cpmb_[CPP.index(k)])  # .tolist()
        
        
        low = round((np.percentile(bunderSorted, 20)/2)-(np.std(bunderSorted)*0.5),1)
        up = round(np.percentile(bunderSorted, 60)+(np.std(bunderSorted)*0.5),1) 


        firstFV, LastFV = MaxStdFV2(NewbunderData,up,low)
        if self.DisplayResults:
            print('NewbunderData',len(NewbunderData))
        if len(firstFV)+len(LastFV)>150:
            firstFV, _ = mergeSameSeries(firstFV, 0.2, self.forecastingStep)
            LastFV, _  = mergeSameSeries(LastFV, 0.2, self.forecastingStep)

        Nmb_ = firstFV+ LastFV 

        return Nmb_ 

    def calculateR2FVtoHDLast(self, FV, TD, DEs, TrainDataLastR2):

        BasedNumber = int(500+np.exp(log(np.sqrt(2*len(FV)))))

        if len(FV) < BasedNumber * 3:

            historicalR2_2 = []
            TTD = TD[-len(FV[0]):]  
            r2FV = []

            for i in FV:

                R2 = r2_score(i, TTD)                    
                historicalR2_2.append(R2)
                
            SFV = sortByPoint(FV, historicalR2_2,True)
            
            generalBounder = int(len(SFV)*0.3)
            specificBounder = int(len(SFV)*0.6)
            exceptionalBoduner = int(len(SFV)*0.3)
            
            return [SFV[:generalBounder],SFV[generalBounder:specificBounder],SFV[-exceptionalBoduner:]],SFV
        
        else:

            historicalR2_2 = []
            TTD = TD[-len(FV[0]):]  
            r2FV = []

            for i in FV:

                R2 = r2_score(i, TTD)

                if TrainDataLastR2 * R2 >= 0:
                    
                    historicalR2_2.append(abs(R2 - TrainDataLastR2))
                    r2FV.append(i)
            
            # FV = r2FV
            # ShistoricalR2 = sorted(historicalR2_2)
            ShistoricalR2 = sortByPoint(r2FV, historicalR2_2,False)
            generalD = ShistoricalR2[:BasedNumber] 
            specificD = ShistoricalR2[int(len(ShistoricalR2) / 2) - int(BasedNumber / 2):int(len(ShistoricalR2) / 2) + int(BasedNumber / 2)]
            exceptionalD = ShistoricalR2[-BasedNumber:] 
            allD = generalD+exceptionalD+specificD
            finalFV = []
            finalDes = []

            return [generalD,exceptionalD,specificD], allD

    def calculateUniqueDes(self, FV, DEs, limitedNumber):

        a = DEs
        index_all = {}
        for i in range(len(a)):
            target = a[i]
            index_ = []  # 
            for index, nums in enumerate(a):  
                if nums == target:
                    index_.append(index)
            # print(index_)
            index_all[target] = index_

        uniqueFV = []
        uniqueDEs = []
        c = 0

        for key, value in index_all.items():

            c += 1

            if len(value) > limitedNumber:

                uniqueDEs.append(a[value[0]])
                uniqueFV.append(FV[value[0]])

        return uniqueFV, uniqueDEs

    def trainModelAndKDEresults(self):

        # print('Create the FT model')
        forecastingResultsByFT,forecastingResultsByKDE,self.construction_FN = self.DefusionForecasting(self.data, self.originalData,self.forecastingStep,self.distanceMaxMin,self.regularData,self.DisplayResults,self.TreeType)

        return forecastingResultsByFT,forecastingResultsByKDE

    def calculateCRV(self,data):

        CRV = []

        for k in range(1, len(data)):
            ta = (data[k] - data[k - 1])
            CRV.append(ta)
        # print(Ne)
        return CRV

    def DefusionForecasting(self, StrainData_, OtrainData_, forecastingStep_, SP, regularData_, DisplayResults,TreeType):

        # Finde the best upper and lower bounder

        VStureD = OtrainData_[len(StrainData_) - forecastingStep_:len(StrainData_)]
        VStrainD = StrainData_[:len(StrainData_) - forecastingStep_]
        VSOtrainD = OtrainData_[:len(StrainData_) - forecastingStep_]

        FinalSepareationPoints = separateFunction(VStrainD, SP)
        forecastingTree, graph_cells_, graph_edge_, Lastlen_1 = self.constructureNetwork(VStrainD, VSOtrainD,
                                                                                    FinalSepareationPoints,
                                                                                    forecastingStep_, regularData_,
                                                                                    DisplayResults,TreeType)

        # print('FT-forecastingresults-1')
        forecastingValuesTEXT_1 = self.forecasting(forecastingTree, OtrainData_[len(VStrainD)], regularData_, Lastlen_1)
        TheBestR2_1, r2sets, maeSet, mseSet, _ = BPreAndTrue(forecastingValuesTEXT_1, VStureD, forecastingStep_, False)

        FtrainD = StrainData_
        FinalSepareationPoints = separateFunction(FtrainD, SP)

        forecastingTree, graph_cells_, graph_edge_, Lastlen_2 = self.constructureNetwork(FtrainD, OtrainData_,
                                                                                    FinalSepareationPoints,
                                                                                    forecastingStep_, regularData_,
                                                                                    DisplayResults,TreeType)

        # print('FT-forecastingresults-2')
        forecastingValuesTEXT_2 = self.forecasting(forecastingTree, OtrainData_[-1], regularData_, Lastlen_2)
        
        TBr2 = max(r2sets)
        TBmse = min(mseSet)
        TBmae = min(maeSet)
        
        ForecastingResultBestV, ForecastingResultTest, TestCharacters, TBTBC, TrainDataLastR2 = self.CalculateTBKDE(
            forecastingValuesTEXT_1, forecastingValuesTEXT_2, TheBestR2_1, forecastingStep_, FtrainD, VStureD, VStrainD,
            regularData_)
        
        self.forecastingResultsByFT = forecastingValuesTEXT_2
        self.forecastingResultsByKDE = ForecastingResultTest
        self.TestCharacters = TestCharacters
        self.TrainDataLastR2 = TrainDataLastR2

        return forecastingValuesTEXT_2, ForecastingResultTest, graph_edge_

    def calculateCharateries(self, data, forecastingStep_, DG):

        differens = []
        FSteps = forecastingStep_

        for i in data:

            stds = np.std(i[:FSteps])
            medians = np.median(i[:FSteps])
            CRV = self.calculateCRV(i[:FSteps])
            if DG == 0:

                s = log(1 + np.mean(CRV) + math.exp(medians))

            else:
                # print(np.mean(CRV),medians,DG,stds)
                Ds = (medians) / (DG * stds)
                if Ds>700:
                    Ds=700
                s = log(1 + np.mean(CRV) + math.exp(Ds))

            differens.append(s)

        return differens

    def intervalProbility(self, data, ydata, numberIn, TheMaxIndex):

        center_ = data[TheMaxIndex]
        min_ = min(data)
        max_ = max(data)
        std_ = np.std(data) / numberIn 
        interval_ = [99]
        interval_max = [0]

        intervalIndex = [TheMaxIndex]
        intervalIndex_max = [TheMaxIndex]


        k = 1
        while interval_[-1] > min_:

            N_in = center_ - (k * std_)
            interval_.append(N_in)

            for i in range(intervalIndex[-1], -1, -1):
                
                if data[i] < N_in:
                    intervalIndex.append(i)
                    
                    break

            k += 1

        intervalIndex.append(0)
        # print(intervalIndex)
        j = 1
        while interval_max[-1] < max_:

            P_in = center_ + (j * std_)
            interval_max.append(P_in)
            
            for i in range(intervalIndex_max[-1], len(data), 1):


                if data[i] > P_in:
                    
                    intervalIndex_max.append(i)
                    
                    break

            j += 1

        intervalIndex_max.append(len(data) - 1)
        merge_ab = intervalIndex[1:] + intervalIndex_max
        merge_ab = sorted(merge_ab)

        X_intarval = []
        Y_intarval = []

        for i in range(len(merge_ab) - 1):
            Low = merge_ab[i]
            Up = merge_ab[i + 1]
            MY = ydata[Up] 
            X_intarval.append([data[Low], data[Up]])
            Y_intarval.append(MY / max(ydata))
        # print(Y_intarval)

        return X_intarval, Y_intarval

    def findInterval(self, point, Xinterval, Yintarval):

        for i in range(len(Xinterval)):

            if point <= Xinterval[i][1] and point >= Xinterval[i][0]:

                return i, Xinterval[i][0], Xinterval[i][1], Yintarval[i]

    def CalculateTBKDE(self,forecastingV1, forecastingV2, TheBestscoreR2V1, forecastingStep_, TrainData, TureDataV2,
                       TrainDataV2, TrianDataStype):

        DG_ = 2
        R2ToCharateries2 = []
        R2ToCharateries1 = []
        R2ToInterval2 = []
        R2ToInterval1 = []
        R2ToIntervalBounder = []
        R2ToIntervalmean = []
        R2ToIntervalVSBounder = []
        R2ToIntervalVStoTTIntravalBounder = []
        R2ToIntervalVStoNumber = []
        R2TOTheBestChara = []
        R2TOLowprobility = []
        R2ToMeanC1C2 = [9]
        TrainDataLastR2 = r2_score(TrainData[-forecastingStep_:], TrainData[-forecastingStep_ * 2:-forecastingStep_])
        Numbers_ = 0
        while R2ToMeanC1C2[-1] > 0.01:
            Numbers_+=1
            if Numbers_>100:
                break
            charateries_1 = self.calculateCharateries(forecastingV1, forecastingStep_, DG_)
            TheBestChara = charateries_1[TheBestscoreR2V1]
            input_array = charateries_1
            bandwidth = 1.05 * np.std(input_array) * (len(input_array) ** (-1 / 5))
            x_1_array = np.linspace(min(input_array), max(input_array), 100)
            y_1_array = [get_kde(x_1_array[i], input_array, bandwidth) for i in range(x_1_array.shape[0])]
            sortedcharateries_1 = sorted(charateries_1)

            charateries_2 = self.calculateCharateries(forecastingV2, forecastingStep_, DG_)
            input_array = charateries_2
            bandwidth = 1.05 * np.std(input_array) * (len(input_array) ** (-1 / 5))
            x_2_array = np.linspace(min(input_array), max(input_array), 100)
            y_2_array = [get_kde(x_2_array[i], input_array, bandwidth) for i in range(x_2_array.shape[0])]

            Xintavarl_1, Y_intarval_1 = self.intervalProbility(x_1_array, y_1_array, 1, y_1_array.index(max(y_1_array)))
            number_1, UP_I_1, LOW_I_1, YI_1 = self.findInterval(TheBestChara, Xintavarl_1, Y_intarval_1)

            Xintavarl_2, Y_intarval_2 = self.intervalProbility(x_2_array, y_2_array, 1, y_2_array.index(max(y_2_array)))

            R2ToMeanC1C2.append(abs(np.mean(charateries_2) - np.mean(charateries_1)))

            R2ToCharateries2.append(charateries_2)
            R2ToCharateries1.append(charateries_1)
            R2ToInterval2.append(Xintavarl_2)
            R2ToInterval1.append(Xintavarl_1)
            R2TOTheBestChara.append(TheBestChara)
            R2TOLowprobility.append(TheBestChara / UP_I_1)
            correctM = (np.mean(charateries_1) / np.mean(charateries_2))
            VSbounder = TheBestChara / np.mean(charateries_1)
            R2ToIntervalmean.append([np.mean(charateries_1), np.mean(charateries_2)])
            R2ToIntervalBounder.append(VSbounder * correctM)
            R2ToIntervalVSBounder.append(VSbounder)
            correctIntarvalBounder = TheBestChara / LOW_I_1
            R2ToIntervalVStoTTIntravalBounder.append(correctIntarvalBounder)
            R2ToIntervalVStoNumber.append(number_1)
            DG_ = DG_ * 2

        TBIP = R2ToMeanC1C2.index(min(R2ToMeanC1C2)) - 1
        TBIN_VS = R2ToIntervalVStoNumber[TBIP]
        TBI_VS = R2ToInterval1[TBIP][TBIN_VS]
        IntervalBest = []
        
        for k in R2ToInterval2[TBIP]:
            
            Dup = abs(max(k) - max(TBI_VS))
            Dlow = abs(min(k) - min(TBI_VS))
            IntervalBest.append(Dup + Dlow)

        TBC2 = R2ToCharateries2[TBIP]
        TBTBC = R2TOTheBestChara[TBIP]
        ForecastingResultTest = []
        TestCharacters = []
        LEE = (np.sqrt(log(len(R2ToCharateries2[-1])))/log(len(R2ToCharateries2[-1])))
        TBP = LEE*abs(np.mean(R2ToCharateries2[-1]) - TBTBC) 
        
        FUP = TBTBC + TBP
        FLOW = TBTBC - TBP
        
        for i in range(len(TBC2)):

            if TBC2[i] <= FUP and TBC2[i] >= FLOW:

                TestCharacters.append(TBC2[i])
                ForecastingResultTest.append(forecastingV2[i][:forecastingStep_])

        return forecastingV1[TheBestscoreR2V1][:forecastingStep_], ForecastingResultTest, TestCharacters, TBTBC, TrainDataLastR2

    def resultPorbabilityToTrendency(self,arr):

        result = {}

        for i in set(arr):
            result[i] = arr.count(i)

        allProbability = sorted(result.items(), key=lambda x: x[1])
        # print('allProbability', allProbability)
        maximumProbability = sorted(result.items(), key=lambda x: x[1])[len(sorted(result.items(), key=lambda x: x[
            1])) - 1] 

        return maximumProbability

    def calculationTrendency(self,data):

        TemporaryA = []

        for i in range(len(data) - 1):

            a = (data[-1] - data[i]) / data[-1]

            if abs(a) > 0.01:

                TemporaryA.append(a)

            else:

                TemporaryA.append(0)

        return np.mean(TemporaryA)

    def calculationTrendency_1(self,data):

        TemporaryA = []
        a = (data[-1] - data[0])

        return a / len(data)

    def cellsDistanceF(self,cell1, cell2):

        Tchange = cell1[8]
        Tmax = cell1[4]
        Tmin = cell1[5]

        Tchange_1 = cell2[8]
        Tmax_1 = cell2[4]
        Tmin_1 = cell2[5]

        if Tchange * Tchange_1 > 0:

            MaxD = np.sqrt(np.square(Tmax - Tmax_1))
            MinD = np.sqrt(np.square(Tmin - Tmin_1))
            ChangeD = np.sqrt(np.square(Tchange - Tchange_1))
            #         print(cell1,cell2,MaxD,MinD,ChangeD)
            if (MaxD + MinD + ChangeD) / 3 == 0:

                return 1
            else:
                return 0
        else:
            return 0

    def calculatesimilarNote(self, NoteGroup, singalNote, regularData):

        Allresutls = []

        if regularData == 1:

            Basedline = 0.5

        else:

            Basedline = 0.9

        if len(NoteGroup) > 1:

            for k in NoteGroup:

                maxBounder = abs(k[4] - singalNote[4]) / abs(k[4] + singalNote[4])
                minBounder = abs(k[5] - singalNote[5]) / abs(k[5] + singalNote[5])
                LenWeight = abs(k[7] - singalNote[7]) / abs(k[7] + singalNote[7])
                changeWeight = abs(k[8] - singalNote[8]) / abs(k[8] + singalNote[8])
                Allresutls.append((LenWeight + changeWeight) / 2)

            if np.max(Allresutls) >= Basedline:

                return 1

            else:

                return 0
        else:

            return 1

    def constructureNetwork(self, data, originalData_, FinalSepareationPoints, forecastingStep, regularData, DisplayResults, Treetype='BT'):

        cells = []
        allLens = []
        graph_cells = []
        graph_edge = []
        allnumbers = []
        Baseline = 1
        for k in range(len(FinalSepareationPoints) - 1):

            currentCell = data[FinalSepareationPoints[k]:FinalSepareationPoints[k + 1]]
            originalData = originalData_[FinalSepareationPoints[k]:FinalSepareationPoints[k + 1]]
            maxValue = np.max(currentCell)
            minValue = np.min(currentCell)
            meanValue = np.mean(currentCell)

            lenValue = len(currentCell)
            trendencyValue = self.calculationTrendency_1(currentCell)
            noise = [0]

            for p in range(1, len(originalData)):
                
                noise.append((originalData[p] - originalData[p - 1]))
                
            try:

                fathercode = k - 1

            except:

                fathercode = 0

            cells.append(['{}'.format(fathercode), '{}'.format(k), noise,
                          0, maxValue, minValue, lenValue,
                          lenValue, trendencyValue, log(1 + (maxValue * minValue * lenValue * trendencyValue))])

            allLens.append(lenValue)

        cellsWeight = [['None']]

        for n in range(len(cells) - 1):
            a = cells[n]
            b = cells[n + 1]

            maxW = b[4] / a[4]
            minW = b[5] / a[5]
            lenW = b[7] / a[7]
            trendW = b[8]
            cellsWeight.append([b[1], maxW, minW, lenW, trendW])

        constructureCells = [[[0], cells[-3], cells[-2]]]
        constructureForecastingpath = []

        if regularData == 1:

            compeletedFS = forecastingStep + allLens[-1]  #

        else:

            compeletedFS = forecastingStep

        layerN = 1
        TempararyAllLen = [0]

        while len(constructureForecastingpath) < 8000 and len(constructureCells) != 0:

            if DisplayResults == True:

                print(len(constructureCells), len(constructureForecastingpath))
            
            layerN += 1
            TmepararyLen = []
            TempararyPF = []
            Tnewcode = 0

            for p in range(len(constructureCells)):

                Tlen = constructureCells[p][-2][7]
                Tchange = constructureCells[p][-2][8]
                Tmax = constructureCells[p][-2][4]
                Tmin = constructureCells[p][-2][5]

                Tlen_1 = constructureCells[p][-1][7]
                Tchange_1 = constructureCells[p][-1][8]
                Tmax_1 = constructureCells[p][-1][4]
                Tmin_1 = constructureCells[p][-1][5]
                TlenProbability_1 = constructureCells[p][-1][9]
                Tname = constructureCells[p][-1][1]
                TALLlen = constructureCells[p][-1][6]

                Tdistance = []
                Tcode = []

                if layerN == 2 and regularData == 1:

                    # print(len(constructureCells))
                    for i in range(len(cells) - 3):

                        if Tchange * cells[i][8] >= 0 and Tchange_1 * cells[i + 1][8] >= 0 and cells[i + 2][8] * \
                                cells[-1][8] >= 0 and cells[i + 2][-3] >= allLens[-1]:  #

                            Tcode.append(cells[i + 2])

                            if Tchange_1 > 0:

                                goalBounder = abs(cells[i][5] - Tmin) + abs(cells[i + 1][5] - Tmin_1)

                            else:

                                goalBounder = abs(cells[i][4] - Tmax) + abs(cells[i + 1][5] - Tmax_1)

                            allLenWeight = 1 * abs(Tlen - cells[i][7]) + 1 * abs(Tlen_1 - cells[i + 1][7])
                            changeWeight = abs(cells[i][8] - Tchange) + abs(cells[i + 1][8] - Tchange_1)
                            Tdistance.append(
                                (goalBounder + allLenWeight + changeWeight) / 3) 

                    if len(Tdistance) < 1:

                        for i in range(len(cells) - 2):

                            if Tchange * cells[i][8] >= 0 and Tchange_1 * cells[i + 1][8] >= 0:  #

                                Tcode.append(cells[i + 2])

                                if Tchange_1 > 0:

                                    goalBounder = abs(cells[i][5] - Tmin) + abs(cells[i + 1][5] - Tmin_1)

                                else:

                                    goalBounder = abs(cells[i][4] - Tmax) + abs(cells[i + 1][5] - Tmax_1)

                                allLenWeight = 1 * abs(Tlen - cells[i][7]) + 1 * abs(Tlen_1 - cells[i + 1][7])
                                changeWeight = abs(cells[i][8] - Tchange) + abs(cells[i + 1][8] - Tchange_1)
                                Tdistance.append(
                                    (goalBounder + allLenWeight + changeWeight) / 3) 

                else:

                    for i in range(len(cells) - 2):

                        if Tchange * cells[i][8] >= 0 and Tchange_1 * cells[i + 1][8] >= 0:

                            resultsSimilar = self.calculatesimilarNote(Tcode, cells[i + 2], regularData)

                            if resultsSimilar == 1:

                                Tcode.append(cells[i + 2])  ##!!!!!!!!!!

                                if Tchange_1 > 0:

                                    goalBounder = abs(cells[i][5] - Tmin) + abs(cells[i + 1][5] - Tmin_1)

                                else:

                                    goalBounder = abs(cells[i][4] - Tmax) + abs(cells[i + 1][4] - Tmax_1)
                                allLenWeight = 1 * abs(Tlen - cells[i][7]) + 1 * abs(Tlen_1 - cells[i + 1][7])
                                changeWeight = abs(cells[i][8] - Tchange) + abs(cells[i + 1][8] - Tchange_1)
                                Tdistance.append(
                                    (goalBounder + allLenWeight + changeWeight) / 3)  
                                
                sortTcodes = sorted(Tdistance)
                allnumbers.append(len(sortTcodes))
                
                if Treetype == 'BT':
                    
                    sortTcode = sortTcodes[:2] 

                    if int(len(Tdistance)) > 3:
                        sortTcode.append(sortTcodes[int(len(Tdistance) / 2)])
                        
                elif Treetype == 'AT':
                    sortTcode = sortTcodes
                    
                elif Treetype == 'MDT':
                    sortTcode = sortTcodes[:2]
                    
                elif Treetype == 'MDiT':
                    sortTcode = [sortTcodes[0],sortTcodes[-1]]
                    
                for n in range(len(sortTcode)):  
                    # print('n',n)
                    currentPF = copy.deepcopy(constructureCells[p])
                    currentCode = Tcode[Tdistance.index(sortTcode[n])]
                    TcodeCopy = copy.deepcopy(currentCode)
                    TcodeCopy[0] = Tname
                    TcodeCopy[1] = '{}{}'.format(layerN, Tnewcode)
                    TcodeCopy[9] = (TcodeCopy[9] * sortTcode[n])
                    TcodeCopy[3] = currentCode[1]

                    if layerN == 2:

                        TcodeCopy[6] = TcodeCopy[7]
                        TcodeCopy[9] = TcodeCopy[9]

                    else:

                        TcodeCopy[6] = TALLlen + TcodeCopy[7]
                        TcodeCopy[9] = TlenProbability_1 + TcodeCopy[9]

                    currentPF.append(TcodeCopy)
                    currentPF[0][0] += TcodeCopy[7]
                    TmepararyLen.append(currentPF[0][0])

                    if currentPF[0][0] >= compeletedFS:

                        constructureForecastingpath.append(currentPF)

                    else:

                        TempararyPF.append(currentPF)
                    Tnewcode+=1
                    graph_cells.append('{}{}'.format(layerN, Tnewcode))
                    graph_edge.append([TcodeCopy[0], TcodeCopy[1], 1])

            del constructureCells[:]
            constructureCells += TempararyPF
            TempararyAllLen.append(np.mean(TmepararyLen))

        return constructureForecastingpath, graph_cells, graph_edge, allLens[-1]

    def forecasting(self,constructureForecastingpath, initionalFV, regularData, Lastlen):

        Allforecastingresults = []
        maxline = np.max(self.originalData[-self.forecastingStep:])
        minline = np.min(self.originalData[-self.forecastingStep:])

        if self.DisplayResults:

            print('constructureForecastingpath',regularData,len(constructureForecastingpath))

        for i in constructureForecastingpath:

            TempararyForecasintresults = [initionalFV]

            for k in range(3, len(i)):

                Node = i[k]

                if regularData == 1:

                    if k == 3:

                        Tlen = Node[-3] - Lastlen
                        C_c = Node[2][-Tlen:]

                    else:

                        Tlen = Node[-3]
                        C_c = Node[2]
                else:

                    Tlen = Node[-3]
                    C_c = Node[2]
                    
                if self.Noice:
                    weightNoise=1
                else:
                    weightNoise=0
                    
                for g in range(Tlen):

                    ForecastingValue = (TempararyForecasintresults[-1] + Node[-2]) + (weightNoise*C_c[g])

                    if ForecastingValue > 1:

                        TempararyForecasintresults.append(1)

                    elif ForecastingValue < 0:

                        TempararyForecasintresults.append(0)

                    else:

                        TempararyForecasintresults.append(ForecastingValue)

            Allforecastingresults.append(TempararyForecasintresults[0:self.forecastingStep])

        return Allforecastingresults