import numpy as np
import random
from utils.Modeltools import meragePredictData_FF
from utils.Modeltools import BPreAndTrue
from utils.Modeltools import separateFunction
from utils.Modeltools import DisplayseparationSeries
from utils.Modeltools import get_kde
from utils.Modeltools import mergeSameSeries
from utils.Modeltools import Tfunction
from utils.Modeltools import meragePredictData_MEAN
from utils.Modeltools import MaxStdFV2
from utils.Modeltools import meragePredictData_WEIGHT
from utils.Modeltools import sortByPoint
from utils.Modeltools import GnerateGF
from utils.Modeltools import calculateUniques
from utils.Modeltools import cacluationError_BestFV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import log
import copy
import math
import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

class FSTSFModel():

    def __init__(self, Smootheddata, originalData_, args, forecastingTree_V=None, Lastlen_V=None, forecastingTree_C=None, Lastlen_C=None, errors=10000, CorrData=None):

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
        self.errors = errors
        self.forecastingTree_V_L = forecastingTree_V
        self.forecastingTree_C_L = forecastingTree_C
        self.Lastlen_V_L = Lastlen_V
        self.Lastlen_C_L = Lastlen_C
        self.CorrData_L = CorrData
        self.reconstructure = self.calculatingRec()

    def calculatingRec(self):
        return True

    def displayFTmodelConstrucation(self):

        GnerateGF(self.construction_FN,self.name)

    def BackwardFiltering(self, TrueData=[], oringalTrainData=[]):
        
        O_FL = len(self.forecastingResultsByFT)
        print('The number of K:{},runing the BF:'.format(O_FL ))

        # 0 the forecasting result by GSF model

        # 1 KDE method
        KDE_FL = len(self.forecastingResultsByKDE)
        
        print('1 runing the PSS method','number:',O_FL)
        if KDE_FL>0:
            UniqueFVs, UniqueDes = self.calculateUniqueDes(self.forecastingResultsByKDE, self.TestCharacters, 0)
        else:
            print('KDE~~Err')
            charateries_ = self.calculateCharateries(self.forecastingResultsByFT, self.forecastingStep, 0.1)
            UniqueFVs, UniqueDes = self.calculateUniqueDes(self.forecastingResultsByFT, charateries_, 0)

        # 2 the PBS method
        print('2 runing the PBS method','number:',KDE_FL)
        LastFVs, LastDes = self.calculateR2FVtoHDLast(UniqueFVs, self.data, UniqueDes, self.TrainDataLastR2)
        LM_FL = len(LastFVs)

        #3 the RSS method'
        print('3 runing the RSS method','number:',LM_FL)
        
        if LM_FL>5:
            FEND = self.fusionUntilEnd(LastFVs)
            UniquesIndexs = calculateUniques(FEND)
            BASEDL = int(len(FEND) * 0.5)
            if BASEDL < 5:
                BASEDL = 5
            FinalFvs = sortByPoint(FEND, UniquesIndexs)[:BASEDL]
            MHMH_FL = len(FinalFvs)
        else:
            FinalFvs = LastFVs
            MHMH_FL = LM_FL
            
        print('4 runing the GR and NS method','number:',MHMH_FL)
        BM_1,FinalForecastingResults_1,BM_2,FinalForecastingResults_2,BM_3,FinalForecastingResults_3,CorrData = self.bilateralMatching(self.originalData, self.forecastingStep, FinalFvs, TrueData)

        return BM_1,FinalForecastingResults_1,BM_2,FinalForecastingResults_2,CorrData

