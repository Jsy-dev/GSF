import pandas as pd
import numpy as np
import model.GSF as fs
from utils.Modeltools import cacluationError_BestFV
import warnings
warnings.filterwarnings("ignore")
import os


class experiment2():

    def __init__(self, args):

        self.args = args
        self.name = '{}-{}-{}-{}'.format(args.Data, args.Pre_len, args.SNumber, args.Treetype)
        self.errors = [10,100]
        self.FTV = None
        self.FTC = None
        self.LV = 0
        self.LC = 0
        self.CorrData = None
        self.Count = 10000
        self.Exname = args.Exname
        print(self.name)
        if args.Data == 'ETTh1' or 'ETTh2':
            self.inverseBasecline = 11520
            self.TrainBaedline = 1440  # 1440
            self.initionalP = args.initialPosition
            if args.endPosition == 14400:
                self.finalP = args.endPosition- args.Pre_len
            else:
                self.finalP = args.endPosition

        if args.Data == 'ETTm1':
            self.inverseBasecline = 46080
            self.TrainBaedline = 2304  # 2304
            self.initionalP = args.initialPosition
            self.finalP = 57600 - args.Pre_len
            if args.endPosition == 57600:
                self.finalP = args.endPosition- args.Pre_len
            else:
                self.finalP = args.endPosition

        if args.Data == 'heat':
            self.inverseBasecline = 2880
            self.TrainBaedline = 960
            self.initionalP = args.initialPosition
            if args.endPosition == 4800:
                self.finalP = args.endPosition - args.Pre_len
            else:
                self.finalP = args.endPosition


    def run(self):
        SmoothedData = pd.read_pickle('data/Ex2&3/{}_Smthooed_31'.format(self.args.Data)).reshape(-1)
        originalData = pd.read_pickle('data/Ex2&3/{}_MaxMin_31'.format(self.args.Data)).reshape(-1)
        TrainData = SmoothedData[:self.TrainBaedline]
        OrignalTrainData = originalData[:self.TrainBaedline]
        electricity = pd.read_csv('data/Ex2&3/{}.csv'.format(self.args.Data), encoding='gbk')
        if self.args.Data == 'Google':
            OOringaldata = np.array(electricity.Close.values)
            INVRSDATA = OOringaldata[:self.inverseBasecline]
        else:
            OOringaldata = np.array(electricity.OT.values)
            INVRSDATA = OOringaldata[:self.inverseBasecline]
        forecastingstpes = self.args.Pre_len
        Pname = '{}-{}'.format(self.initionalP, self.finalP)
        names = '{}_{}_{}-{}'.format(self.args.Exname, self.args.Data, forecastingstpes, Pname)

        try:
            ddsc2 = pd.read_pickle('results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names))            # true = pd.read_pickle('results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names+'_true'))
            resultsFinal = ddsc2
            IPs = int(resultsFinal[-1][0]) + 1

        except:
            print('non-exist')
            resultsFinal = []
            IPs = self.initionalP

        for k in range(IPs, self.finalP):

            try:
                BasedT = k
                KKS = originalData[BasedT - forecastingstpes:BasedT]
                cp = KKS[0]
                KSD = TrainData   + (cp - TrainData[-1])
                KSDO = OrignalTrainData   + (cp - OrignalTrainData[-1])
                LabelData = np.append(KSD, originalData[BasedT - forecastingstpes:BasedT])
                LabelOrignalData = np.append(KSDO, originalData[BasedT - forecastingstpes:BasedT])
                TrueData = originalData[BasedT:BasedT + forecastingstpes]
                OTrueData = SmoothedData[BasedT:BasedT + forecastingstpes]
                print(k, BasedT - forecastingstpes, BasedT, BasedT + forecastingstpes)
                if self.LV<1:
                    self.LV=1
                if self.LC<1:
                    self.LC=1
                Model = fs.FSTSFModel(Smootheddata=LabelData, originalData_=LabelOrignalData, args=self.args,
                                      forecastingTree_V=self.FTV,forecastingTree_C=self.FTC,errors=self.Count,Lastlen_V=self.LV,Lastlen_C=self.LC,CorrData=self.CorrData)
                forecastingResultsByFT, forecastingResultsByKDE,FTV,LV,FTC,LC = Model.trainModelAndKDEresults()
                self.FTV = FTV
                self.FTC = FTC
                self.LV = LV
                self.LC = LC

                BM_1, FinalForecastingResults_1, BM_2, FinalForecastingResults_2,CorrData = Model.BackwardFiltering(
                    TrueData)
                self.CorrData = CorrData
                _, BM1_r2sets, BM1_maeSet, BM1_mseSet, BM1_mapeSet = cacluationError_BestFV(BM_1, OTrueData, forecastingstpes,
                                                                               display=True, Savefigure=False,
                                                                               oringaldata=INVRSDATA)
                _, BM2_r2sets, BM2_maeSet, BM2_mseSet, BM2_mapeSet  = cacluationError_BestFV(forecastingResultsByFT, OTrueData,
                                                                               forecastingstpes, display=True,
                                                                               Savefigure=False, oringaldata=INVRSDATA)

                _, FFV1_r2sets, FFV1_maeSet, FFV1_mseSet, FFV1_mapeSet,prev,truev = cacluationError_BestFV([FinalForecastingResults_1], OTrueData,
                                                                                  forecastingstpes, display=True,
                                                                                  oringaldata=INVRSDATA,Savefigure=False,retrunPre=True)
                _, FFV2_r2sets, FFV2_maeSet, FFV2_mseSet, FFV2_mapeSet  = cacluationError_BestFV([FinalForecastingResults_2], OTrueData,
                                                                                  forecastingstpes, display=False,
                                                                                  oringaldata=INVRSDATA)


                self.errors.append(FFV1_r2sets)
                self.Count+=1
                if self.Count > 10000:
                    self.Count=1
                del self.errors[0]
                BM1_FL = len(BM_1)
                BM2_FL = len(forecastingResultsByFT)
                resultsFinal.append(['{}'.format(BasedT),
                                     BM1_FL, BM1_r2sets, BM1_maeSet, BM1_mseSet, BM1_mapeSet,
                                     BM2_FL, BM2_r2sets, BM2_maeSet, BM2_mseSet, BM2_mapeSet,
                                     0, FFV1_r2sets, FFV1_maeSet, FFV1_mseSet, FFV1_mapeSet,
                                     0, FFV2_r2sets, FFV2_maeSet, FFV2_mseSet, FFV2_mapeSet,
                                     ])

                directory = 'results/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                pd.to_pickle(self.FTV,
                             'results/{}/{}/{}/{}'.format(self.args.Exname, self.args.Data, self.args.Pre_len, names + "_FTV"))
                pd.to_pickle(self.FTC,
                             'results/{}/{}/{}/{}'.format(self.args.Exname, self.args.Data, self.args.Pre_len, names + "_FTC"))
                pd.to_pickle(resultsFinal, 'results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names))

            except:
                self.Count = 10000
                print('Error!!!!',forecastingstpes,BasedT,self.Count,names)

