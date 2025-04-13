import pandas as pd
import numpy as np
import model.GSF as fs
from utils.Modeltools import cacluationError_BestFV
import warnings
warnings.filterwarnings("ignore")
import os


class experiment3():

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
            self.TrainBaedline = 11520  # 1440
            self.initionalP = args.initialPosition
            if args.endPosition == 14400:
                self.finalP = args.endPosition- args.Pre_len
            else:
                self.finalP = args.endPosition

        if args.Data == 'ETTm1':
            self.TrainBaedline = 46080  # 2304
            self.initionalP = 46080
            self.finalP = 57600 - args.Pre_len
            if args.endPosition == 57600:
                self.finalP = args.endPosition- args.Pre_len
            else:
                self.finalP = args.endPosition

        if args.Data == 'Google':
            self.TrainBaedline = 1227
            self.initionalP = 1227
            self.finalP = 1533 - args.Pre_len

    def run(self):

        SmoothedData = pd.read_pickle('data/Ex2-3/{}_Smthooed'.format(self.args.Data)).reshape(-1)
        originalData = pd.read_pickle('data/Ex2-3/{}_MaxMin'.format(self.args.Data)).reshape(-1)
        TrainData = SmoothedData[:self.TrainBaedline]
        OrignalTrainData = originalData[:self.TrainBaedline]
        electricity = pd.read_csv('data/Ex2-3/{}.csv'.format(self.args.Data), encoding='gbk')
        if self.args.Data == 'Google':
            OOringaldata = np.array(electricity.Close.values)
            INVRSDATA = OOringaldata[:self.TrainBaedline]
        else:
            OOringaldata = np.array(electricity.OT.values)
            INVRSDATA = OOringaldata[:self.TrainBaedline]
        forecastingstpes = self.args.Pre_len
        Pname = '{}-{}'.format(self.initionalP, self.finalP)
        names = '{}_{}_{}-{}'.format(self.args.Exname, self.args.Data, forecastingstpes, Pname)

        try:
            # print('exist')
            ddsc2 = pd.read_pickle('results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names))
            pre = pd.read_pickle('results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names+'_pre'))
            true = pd.read_pickle('results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names+'_true'))
            resultsFinal = ddsc2
            IPs = int(resultsFinal[-1][0]) + 1

        except:

            print('non-exist')
            resultsFinal = []
            pre = []
            true = []
            IPs = self.initionalP
            # ETTh2_experiment_1x

        for k in range(IPs, self.finalP):  # int(ddsc2[-1][0])+1 13656

            # try:

                BasedT = k
                INVRSDATAALL = INVRSDATA
                LabelData = SmoothedData[:BasedT]
                LabelOrignalData = originalData[:BasedT]
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
                                                                               oringaldata=INVRSDATAALL)
                _, BM2_r2sets, BM2_maeSet, BM2_mseSet, BM2_mapeSet  = cacluationError_BestFV(forecastingResultsByFT, OTrueData,
                                                                               forecastingstpes, display=True,
                                                                               Savefigure=False, oringaldata=INVRSDATAALL)

                _, FFV1_r2sets, FFV1_maeSet, FFV1_mseSet, FFV1_mapeSet,prev,truev = cacluationError_BestFV([FinalForecastingResults_1], OTrueData,
                                                                                  forecastingstpes, display=True,
                                                                                  oringaldata=INVRSDATAALL,Savefigure=False,retrunPre=True)
                _, FFV2_r2sets, FFV2_maeSet, FFV2_mseSet, FFV2_mapeSet  = cacluationError_BestFV([FinalForecastingResults_2], OTrueData,
                                                                                  forecastingstpes, display=False,
                                                                                  oringaldata=INVRSDATAALL)


                self.errors.append(FFV1_r2sets)
                # self.Count+=1
                # if self.Count > 10000:
                #     self.Count=1
                del self.errors[0]
                BM1_FL = len(BM_1)
                BM2_FL = len(forecastingResultsByFT)
                pre.append([k,prev])
                true.append([k,truev])
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
                pd.to_pickle(pre, 'results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names+'_pre'))
                pd.to_pickle(true, 'results/{}/{}/{}/{}'.format(self.args.Exname,self.args.Data, self.args.Pre_len, names+'_true'))
            # except:
            #
            #     print('Error!!!!')

