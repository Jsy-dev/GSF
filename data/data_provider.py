import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os
from utils.Modeltools import SoomtingData
data_dict = {
    'USE': ['USE_Smoothing.npy','USE_Oringal.npy',450],
    'WindSpeed': ['WindSpeed_Smoothing.npy','WindSpeed_Oringal.npy',1000],
    'SunPower': ['SunPower_Smoothing.npy','SunPower_Oringal.npy',1000],
    'SunSpots': ['SunSpots_Smoothing.npy','SunSpots_Oringal.npy',1200],
    'ETTh1': ['SunSpots_Smoothing.npy','SunSpots_Oringal.npy',1200],
    
    # 'custom': [],
}

def data_provider(args):
    # directory = os.getcwd()
    # print(directory)
    # os.path.join('./Ex1',
    #              [args.Data][0])

    Data = np.load(os.path.join('./data/Ex1/',data_dict[args.Data][0]))
    OringalData  = np.load(os.path.join('./data/Ex1/',data_dict[args.Data][1]))
    OringalData = OringalData.reshape(-1,)
    Basedline = data_dict[args.Data][2]

    if args.Smoothing:
        SmootedtrianData = SoomtingData(Data)[:Basedline]
    else:
        SmootedtrianData = Data[:Basedline]

    if args.Scale:
        SclarF = MinMaxScaler()
        print('Scale')
        MaxMintrainData = SclarF.fit_transform(OringalData[:Basedline].reshape(-1,1))
        MaxMintrainData = MaxMintrainData.reshape(-1,)

    else:
        SclarF = None
        MaxMintrainData = OringalData[:Basedline]

    trueData  = OringalData[Basedline:Basedline+args.Pre_len]
    OringalTrainData = OringalData[:Basedline]
    if max(SmootedtrianData)>1 or min(SmootedtrianData)<0:

        return print('The data must be to scale with 0 to 1')

    return SmootedtrianData,MaxMintrainData,OringalTrainData,trueData,OringalData,SclarF
