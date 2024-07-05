# -----------------------------------------------------------------------------
# File    : data.py
# Created : 2024-04-18
# By      : TRILLA Alexandre (alexandre.trilla@alstomgroup.com)
#
# DiaLogs - Log Analytics
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error
from math import sqrt

def DA_GenData(N):
    """ Generate synthetic dataset for a 3 channel redundancy saftey system.
        
    Args:
        N (int): number of instances.

    Returns:
        data (numpy int tensor): [N,Time,Variable], [N,100,4].
            Variable: failure, ch1, ch2, ch3
        rc (numpy int array): Root causes: 1,2,3.
    """
    np.random.seed(42) 
    T = 100
    data = 0.05*np.random.randn(N,T,4)
    labels = np.random.random_integers(1,3,N)
    for n in range(N):
        for t in range(2,T):
            data[n,t,1] += 0.4*data[n,t-1,2] + 0.4*data[n,t-2,3] 
            data[n,t,2] += 0.4*data[n,t-1,1] + 0.4*data[n,t-1,3] 
            data[n,t,3] += 0.4*data[n,t-1,2] + 0.4*data[n,t-2,1] 
            if t > int(T/2):
                data[n,t,labels[n]] += 0.5*np.abs(np.random.randn())
            data[n,t,0] = np.min([1.0, np.abs(0.2*data[n,t,1] +\
                                             0.2*data[n,t,2] +\
                                             0.2*data[n,t,3])])
    data = (20*data+10).astype(int)
    data[:,:,0] -= 10
    for n in range(N):
        for t in range(T):
            if data[n,t,0] < 0:
                data[n,t,0] = 0
            if data[n,t,0] > 10:
                data[n,t,0] = 10
    return data,labels



def DA_Plot(data, labels):
    """ Plot dataset instance by instance.
        
    Args:
        data (numpy tensor): [N,Time,Variable]
        labels (numpy array): Root causes: 1,2,3.
    """
    datatime = np.arange(100)
    for i in range(data.shape[0]):
        plt.figure()
        plt.plot(datatime,data[i,:,0], 'k', label="Alarm Level")
        plt.plot(datatime,data[i,:,1], 'r', label="Channel 1")
        plt.plot(datatime,data[i,:,2], 'g', label="Channel 2")
        plt.plot(datatime,data[i,:,3], 'b', label="Channel 3")
        plt.legend(loc="upper left")
        plt.title("Failure on Channel: " + str(labels[i]))
        plt.xlabel("Time")
        plt.ylabel("Message Count")
        plt.show()


def DA_PlotOne(data, label, predal):
    """ Plot dataset instance by instance.
        
    Args:
        data (numpy tensor): [Time,Variable]
        label (int): Root causes: 1,2,3.
        predal (array): predicted alarm level.
    """
    datatime = np.arange(100)
    plt.figure()
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    plt.plot(datatime,data[:,0], 'k', label="Alarm: $Y$")
    plt.plot(datatime,data[:,1], 'r', label="Channel 1: $X_1$")
    plt.plot(datatime,data[:,2], 'g', label="Channel 2: $X_2$")
    plt.plot(datatime,data[:,3], 'b', label="Channel 3: $X_3$")
    plt.plot(datatime,predal, 'k', linewidth=6, alpha=0.2,
             label="Predicted Alarm: $\hat{Y}$")

    plt.vlines(50, 0, 43, colors='k', linestyles='dashed', 
               label='Incipient Failure (t=T)')
    plt.legend(loc="upper left")
    print("Failure on Channel: " + str(label))
    plt.xlabel("Time (t)")
    plt.ylabel("Message Count")
    plt.show()


def DA_CF(varnam, vardata):
    print("CF: ", varnam)
    plt.figure()
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    plt.plot(vardata[0,:], vardata[1,:], 'k', 
             label="Counterfactual Alarm ($\mu \pm \sigma$): $Y_{*}$(t=T+)")
    plt.plot(vardata[0,32], vardata[1,32], 'ro', markersize=8, alpha=0.4,
             label="Factual Root Cause value: $X_1$(t=T+)")
    plt.plot(vardata[0,8:12], vardata[1,8:12], 'b', linewidth=8, alpha=0.4,
             label="Factual Root Cause value: $X_1$(t<T)")
    #
    hierr = vardata[1,:] + vardata[2,:]
    loerr = vardata[1,:] - vardata[2,:]
    plt.fill_between(vardata[0,:], hierr, loerr, color='k', alpha=0.1)
    #
    plt.legend(loc="lower right")
    plt.xlabel("Counterfactual Root Cause: $X_1^{*}$(t=T+)")
    plt.ylabel("Message Count")
    plt.show()




def paperPlts():
    """ Plots for the workshop paper.
    """
    with open('data_export.pkl', 'rb') as f:
        data = pickle.load(f)
    DA_PlotOne(data['raw_data'], data['fchannel'], data['alarm_pred'])
    print(data['mlp'])
    DA_CF(data['cf_var'], data['cf_data'])
    print("RMSE: ", sqrt(mean_squared_error(data['raw_data'][:,0], data['alarm_pred'])))




#data,labels = DA_GenData(5)
#DA_Plot(data,labels)
paperPlts()
