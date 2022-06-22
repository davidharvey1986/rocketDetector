import numpy as np
from matplotlib import pyplot as plt
import random
import sys



def getTestTrainSets( split=0.3, dataFile=None, seed = 1):
    
    if dataFile is None:
        dataFile = "../data/nndata.npy"
    else:
        dataFile = "../data/%s" % dataFile
        
    dataA = np.load(dataFile)
    dataB = np.load("../data/nndata2.npy")

    data = np.vstack((dataA, dataB))
    
    labels = np.append(np.ones(data.shape[0]//4), np.zeros(data.shape[0]//4))
    labels = np.append(labels, labels)
    
    random.seed(seed) 
    testSetSelection = random.sample(range(0, data.shape[0]), np.int(data.shape[0]*split)*100//100)
    trainSetSelection = np.delete( np.arange(data.shape[0]), testSetSelection)    
    testSet = {'samples':data[testSetSelection, :, :, np.newaxis], 'labels':labels[testSetSelection][:, np.newaxis]}
    trainSet = {'samples':data[trainSetSelection, :, :, np.newaxis], 'labels':labels[trainSetSelection][:, np.newaxis]}
    
    return trainSet, testSet


