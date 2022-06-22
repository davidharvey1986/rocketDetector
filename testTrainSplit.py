'''
This splits the simulated data set in to test and training sets randomly
using the given seed for the random number generator.


'''

import numpy as np
from matplotlib import pyplot as plt
import random
import sys



def getTestTrainSets( split=0.3, dataFile=None, seed = 1):
    
    '''
    Inputs : None
    Keywords : 
        split : float : the fractional split that the test set is as a function of the total number of samples.
                        soon training set size = (1-split)*nTotalSamples
        dataFile : string : the datafile of the training data.
        seed : integer : the seed to be used for the random number generator
        
    Returns : trainSet : a dict with labels "samples" and "labels"
                    trainSet["samples"] = nxmxp array, where n=number of samples, nxm the size of the array (100x100)
                    trainSet["labels"] = a 1 dimensional numpy array with n samples, either 1 or 0, correspinding to the label
                                            of a rocket existing or not.
                                            
                    testSet is in the same format and structure as trainSet 
        
    
    '''
    
    if dataFile is None:
        dataFile = "../data/nndata.npy"
    else:
        dataFile = "../data/%s" % dataFile
        
    data = np.load(dataFile)

    
    labels = np.append(np.ones(data.shape[0]//4), np.zeros(data.shape[0]//4))
    labels = np.append(labels, labels)
    
    random.seed(seed) 
    testSetSelection = random.sample(range(0, data.shape[0]), np.int(data.shape[0]*split)*100//100)
    trainSetSelection = np.delete( np.arange(data.shape[0]), testSetSelection)    
    testSet = {'samples':data[testSetSelection, :, :, np.newaxis], 'labels':labels[testSetSelection][:, np.newaxis]}
    trainSet = {'samples':data[trainSetSelection, :, :, np.newaxis], 'labels':labels[trainSetSelection][:, np.newaxis]}
    
    return trainSet, testSet


