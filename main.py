'''
The main function to train the CNN. In its state it is run in a monte-carlo fashion.
This monte-carlo randomly cuts the test-train samples, each time with a different seed.
This allows me to see the sensitivity of the accuracy of the model to the training set (and by extension the test data)
It also allows me to run the program in the background over the weekend.

SYNTAX: (from command line in bash)

nohup python main.py > mainModel.log &



'''

from globalVariables import *


def monteCarlo(nMonteCarlo=10):
    '''
    Loop through the main function nMonteCarlo times.
    
    '''
    
    for i in range(1,nMonteCarlo):
        fileRootName = "pickles/simple_doubleData_%i" % i
        main(seed=i,fileRootName=fileRootName)

def main( nEpochs=20, testTrainSplit=0.15,\
          fileRootName=None, database=None, \
          dropout=0., seed=1):
    '''
    The main function that trains the model 
    
    OPTIONAL ARGUMENTS :
    ---------------------
    
    - nEpochs : integer : the number of epochs the will train the algorithm
    - testTrainSplit : float : the ratio of test to training split 
    - fileRootName : string : root name for the pickled training sample of clusters
    - dropout : integer : the dropout rate of neurons in the network to avoid overfitting
    - database : string : the database of test and train samples.
    - seed : integer : the seed for the random number generator to split the test and training sets.
    
    '''
    
    #Check for the directory "pickles"
    if not os.path.isdir("pickles"):
        os.system("mkdir pickles")
    
    if fileRootName is None:
        fileRootName = "pickles/main_lr_-4" 
        
        #fileRootName = "pickles/simpleModel_%i_channel_noAtt_dropout_%0.1f_testSplit_%0.3f" % \
        #    (nChannels,dropout,testTrainSplit)
        
        
        
    print("All files saved to %s" % fileRootName)  
    
        
    modelFile = "%s.h5" % (fileRootName)
    csv_file = '%s.csv' % (fileRootName)
    checkpoint_path = '%s.ckpt' % (fileRootName)
        
        
    #Get the training and test labels that will be required.
    trainingSet, testSet = getTestTrainSets(split=testTrainSplit, seed=seed)
        
    print("Number of channels is %i" % trainingSet["samples"].shape[-1])
               
    #Check to see if a previous model exists to continue training off
        
        
    if os.path.isfile(modelFile):
        print("FOUND PREVIOUS MODEL, LOADING...")
        mertensModel = models.load_model(modelFile)
    else:
        mertensModel = simpleModel( trainingSet['samples'][0].shape, dropout=dropout )
            
    mertensModel.summary()
    
    #Set up some logging for the model
    #This is a checkpoint to save along the way
    
    #This is a csv to log the history of the training
    
    csv_logger = CSVLogger(csv_file, append=True)

    #Check to see if the previous csv logger exists, since i will want to continue
    #Number the training from the previous state
    if os.path.isfile( csv_file ):
        try:
            previousEpochs = np.loadtxt( csv_file, delimiter=',',skiprows=1 )
            initial_epoch = previousEpochs.shape[0]
        except:
            #this means the file was created but nothing is in there
            initial_epoch = 0
    else:
        initial_epoch = 0
        
    
    print("Starting from epoch %i" % initial_epoch)
        
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
        
    # Train the model with the new callback
    inceptionHistory = mertensModel.fit(trainingSet['samples'], 
              trainingSet['labels'],  
              epochs=nEpochs,
              validation_data=(testSet["samples"], testSet["labels"]),
              initial_epoch=initial_epoch,
              callbacks=[cp_callback, csv_logger])  # Pass callback to training

    mertensModel.save(modelFile)
    
if __name__ == '__main__':
    monteCarlo()
    
