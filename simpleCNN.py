from globalVariables import *


def getModel( inputShape, nClasses, dropout=0., finalLayer=256, momentum=0.9, learning_rate=3e-6, name='CNN',maxPool=True, globalAv=False, nAttributes=0 ):
    '''
    This sets up the CNN model. I use the example from the tnesor flow website. But will need 
    adapting and changing
    
    Currently it is a CNN clasification problem using 3 convolutional layers, two max pooling
    two dense fully connected layers that do the meat of the training.
    
    The fitting is a softmax 
    
    
    '''
    
 
    inputLayer = Input(shape=inputShape)
    
    #Add in a convolutional layer, with 32 filters, that have weights randomly initialised to be learnt. 
    #The kernel size is (3,3) and the activation is relu = max(1,x)
    model = layers.Conv2D(32, (3, 3), activation='relu')(inputLayer)
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Then it is max pooled from a 2x2 receptive field
    model = layers.MaxPooling2D((2, 2))( model )
    #Then it is convolved using 64 filters and a 3x3 kernel, ahain with the activation max(0,x)
    model = layers.Conv2D(64, (3, 3), activation='relu')(model)
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Max pooled again
    if maxPool:
        model = layers.MaxPooling2D((2, 2))(model)
        model = layers.Conv2D(64, (3, 3), activation='relu')(model)
        model = layers.BatchNormalization(momentum=momentum)(model)
    if globalAv:
        model = layers.GlobalAveragePooling2D( )( model )
        model = layers.BatchNormalization(momentum=momentum)(model)

   
    
    model = layers.Flatten()(model)
 
    #Then we go in to standard neural network with a fully connected layer to a layer 
    #of 64 neurons and a max(x, 0) activation
    model = layers.Dense(finalLayer, activation='relu')(model)

    #Then another fully connected layer to 10 output neurons, for a classifer this last
    #layer should have the output of the number of class names
    



        
    if nAttributes > 0:
        redshiftLayer = Input( shape=(nAttributes,1))
        redshiftModel = layers.Dense(finalLayer, activation='relu')(redshiftLayer)

        redshiftModel = layers.MaxPooling1D(pool_size=2, padding='same')(redshiftModel)

        redshiftModel = layers.Flatten()(redshiftModel)

        model = layers.concatenate([model, redshiftModel])
        
   
        inputLayer = [inputLayer]
        inputLayer.append(redshiftLayer )
  

    model = layers.Dense(finalLayer, activation='relu')(model)

    model = layers.Dropout( dropout )( model)   
    model = layers.Dense(nClasses)(model)
        
        
        
        
        
    finalModel = Model(inputLayer, model, name=name)
    #optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    print("Learning Rate %0.3E" % learning_rate)
    finalModel.compile(optimizer=optimizer,  
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return finalModel
    
def getModelSeparateChannels( inputShape, nClasses, dropout=0., finalLayer=64, momentum=0.9, learning_rate=0.001, name='CNN',maxPool=True, globalAv=False, nAttributes=0 ):
    '''
    This sets up the CNN model. I use the example from the tnesor flow website. But will need 
    adapting and changing
    
    Currently it is a CNN clasification problem using 3 convolutional layers, two max pooling
    two dense fully connected layers that do the meat of the training.
    
    The fitting is a softmax 
    
    
    '''

    allInputLayers = []
    allModelChannels = []
    nChannels = inputShape[-1]
    
    for iChannel in range(nChannels):
    
        
        inputLayer = Input(shape=(inputShape[0],inputShape[1],1))
        allInputLayers.append( inputLayer )
    
        #Add in a convolutional layer, with 32 filters, that have weights randomly initialised to be learnt. 
        #The kernel size is (3,3) and the activation is relu = max(1,x)
        model = layers.Conv2D(32, (3, 3), activation='relu')(inputLayer)
        model = layers.BatchNormalization(momentum=momentum)(model)
        #Then it is max pooled from a 2x2 receptive field
        model = layers.MaxPooling2D((2, 2))( model )
        #Then it is convolved using 64 filters and a 3x3 kernel, ahain with the activation max(0,x)
        model = layers.Conv2D(64, (3, 3), activation='relu')(model)
        model = layers.BatchNormalization(momentum=momentum)(model)
        #Max pooled again
        if maxPool:
            model = layers.MaxPooling2D((2, 2))(model)
            model = layers.Conv2D(64, (3, 3), activation='relu')(model)
            model = layers.BatchNormalization(momentum=momentum)(model)
        if globalAv:
            model = layers.GlobalAveragePooling2D( )( model )
            model = layers.BatchNormalization(momentum=momentum)(model)

   
    
        model = layers.Flatten()(model)
 
        #Then we go in to standard neural network with a fully connected layer to a layer 
        #of 64 neurons and a max(x, 0) activation
        model = layers.Dense(finalLayer, activation='relu')(model)
        

        
    if nAttributes > 0:
        redshiftLayer = Input( shape=(nAttributes,))
        redshiftModel = layers.Dense(finalLayer, activation='relu')(redshiftLayer)
        #redshiftModel = layers.Dense(finalLayer, activation='relu')(redshiftModel)

        concatModels = layers.concatenate([concatModels, redshiftModel])
        allInputLayers.append(redshiftLayer )
  

    concatModels = layers.Dense(finalLayer, activation='relu')(concatModels)

    concatModels = layers.Dropout( dropout )( concatModels)   
    concatModels = layers.Dense(nClasses)(concatModels)

        
        
    finalModel = Model(allInputLayers, concatModels, name=name)
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    
    print("Learning Rate %0.3E" % learning_rate)
    finalModel.compile(optimizer=optimizer,  
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return finalModel
       
    
    
def getDepthWiseConvolution( inputShape, nClasses, dropout=0., finalLayer=64, momentum=0.9, learning_rate=0.001, name='CNN',maxPool=True, globalAv=False, nAttributes=0 ):

    inputLayer = Input(shape=inputShape)
    
    #Add in a convolutional layer, with 32 filters, that have weights randomly initialised to be learnt. 
    #The kernel size is (3,3) and the activation is relu = max(1,x)
    

    model = tf.keras.layers.DepthwiseConv2D((3,3), strides=[1,1], depth_multiplier=100, padding="same")(inputLayer)

    model = layers.Conv2D(32, (3, 3), activation='relu')(model)
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Then it is max pooled from a 2x2 receptive field
    model = layers.MaxPooling2D((2, 2))( model )
    #Then it is convolved using 64 filters and a 3x3 kernel, ahain with the activation max(0,x)
    model = layers.Conv2D(64, (3, 3), activation='relu')(model)
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Max pooled again
    if maxPool:
        model = layers.MaxPooling2D((2, 2))(model)
        model = layers.Conv2D(64, (3, 3), activation='relu')(model)
        model = layers.BatchNormalization(momentum=momentum)(model)
    if globalAv:
        model = layers.GlobalAveragePooling2D( )( model )
        model = layers.BatchNormalization(momentum=momentum)(model)

    nChannels = inputShape[-1]
    

    
    model = layers.Flatten()(model)
 
    #Then we go in to standard neural network with a fully connected layer to a layer 
    #of 64 neurons and a max(x, 0) activation
    model = layers.Dense(finalLayer, activation='relu')(model)

    #Then another fully connected layer to 10 output neurons, for a classifer this last
    #layer should have the output of the number of class names
    



        
    if nAttributes > 0:
        redshiftLayer = Input( shape=(nAttributes,1))
        redshiftModel = layers.Dense(finalLayer, activation='relu')(redshiftLayer)

        redshiftModel = layers.MaxPooling1D(pool_size=4, padding='same')(redshiftModel)

        redshiftModel = layers.Flatten()(redshiftModel)

        model = layers.concatenate([model, redshiftModel])
        
   
        inputLayer = [inputLayer]
        inputLayer.append(redshiftLayer )
  

    model = layers.Dense(finalLayer, activation='relu')(model)

    model = layers.Dropout( dropout )( model)   
    model = layers.Dense(nClasses)(model)
        
        
        
        
        
    finalModel = Model(inputLayer, model, name=name)
    #optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    print("Learning Rate %0.3E" % learning_rate)
    finalModel.compile(optimizer=optimizer,  
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


    return finalModel
       
    
    


def getRegressionModel( class_names, testLabel):
    '''
    The regression model is currently the same as the classifciaction model, except the final layer is the
    outut of the dense, fully connected model.
                           
                           
                           
    '''
    
    (train_images, train_labels), (test_images, test_labels) = getData(class_names, testLabel=testLabel)
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=train_images[0].shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

     
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam',
              loss='mean_squared_error')

    history = model.fit(train_images, train_labels, epochs=30, 
                    validation_data=(test_images, test_labels))


    return model.predict( test_images )
       