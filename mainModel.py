'''
The various models that i have tested and implemented.

simpleModel : simple CNN 
simpleModel_TypeA : simple CNN with additional 7x1 convolutional layer
mainModel : extreely involved CNN with multiple inception modules.

'''


from globalVariables import *
import inceptionModules 

def simpleModel_TypeA( imageShape, nClasses=4, learning_rate=5e-6, dropout=0., finalLayer=64, momentum=0.9, name='CNN'):
    
    '''
    DATED : 20/06/2020
    
    '''

 
    inputLayer = Input(shape=imageShape)
    
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
    model = layers.MaxPooling2D((2, 2))(model)
    model = layers.Conv2D(64, (3, 3), activation='relu')(model)
    model = layers.BatchNormalization(momentum=momentum)(model)
    model = layers.Conv2D(64, (7, 1), activation='relu')(model)
    model = layers.BatchNormalization(momentum=momentum)(model)
    
    model = layers.Flatten()(model)
 
    #Then we go in to standard neural network with a fully connected layer to a layer 
    #of 64 neurons and a max(x, 0) activation
    model = layers.Dense(finalLayer, activation='relu')(model)

    #Then another fully connected layer to 10 output neurons, for a classifer this last
    #layer should have the output of the number of class names
    
    model = layers.Dense(finalLayer, activation='relu')(model)

    model = layers.Dropout( dropout )( model)   
    model = layers.Dense(nClasses)(model)
        
    finalModel = Model(inputLayer, model, name='mainModel')
    
    
    optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate)

    finalModel.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    
    
        
    return finalModel


def simpleModel( imageShape, nClasses=4, learning_rate=5e-6, dropout=0., finalLayer=64, momentum=0.9, name='CNN'):
    
    '''
    This sets up the CNN model. I use the example from the tnesor flow website. But will need 
    adapting and changing
    
    Currently it is a CNN clasification problem using 3 convolutional layers, two max pooling
    two dense fully connected layers that do the meat of the training.
    
    The fitting is a softmax 
    
    
    '''

 
    inputLayer = Input(shape=imageShape)
    
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
    model = layers.MaxPooling2D((2, 2))(model)
    model = layers.Conv2D(64, (3, 3), activation='relu')(model)
    model = layers.BatchNormalization(momentum=momentum)(model)

    
    model = layers.Flatten()(model)
 
    #Then we go in to standard neural network with a fully connected layer to a layer 
    #of 64 neurons and a max(x, 0) activation
    model = layers.Dense(finalLayer, activation='relu')(model)

    #Then another fully connected layer to 10 output neurons, for a classifer this last
    #layer should have the output of the number of class names
    
    model = layers.Dense(finalLayer, activation='relu')(model)

    model = layers.Dropout( dropout )( model)   
    model = layers.Dense(nClasses)(model)
        
    finalModel = Model(inputLayer, model, name='mainModel')
    
    
    optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate)

    finalModel.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    
    
        
    return finalModel

def mainModel(imageShape, dropout=0.2, momentum=0.9, nClasses=2, learning_rate=1e-4):
    '''
    This is the main model that is based on the architecture of Mertens et al 2015
    
    Conv(3,3,2,2,v,32)
    AN
    Activation ReLU( 0.03 )
    BN
    Conv(3,3,1,1,v,32)
    ReLU( 0.03 )
    Conv(3,3,1,1,s,64)
    BN
    StemInception
    BN
    InceptionA
    BN
    ReductionA
    BN
    InceptionB
    BN
    ReductionB
    BN
    InceptionC
    BN
    GlobalAvgPool
    Dropout (0.3)
    FC (9)
    Softmax
    
    Defaults : Dropout == 0.33, momentum = 0.99
    
    '''
    
    
    
    inputLayer = Input(shape=imageShape)
    
    #Layer 1
    model = layers.Conv2D(32, (3, 3), activation='relu', strides=(2,2), padding='valid')(inputLayer)
    #Normalise
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Layer 2
    model = layers.Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='valid')(model)
    #Normalise
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Layer 3
    model = layers.Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same')(model)
    #Normalise
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Inception A
    model  = inceptionModules.inceptionA( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Reducion A
    model  = inceptionModules.reductionA( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Inception B
    model  = inceptionModules.inceptionB( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Reducion B
    model  = inceptionModules.reductionB( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #Inception C
    model  = inceptionModules.inceptionC( model )
    model = layers.BatchNormalization(momentum=momentum)(model)
    #GlobalAvg
    model = layers.GlobalAveragePooling2D( )( model )
    #DRH: Add another dense layer here.
    model = layers.Dense(256, activation='relu')(model)
    #Dropout
    model = layers.Dropout( dropout )( model)
    #Dense, fully connected layer
    model = layers.Dense( nClasses )(model)
    
    finalModel = Model(inputLayer, model, name='mainModel')
    optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate)

    finalModel.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    
    
    return finalModel

    
    
    
    
