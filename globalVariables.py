#All various required imports

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, Input, Model
import tensorflow as tf
from keras.callbacks import CSVLogger
import os
from mainModel import simpleModel, mainModel
from testTrainSplit import getTestTrainSets