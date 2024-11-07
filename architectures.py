'''
Copyright (C) November 2024  Alessandro De Santis, alessandro.desantis@roma2.infn.it
'''

from tensorflow.keras.layers    import Input
from tensorflow.keras.layers    import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers    import LeakyReLU, Concatenate, Add
from tensorflow.keras.layers    import GaussianNoise
from tensorflow.keras.layers    import Conv1D, Conv2D,  UpSampling2D, Conv1DTranspose, MaxPool1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, BatchNormalization
import tensorflow as tf
import numpy      as np
import matplotlib.pyplot as plt


def custom_activation_tanh(x):
  return tf.math.tanh(x)

def custom_activation_sigmoid(x):
  return 1/(1+tf.math.exp(-x))

def custom_activation_sin(x):
  return tf.math.sin(x)

def custom_activation_softplus(x):
  return tf.math.log(1 + tf.math.exp(x))

def custom_activation_linear(x):
  return x


def get_architecture(training,geometry):

  if training.activation == 'tanh':
    custom_activation = custom_activation_tanh
  if training.activation == 'sigmoid':
    custom_activation = custom_activation_sigmoid
  if training.activation == 'sin':
    custom_activation = custom_activation_sin
  if training.activation == 'softplus':
    custom_activation = custom_activation_softplus
  if training.activation == 'linear':
    custom_activation = custom_activation_linear

  if training.architecture_type == 'FC':
    
    input_layer   = Input(shape=(1))
    branch        = input_layer
    
    for Nlayer in range(training.Nlayers):
      branch        = Dense(training.Nneurons, kernel_initializer=training.initializer)(branch)
      branch        = custom_activation(branch)                 
    
    output_layer  = Dense( geometry.Noutputs, kernel_initializer=training.initializer)(branch)


  if training.architecture_type == 'GreenMile':
    
    input_layer   = Input(shape=(1))
    
    branch1        = input_layer
    for Nlayer in range(training.Nlayers):
      branch1  = Dense(training.Nneurons, kernel_initializer=training.initializer)(branch1)
      branch1  = custom_activation(branch1)                 
    
    branch2        = input_layer
    for Nlayer in range(training.Nlayers):
      branch2  = Dense(training.Nneurons, kernel_initializer=training.initializer)(branch2)
      branch2  = custom_activation(branch2)                 
    
    output_rho  = Dense(1,kernel_initializer=training.initializer)(branch1)
    output_phi  = Dense(1,kernel_initializer=training.initializer)(branch1)
    output_Prho = Dense(1,kernel_initializer=training.initializer)(branch2)

    joint_layer  = Concatenate()([output_rho,output_Prho,output_phi])
    output_layer = joint_layer


  return input_layer, output_layer



def get_scheduler(training):

  if training.lr_scheduler_type == 0:
    def scheduler(epoch, lr):
      return     training.lri


  if training.lr_scheduler_type == 1:
    def scheduler(epoch, lr):
      initial_lr = training.lri
      final_lr   = training.lri/10
      center     = 100000
      sigma      = 20000
      return     final_lr + (initial_lr-final_lr)/(1+np.exp(-(center-epoch)/sigma))


  return LearningRateScheduler(scheduler)


