'''
Copyright (C) November 2024  Alessandro De Santis, alessandro.desantis@roma2.infn.it
'''

import numpy      as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import shutil
import os

from datetime     import datetime
from pathlib      import Path
from Losses       import get_preconditioner, Loss_PlanarGeometry, Loss_NonPlanarGeometry, Loss_SeparableGeometry, Loss_NonlinearOscillator, Loss_ChaoticOscillator

from tensorflow.keras.models import Model, Sequential
from architectures import get_architecture, get_scheduler
from tensorflow.keras.optimizers.legacy import Adam
from keras.initializers import HeNormal, GlorotNormal, GlorotUniform
from keras.callbacks import CSVLogger
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class Training:

  def __init__(self,dict,input,geometry):
        
    self.epochs                  = int(dict["epochs"])                   if "epochs"                  in dict.keys() else 10000
    self.lambda_                 = float(dict["lambda"])                 if "lambda"                  in dict.keys() else 0.5
    self.initializer_type        = str(dict["initializer"])              if "initializer"             in dict.keys() else 'random_normal'
    self.lri                     = float(dict["lri"])                    if "lri"                     in dict.keys() else 4e-8
    self.lr_scheduler_type       = int(dict["lr_scheduler_type"])        if "lr_scheduler_type"       in dict.keys() else 1
    self.preconditioner_type     = int(dict["preconditioner_type"])      if "preconditioner_type"     in dict.keys() else 0
    self.set_architecture(dict['architecture'])
    
    self.set_initializer()
    self.training_time = self.read_training_time(dict)
    self.print_time()
    input.create_output_directory(self,geometry)

  def set_architecture(self,dict):
    
    print(dict)
    self.architecture_type = dict['type']       if  'type'       in dict.keys() else 'FC'
    self.Nlayers           = dict['layers']     if  'layers'    in dict.keys()  else  2
    self.Nneurons          = dict['neurons']    if  'neurons'   in dict.keys()  else  48
    self.activation        = dict['activation'] if  'activation' in dict.keys() else 'tanh'
    self.architecture_name = f'{self.architecture_type}_L{self.Nlayers}_N{self.Nneurons}_{self.activation}' if self.architecture_type == 'FC' else self.architecture_type
    print(self.architecture_name)

  def read_training_time(self,dict):

    tmp_ = []
    for key in dict.keys():
      if "time_range"  in key:
        range_type =  dict[key]['type'] if 'type' in dict[key].keys() else 'linear'
        print(range_type)
        if range_type == 'linear':
          range_time =  np.linspace(dict[key]["t_start"],dict[key]["t_final"],dict[key]["Npoints"]) 
        if range_type == 'log':
          range_time    = np.sort(  1 + dict[key]["t_final"] - np.logspace( dict[key]["t_start"], np.log(dict[key]["t_final"]+1), dict[key]["Npoints"], base=np.exp(1), endpoint = True) )
          range_time[0] = 0
        tmp_.append(range_time)
    
    tmp_ = np.concatenate(tmp_)
    tmp_.sort()
    return tmp_

  def write_info(self,input,geometry):

    fp =  open(input.destination_path/Path('info.dat'),'w+')
    fp.write(f'# {datetime.now()}\n\n')
    geometry.write_to_file(fp)
    fp.write('\n[TRAINING]\n\n')
    fp.write(f'architecture        {self.architecture_name}\n')
    fp.write(f'Noutputs            {geometry.Noutputs}\n')
    fp.write(f'epochs              {self.epochs}\n')
    fp.write(f'initializer_type    {self.initializer_type}\n')
    fp.write(f'preconditioner_type {self.preconditioner_type}\n')
    fp.write(f'scheduler_type      {self.lr_scheduler_type}\n')
    fp.write(f'lambda              {self.lambda_}\n')
    fp.write(f'lri                 {self.lri}\n')
    fp.write(f't_start             {min(self.training_time)} \n')
    fp.write(f't_final             {max(self.training_time)} \n')
    fp.write(f'npoints             {len(self.training_time)} \n')
    fp.write('\n')
    fp.write('t [%-3i] =  %12.10f\n' % (1,self.training_time[0]))
    for i in range(1,len(self.training_time)):
      fp.write('t [%-3i] =  %12.10f    %6.5e\n' % (i+1,self.training_time[i],self.training_time[i]-self.training_time[i-1]))
    fp.close()

  def set_initializer(self):

    if self.initializer_type == 'random_normal':
      self.initializer = 'random_normal'
    if self.initializer_type == 'HeNormal':
      self.initializer = HeNormal()      
    if self.initializer_type == 'GlorotNormal':
      self.initializer = GlorotNormal()
    if self.initializer_type == 'GlorotUniform':
      self.initializer = GlorotUniform()

  ###################################

  def set_model(self,input,geometry):

    self.X_train = self.training_time
    self.Y_train = np.zeros(len(self.X_train))
    opt          = Adam(learning_rate=self.lri)

    self.callbacks  = []
    save_best_model = tf.keras.callbacks.ModelCheckpoint(input.destination_path / Path('weights_best_model.h5'),save_weights_only= True,save_best_only = True,monitor='loss',mode='min',verbose=0)  
    csv_logger      = CSVLogger(input.destination_path / Path('loss.dat'), append=False, separator=' ')
    scheduler       = get_scheduler(self)
    
    self.callbacks.append(save_best_model)
    self.callbacks.append(csv_logger)
    self.callbacks.append(scheduler)
    
    self.input_layer, self.output_layer = get_architecture(self,geometry)
    self.model  = Model( inputs = self.input_layer , outputs = self.output_layer)
    self.model.summary()
    param = self.model.count_params()
    self.h, self.dh_dt = get_preconditioner(self)
    
    if input.geometry_type == 'planar':
      custom_loss = Loss_PlanarGeometry
    if input.geometry_type == 'non_planar':
      custom_loss = Loss_NonPlanarGeometry
    if input.geometry_type == 'separable':
      custom_loss = Loss_SeparableGeometry        
    if input.geometry_type == 'nonlinear_oscillator':
      custom_loss = Loss_NonlinearOscillator
    if input.geometry_type == 'chaotic_oscillator':
      custom_loss = Loss_ChaoticOscillator
    
    self.model.compile( loss      =  custom_loss(   self.input_layer,self.output_layer,self,geometry,self.h,self.dh_dt,self.lambda_,0),
                        metrics   = [
                                    tf.keras.metrics.MeanMetricWrapper(custom_loss(   self.input_layer,self.output_layer,self,geometry,self.h,self.dh_dt,0,0), name="metrics_dynamics"),
                                    tf.keras.metrics.MeanMetricWrapper(custom_loss(   self.input_layer,self.output_layer,self,geometry,self.h,self.dh_dt,1,0), name="metrics_energy"),
                                    # tf.keras.metrics.MeanMetricWrapper(custom_loss(   self.input_layer,self.output_layer,self,geometry,self.h,self.dh_dt,0,1), name="relative_metrics_dynamics"),
                                    # tf.keras.metrics.MeanMetricWrapper(custom_loss(   self.input_layer,self.output_layer,self,geometry,self.h,self.dh_dt,1,1), name="relative_metrics_energy")                                   
                                      ],  
                                        optimizer = opt )

    fp =  open(input.destination_path/Path('architecture.dat'),'w+')
    self.model.summary(print_fn=lambda x: fp.write(x + '\n\n'))
    fp.close()

  def perform_training(self,input):

    print(f'Start training for {self.epochs} epochs ...\n')

    if input.mode == 'restart':
      if (input.destination_path / Path('weights_best_model.h5')).exists():
        self.model.load_weights(input.destination_path / Path('weights_best_model.h5'))         
      else:
        print('Previous weights not found, training from scratch')

    start_time   = time.time()
    self.history = self.model.fit( x = self.X_train, y = None, epochs=self.epochs, verbose = 0,batch_size = len(self.X_train), callbacks = self.callbacks)
    self.elapsed_time = time.time()-start_time

    print('Elapsed time = %i sec = %i min' % (self.elapsed_time,self.elapsed_time/60))
    fp = open(input.destination_path / Path('architecture.dat'),'a')
    fp.write('\n\n')
    fp.write('Training time = %i sec = %i min' % (self.elapsed_time,self.elapsed_time/60))
    fp.close()
      
    os.remove(input.destination_path / Path('loss.dat'))

    loss           = np.array(self.history.history['loss'])
    lr             = np.array(self.history.history['lr'])
    loss_dynamics  = np.array(self.history.history['metrics_dynamics'])
    loss_energy    = np.array(self.history.history['metrics_energy'])
    # relative_loss_dynamics  = np.sqrt(np.array(self.history.history['relative_metrics_dynamics']))
    # relative_loss_energy    = np.sqrt(np.array(self.history.history['relative_metrics_energy'])  )
    loss_sum       = loss_dynamics + loss_energy
    epochs         = np.arange(1,len(loss)+1)

    fp = open(input.destination_path / Path('loss.dat') , 'w+')
    fp.write('# $1 epochs             \n')
    fp.write('# $2 minimized loss     \n')
    fp.write('# $3 energy             \n')
    fp.write('# $4 dynamics           \n')
    fp.write('# $5 energy + dynamics  \n')
    fp.write('# $6 learning rate      \n')
    fp.write('\n\n')

    nstep  = 10

    if nstep > 1:
      fp.write('%-12i      %7.6e      %7.6e      %7.6e      %7.6e      %7.6e\n' % (1,loss[0],loss_energy[0],loss_dynamics[0],loss_sum[0],lr[0]))

    for i in range(nstep-1,self.epochs,nstep):
      fp.write('%-12i      %7.6e      %7.6e      %7.6e      %7.6e      %7.6e\n' % (i+1,loss[i],loss_energy[i],loss_dynamics[i],loss_sum[i],lr[i]))
    fp.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs,loss,         color='black',label=r'$(1-\lambda)L^\mathrm{dyn} + \lambda L^\mathrm{E}$')
    plt.plot(epochs,loss_dynamics,color='blue' ,label=r'$L^\mathrm{dyn}$')
    plt.plot(epochs,loss_energy,  color='green',label=r'$L^\mathrm{E}$')
    plt.plot(epochs,loss_sum,     color='red' ,label=r'$L^\mathrm{dyn} + L^\mathrm{E}$')
    plt.xlabel(r'Epochs', fontsize=14)
    plt.ylabel(r'Loss'  , fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(lw=0.5,ls='dotted',color='gray')
    plt.legend(loc='best',fontsize=14)
    plt.tight_layout()
    plt.savefig(input.destination_path / Path('loss.pdf'),format='pdf') 
    plt.close()

  def print_time(self):
    print(f'Number of points = {len(self.training_time)}\n')
    for i in range(len(self.training_time)):
      print('t[%-3i] =  %12.10f' % (i+1,self.training_time[i]))

  def prediction(self,input,geometry,methods):
    
    Npoints = max(2000,len(self.training_time))
    evaluation_time = np.linspace(min(self.training_time),max(self.training_time),Npoints)

    self.model.load_weights(input.destination_path / Path('weights_best_model.h5'))         
    geometry.Predict(self,evaluation_time,input)

    for method in methods:
      geometry.Numerical_integration(evaluation_time,method,input)
    



