

# !pip install tensorflow==2.15.1

# TRAINING

import os
import time
import numpy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, RMSprop, Nadam, Adagrad
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf
from scipy.integrate import solve_ivp

from tensorflow import keras, Variable
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU, Concatenate, ReLU
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Conv2D,  UpSampling2D, Conv1DTranspose, MaxPool1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.initializers import HeNormal, Zeros, Ones
from tensorflow.python.framework.ops import disable_eager_execution
import sys
disable_eager_execution()

tf.keras.backend.set_floatx('float64')  # Set the default float type to float64


def info(obj,name):
  print(name,' shape: ',np.shape(obj),' type: ',type(obj), '\n')

np.random.seed(1)
keras.utils.set_random_seed(5)


#### geodesic parameters
L  = 1
af = 0.1
b  = 1.9 # 1.88 ok
rho_crit = 0.8944271909999158785636694674925104941762

#### Integration parameters
rho_0   = 10
phi_0   = np.arcsin(b/rho_0)
t_start = 0
t_final = 15
points  = int((t_final-t_start)*2)
t_eval  = np.linspace(t_start,t_final,points)
# t_eval  = np.concatenate(  (np.linspace(0.0,15.0,5),np.linspace(15,t_final,140)))
#t1 = np.arange(0,30,0.5)
#t2 = np.arange(30.25,60.2,0.5)
#t_eval = np.concatenate((t1,t2))
#print(t_eval)


def heaviside(x):
    return tf.where(x >= 0,  tf.ones_like(x), tf.zeros_like(x))

def drho_dt_tf(rho):
  return    (-(rho**2+af**2)*tf.math.sqrt(  (rho**2+L**2)**2+(af-b)*(rho**2 *(af+b)+2*af*L**2)    )/(af**2*(rho**2+2*L*L)+(rho**2+L**2)**2-af*b*L*L  ))# *heaviside(rho-rho_crit)

def radicand_drho_dt(rho):
  return (rho**2+L**2)**2+(af-b)*(rho**2 *(af+b)+2*af*L**2)

def denominator_drho_dt(rho):
  return  af**2*(rho**2+2*L*L)+(rho**2+L**2)**2-af*b*L*L

def dphi_dt(rho):
  return (b*rho*rho+af*L*L)/((rho**2+L**2)**2+(af**2)*(rho**2+2*L*L)-af*b*L*L)

def drho_dt_float(rho):
  return -(rho**2+af**2)*np.sqrt((rho**2+L**2)**2+(af-b)*(rho**2 *(af+b)+2*af*L**2))/(af**2*(rho**2+2*L*L)+(rho**2+L**2)**2-af*b*L*L)


### System for numerical integration via  solve_ivp
def system(t, y):
  rho, phi = y
  drho_dt  = -(rho**2+af**2)*np.sqrt(  np.abs((rho**2+L**2)**2+(af-b)*(rho**2 *(af+b)+2*af*L**2))   )/(af**2*(rho**2+2*L*L)+(rho**2+L**2)**2-af*b*L*L)
  dphi_dt  = (b*rho*rho+af*L*L)/((rho**2+L**2)**2+(af**2)*(rho**2+2*L*L)-af*b*L*L)
  return [drho_dt, dphi_dt]


#### Functions for the HNN
def f(t):
  return 1 - tf.math.exp(-t)

def df(t):
  return tf.math.exp(-t)

def weight(t):
  return 1/(1+tf.math.exp(-(t-40)/5))

def activation_tanh(x):
  return tf.math.tanh(x)

def activation_sin(x):
  return tf.math.sin(x)

def activation_cos(x):
  return tf.math.cos(x)




  # rho = tf.clip_by_value(rho, 0.99999*rho_crit, 1e3)
  # eq3           = tf.square(rho_dot)
  # eq3           = tf.square(rho-rho_crit)
  # lambda_       = weight(t)
  # total_loss  = (1-lambda_)*(eq1)+lambda_*eq3 + eq2


def custom_loss(t,N):

  def loss(y_true, y_pred):

    ft  =  f(t)
    dft = df(t)

    with tf.GradientTape() as tape:
      tape.watch(t)
      predictions = model(t,training=False)
    gradients     = tape.batch_jacobian(predictions,t)
    gradients     = tf.reshape(gradients, (-1,1))

    rho           = rho_0+ft[:,0]*N[:,0]
    # rho = tf.clip_by_value(rho, rho_crit, 1e3)

    rho_dot       = ft[:,0]*gradients[:,0] + dft[:,0]*N[:,0]
    # rho_dot       = tf.clip_by_value(rho_dot, -1,0.1)

    eq1           = tf.square(rho_dot-drho_dt_tf(rho))
    total_loss    = eq1

    return tf.reduce_mean(total_loss)

  return loss



#### Training parameters
arc          = 'arc1'
lri          = 1e-3
X_train      = t_eval
Y_train      = np.zeros(len(X_train))
initializer  = 'random_normal' # HeNormal()# 'random_normal'
opt          = Adam(learning_rate=lri,clipnorm=0.1)
csv_logger   = CSVLogger(f'loss.dat', append=False, separator=' ')


# Callbacks
def scheduler(epoch, lr):
  if epoch <= 10000:
    lr = lri
  else:
    # lr = lr/(1+0.000001*epoch)
    lr = 0.0
  return lr

reduce_lr = ReduceLROnPlateau(
    monitor='loss',       # Metrica da monitorare (es. 'val_loss', 'loss', 'accuracy')
    factor=0.1,           # Fattore di riduzione del learning rate (es. 0.1 riduce il lr di un decimo)
    patience=10,          # Numero di epoche con nessun miglioramento dopo cui ridurre il learning rate
    verbose=1,            # Stampa messaggi quando il learning rate viene ridotto
    # min_delta=,
    mode='min',           # 'min' cerca il valore minimo della metrica monitorata (come la loss)
    min_lr=1e-6           # Limite inferiore del learning rate (non può andare sotto questo valore)
)

save_best_model = tf.keras.callbacks.ModelCheckpoint(
    # filepath=save_path+f'{s0}m_'+'{epoch:02d}'+f'_{yu}',
    filepath=f'weights_T{t_final}.h5',save_weights_only= True,save_best_only = True,monitor='loss',mode='min',verbose=0)

learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0,)

my_callbacks = [save_best_model]


EPOCHS       = 160000
neurons      = 32
#### Neural network architecture

input_layer   = Input(shape=(1,))
# branch1       = GaussianNoise(0.2)(input_layer)
branch1       = Dense(32, kernel_initializer=initializer,dtype=tf.float64)(input_layer)
branch1       = activation_sin(branch1)
branch1       = Dense(32, kernel_initializer=initializer,dtype=tf.float64)(branch1)
branch1       = activation_sin(branch1)
output_layer  = Dense( 1, kernel_initializer=initializer,dtype=tf.float64)(branch1)

# input_layer   = Input(shape=(1,))
# branch1       = Dense(neurons, kernel_initializer=initializer,dtype=tf.float64)(input_layer)
# branch1       = activation_tanh(branch1)
# branch1       = Dense(neurons, kernel_initializer=initializer,dtype=tf.float64)(input_layer)
# branch1       = activation_tanh(branch1)
# branch1       = Dense(1, kernel_initializer=initializer,dtype=tf.float64)(branch1)
# branch2       = Dense(neurons, kernel_initializer=initializer,dtype=tf.float64)(input_layer)
# branch2       = activation_sin(branch2)
# branch2       = Dense(neurons, kernel_initializer=initializer,dtype=tf.float64)(input_layer)
# branch2       = activation_sin(branch2)
# branch2       = Dense(1, kernel_initializer=initializer,dtype=tf.float64)(branch2)
# # branch3       = Dense(neurons, kernel_initializer=initializer,dtype=tf.float64)(input_layer)
# # branch3       = activation_cos(branch3)
# # branch3       = Dense(neurons, kernel_initializer=initializer,dtype=tf.float64)(branch3)
# # branch3       = activation_cos(branch3)
# # branch3       = Dense(1, kernel_initializer=initializer,dtype=tf.float64)(branch3)
# common        = Concatenate()([ branch1, branch2])
# output_layer  = Dense( 1, kernel_initializer=initializer,dtype=tf.float64)(common)


# branch1       = Dropout(rate=0.5)(branch1)
#### Compiling the model and training
print(f'First training with {EPOCHS} epochs ...')
model  = Model( inputs = input_layer , outputs = output_layer)
model.compile( loss=custom_loss(input_layer,output_layer), optimizer = opt )
model.save_weights('initial_weights.h5')
model.summary()
start_time = time.time()
history = model.fit( x = X_train, y = None, epochs=EPOCHS, verbose = 0, batch_size = int(len(X_train)),callbacks=[my_callbacks])
loss_1  = history.history['loss']
print(f'Elapsed time = {time.time()-start_time} sec = {(time.time()-start_time)/60} min')

for i in range(len(loss_1)):
  if np.isnan(loss_1[i]):
    break
print(i-1,loss_1[i-1])

#### Plot loss-function
plt.figure(figsize=(6,4))
plt.plot(loss_1,color='red')
plt.title('last = %4.3e' % (loss_1[-1]))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.grid(0.5,ls='dotted')
plt.savefig('loss.pdf',format='pdf')
plt.show()

#### Plot loss-function
plt.figure(figsize=(6,4))
plt.plot(loss_1[-4000:],color='red')
plt.title('last = %4.3e' % (loss_1[-1]))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.grid(0.5,ls='dotted')
plt.savefig('loss.pdf',format='pdf')
plt.show()

model.load_weights(f'weights_T{t_final}.h5') ##############################################


def get_prediction(time,model):
  prediction = (model.predict(time)).reshape(len(time))
  print(np.shape(prediction))
  rho = [rho_0 + (1-np.exp(-time[n]))* prediction[n] for n in range(len(prediction))]
  return np.array(rho)


def numerical_solution(time):
  # numerical integration with Runge-Kutta method
  method = 'BDF' if b != 1.9 else 'LSODA'
  sol    = solve_ivp(system, t_span=(time[0],time[-1]), y0=[rho_0,phi_0], t_eval=time,method=method)
  rho = sol.y[0]
  phi = sol.y[1]
  x   = rho*np.cos(phi)
  y   = rho*np.sin(phi)
  return rho, phi, x, y, method



t_post = np.linspace(t_start,100,5000)
rho_nn_post     = get_prediction(t_post,model)
rho_nn_training = get_prediction(t_eval,model)
rho_num_post, phi_num_post, x_num_post, y_num_post                 , method = numerical_solution(t_post)
rho_num_training, phi_num_training, x_num_training, y_num_training , method = numerical_solution(t_eval)


plt.figure(figsize=(8,6))
plt.plot(t_post, rho_num_post  ,  ls='-', color='blue',   label=method)
plt.plot(t_eval,rho_nn_training ,         color='red',    label='Neural network', marker='o',markersize=6,markerfacecolor='none')
plt.plot(t_post, rho_nn_post  , ls='-',   color='black')
plt.axhline(rho_crit,ls='--',color='blue')
plt.xlabel(r'$t$')
plt.ylabel(r'$\rho(t)$')
plt.grid(0.5,ls='dotted')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('rho.pdf',format='pdf')
plt.show()


i_start  = 15
i_finish = 25
i_start  = min(range(len(t_post)), key=lambda i: abs(t_post[i] - i_start))
i_finish = min(range(len(t_post)), key=lambda i: abs(t_post[i] - i_finish))

Fig, axs = plt.subplots(1,3,sharex=True,sharey=False,figsize=(12,6))
axs[0].plot(t_post[i_start:i_finish], rho_num_post[i_start:i_finish]  ,  ls='-', color='blue',   label=method)
axs[0].plot(t_post[i_start:i_finish],   rho_nn_post[i_start:i_finish], ls='-', color='black', label='neural network (post)')
# axs[0].plot(t_eval,   rho_nn_training, ls='-', color='red', label='neural network (post)')
axs[0].axhline(rho_crit,ls='--',color='orange')
axs[1].plot(t_post[i_start:i_finish], rho_num_post[i_start:i_finish]-rho_nn_post[i_start:i_finish], ls='-', color='black')
axs[2].plot(t_post[i_start:i_finish], rho_nn_post[i_start:i_finish]-rho_crit, ls='-', color='black')
axs[2].axhline(0,ls='--',color='orange')
for i in range(2):
  axs[i].grid(0.5,ls='dotted')
  axs[i].legend(loc='best')
  axs[i].set_xlabel(r'$t$')
axs[0].set_ylabel(r'$\rho(t)$')
plt.tight_layout()
plt.savefig('rho.pdf',format='pdf')
plt.show()

Fig, axs = plt.subplots(1,2,sharex=True,sharey=False,figsize=(12,6))
axs[0].plot(t_post, drho_dt_float(rho_num_post)  ,  ls='-', color='blue',    label=method)
axs[0].plot(t_eval, drho_dt_float(rho_nn_training) , ls='',  color='red',    label='Neural network', marker='o',markersize=6,markerfacecolor='none')
axs[1].plot(t_post, drho_dt_float(rho_num_post)  - drho_dt_float(rho_nn_post) ,  ls='-', color='black',    label='numerical - neural netwrok (post)')
axs[1].plot(t_eval, drho_dt_float(rho_num_training)  - drho_dt_float(rho_nn_training) ,  ls='-', color='red',    label='numerical - neural netwrok (training)')

axs[0].set_ylabel(r'$\frac{\mathrm{d}\rho(t)}{\mathrm{d}t}$')
axs[1].axhline(0,ls='--',color='steelblue')
for i in range(2):
  axs[i].grid(0.5,ls='dotted')
  axs[i].legend(loc='best')
  axs[i].set_xlabel(r'$t$')
plt.tight_layout()
plt.savefig('drho.pdf',format='pdf')
plt.show()
