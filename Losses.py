'''
Copyright (C) November 2024  Alessandro De Santis, alessandro.desantis@roma2.infn.it
'''

import tensorflow as tf
import numpy as np 

def f(t):
  return 1-tf.math.exp(-t)

def df_dt(t):
  return tf.math.exp(-t)


def Loss_PlanarGeometry(t,N,training,geometry,h,dh_dt,lambda_,lambda_relative):    
  # preconditioner applied to Prho(s)
  def loss(y_true, y_pred):
    ft  = f(t)
    dft = df_dt(t)
    ht  = h(t)
    dht = dh_dt(t)
    with tf.GradientTape() as tape:
      tape.watch(t)
      predictions = training.model(t,training=False)
    gradients = tape.batch_jacobian(predictions,t)        
    gradients = tf.reshape(gradients, (-1,geometry.Noutputs))
    rho       = geometry.rho_0   + ft[:,0]*   N[:,0]
    prho      = geometry.Prho_0  + ft[:,0]*(  N[:,1] + ht[:,0] )
    phi       = geometry.phi_0   + ft[:,0]*   N[:,2]
    rho_dot   = ft[:,0]*  gradients[:,0] + dft[:,0]             *N[:,0]
    prho_dot  = ft[:,0]* (gradients[:,1] + dht[:,0]) + dft[:,0]*(N[:,1] +ht[:,0])
    phi_dot   = ft[:,0]*  gradients[:,2] + dft[:,0]             *N[:,2]
    eom_rho    = tf.square(rho_dot  - geometry.RHO_DOT(rho,prho,phi))
    eom_Prho   = tf.square(prho_dot - geometry.PRHO_DOT(rho,prho,phi))
    eom_phi    = tf.square(phi_dot  - geometry.PHI_DOT( rho))
    energy     = tf.square(geometry.HAMILTONIAN(rho,prho,phi)-geometry.H0)
    total_loss  = (1-lambda_)*(eom_rho + eom_Prho + eom_phi) + lambda_*energy
    return tf.reduce_mean(total_loss)
  return loss




def Loss_SeparableGeometry(t,N,training,geometry,h,dh_dt,lambda_,lambda_relative):    
  # Preconditioner applied to rho(t)
  def loss(y_true, y_pred):

    ft  = f(t)
    dft = df_dt(t)
    ht  = h(t)
    dht = dh_dt(t)    
    with tf.GradientTape() as tape:
      tape.watch(t)
      predictions = training.model(t,training=False)
    gradients = tape.batch_jacobian(predictions,t)        
    gradients = tf.reshape(gradients, (-1,geometry.Noutputs))
    rho       = geometry.rho_0   + ft[:,0]*   (N[:,0] + ht[:,0] )
    phi       = geometry.phi_0   + ft[:,0]*   N[:,1]
    rho_dot   = ft[:,0]* (gradients[:,0] + dht[:,0]) + dft[:,0]*(N[:,0] +ht[:,0])
    phi_dot   = ft[:,0]*  gradients[:,1] + dft[:,0] *N[:,1]
    eom_rho   = tf.square( rho_dot  - geometry.RHO_DOT(rho))
    eom_phi   = tf.square( phi_dot  - geometry.PHI_DOT(rho))
    total_loss  = eom_rho  + eom_phi
    return tf.reduce_mean(total_loss)
  return loss



def Loss_NonlinearOscillator(t,N,training,geometry,h,dh_dt,lambda_,lambda_relative):    
  # Preconditioner applied to q(t)
  def loss(y_true, y_pred):
    ft  = f(t)
    dft = df_dt(t)
    ht  = h(t)
    dht = dh_dt(t)    
    with tf.GradientTape() as tape:
      tape.watch(t)
      predictions = training.model(t,training=False)
    gradients = tape.batch_jacobian(predictions,t)        
    gradients = tf.reshape(gradients, (-1,geometry.Noutputs))
    q       = geometry.q_0   + ft[:,0]*   (N[:,0] + ht[:,0] )
    p       = geometry.p_0   + ft[:,0]*   N[:,1]
    q_dot   = ft[:,0]* (gradients[:,0] + dht[:,0]) + dft[:,0]*(N[:,0] +ht[:,0])
    p_dot   = ft[:,0]*  gradients[:,1] + dft[:,0] *N[:,1]
    eom_q   = tf.square(q_dot  - geometry.Q_DOT(p))
    eom_p   = tf.square(p_dot  - geometry.P_DOT(q))
    energy  = tf.square(geometry.HAMILTONIAN(q,p)-geometry.H0)
    total_loss  = (1-lambda_)*(eom_q  + eom_p) + lambda_*energy
    return tf.reduce_mean(total_loss)
  return loss



def Loss_ChaoticOscillator(t,N,training,geometry,h,dh_dt,lambda_,lambda_relative):    
  # Preconditioner applied to x(t)
  def loss(y_true, y_pred):
    ft  = f(t)
    dft = df_dt(t)
    ht  = h(t)
    dht = dh_dt(t)    
    with tf.GradientTape() as tape:
      tape.watch(t)
      predictions = training.model(t,training=False)
    gradients = tape.batch_jacobian(predictions,t)        
    gradients = tf.reshape(gradients, (-1,geometry.Noutputs))
    x       = geometry.x_0   + ft[:,0]*   (N[:,0] + ht[:,0] )
    y       = geometry.y_0   + ft[:,0]*    N[:,1]
    px      = geometry.px_0  + ft[:,0]*    N[:,2]
    py      = geometry.py_0  + ft[:,0]*    N[:,3]
    x_dot   = ft[:,0]* (gradients[:,0] + dht[:,0]) + dft[:,0]*(N[:,0] +ht[:,0])
    y_dot   = ft[:,0]*  gradients[:,1] + dft[:,0] *N[:,1]
    px_dot  = ft[:,0]*  gradients[:,2] + dft[:,0] *N[:,2]
    py_dot  = ft[:,0]*  gradients[:,3] + dft[:,0] *N[:,3]
    eom_x   = tf.square(x_dot   - geometry.X_DOT(px))
    eom_y   = tf.square(y_dot   - geometry.Y_DOT(py))
    eom_px  = tf.square(px_dot  - geometry.PX_DOT(x,y))
    eom_py  = tf.square(py_dot  - geometry.PY_DOT(x,y))
    energy  = tf.square(geometry.HAMILTONIAN(x,y,px,py)-geometry.H0)
    total_loss  = (1-lambda_)*(eom_x + eom_y +eom_px + eom_py) + lambda_*energy
    return tf.reduce_mean(total_loss)
  return loss


def Loss_NonPlanarGeometry(t,N,training,geometry,h,dh_dt,lambda_,lambda_relative):    
  # preconditioner applied to Prho(s)
  def loss(y_true, y_pred):
    ft  = f(t)
    dft = df_dt(t)
    ht  = h(t)
    dht = dh_dt(t)
    with tf.GradientTape() as tape:
      tape.watch(t)
      predictions = training.model(t,training=False)
    gradients = tape.batch_jacobian(predictions,t)        
    gradients = tf.reshape(gradients, (-1,geometry.Noutputs))
    rho       = geometry.rho_0    + ft[:,0]*   N[:,0]
    Prho      = geometry.Prho_0   + ft[:,0]*(  N[:,1] + ht[:,0] )
    phi       = geometry.phi_0    + ft[:,0]*   N[:,2]
    theta     = geometry.theta_0  + ft[:,0]*   N[:,3]
    Ptheta    = geometry.Ptheta_0 + ft[:,0]*   N[:,4]
    rho_dot    = ft[:,0]*  gradients[:,0] + dft[:,0]             *N[:,0]
    Prho_dot   = ft[:,0]* (gradients[:,1] + dht[:,0]) + dft[:,0]*(N[:,1] +ht[:,0])
    phi_dot    = ft[:,0]*  gradients[:,2] + dft[:,0]             *N[:,2]
    theta_dot  = ft[:,0]*  gradients[:,3] + dft[:,0]             *N[:,3]
    Ptheta_dot = ft[:,0]*  gradients[:,4] + dft[:,0]             *N[:,4]
    eom_rho    = tf.square(rho_dot    - geometry.RHO_DOT(   rho,Prho,theta))
    eom_Prho   = tf.square(Prho_dot   - geometry.PRHO_DOT(  rho,Prho,theta,Ptheta))
    eom_phi    = tf.square(phi_dot    - geometry.PHI_DOT(   rho,theta))
    eom_theta  = tf.square(theta_dot  - geometry.THETA_DOT( rho,theta,Ptheta))
    eom_Ptheta = tf.square(Ptheta_dot - geometry.PTHETA_DOT(rho,Prho,theta,Ptheta))
    energy     = tf.square(geometry.HAMILTONIAN(rho,Prho,theta,Ptheta)-geometry.H0)
    total_loss  = (1-lambda_)*(eom_rho + eom_Prho + eom_phi + eom_theta + eom_Ptheta) + lambda_*energy
    return tf.reduce_mean(total_loss)
  return loss


def get_preconditioner(Training):

    if Training.preconditioner_type == 0:
      
      def h(t):
        return 0.0*t
      def dh_dt(t):
        return 0.0*t 

    if Training.preconditioner_type == 1:

        def h(t):
            n0 = -67298.58647951778
            n1 = 7295.752997399679
            n2 = 49.05846765119071
            n3 = -24.180054243227065
            n4 = 0.5688460244073691
            d0 = 16004.961902576088
            d1 = -125.93440777363304
            d2 = -76.17081709514477
            d3 = -6.5619313910448245
            d4 = 0.4975094845878163
            return - (n0+n1*t+n2*t**2+n3*t**3+n4*t**4)/(d0+d1*t+d2*t**2+d3*t**3+d4*t**4)

        def dh_dt(t):
            n0 = -67298.58647951778
            n1 = 7295.752997399679
            n2 = 49.05846765119071
            n3 = -24.180054243227065
            n4 = 0.5688460244073691
            d0 = 16004.961902576088
            d1 = -125.93440777363304
            d2 = -76.17081709514477
            d3 = -6.5619313910448245
            d4 = 0.4975094845878163
            return  -((n1 + 2 * n2 * t + 3 * n3 * t**2 + 4 * n4 * t**3) * (d0 + d1 * t + d2 * t**2 + d3 * t**3 + d4 * t**4) - (n0 + n1 * t + n2 * t**2 + n3 * t**3 + n4 * t**4) * (d1 + 2 * d2 * t + 3 * d3 * t**2 + 4 * d4 * t**3)) / (d0 + d1 * t + d2 * t**2 + d3 * t**3 + d4 * t**4)**2

    if Training.preconditioner_type == 2:

      def h(t):
        a = -5.2045382e+00
        b = 5.2045382e-06
        return a*t+b
      
      def dh_dt(t):
        a = -5.2045382e+00
        return a

    if Training.preconditioner_type == 3:

      def h(t):
        n0 = 4.77706330e+01
        n1 =-4.77848794e+04
        n2 = 1.42476331e+04 
        n3 =-1.25419766e+03
        n4 = 3.44501576e+01
        return n0 + n1*t + n2*t**2 + n3*t**3 + n4*t**4

      def dh_dt(t):
        n0 = 4.77706330e+01
        n1 =-4.77848794e+04
        n2 = 1.42476331e+04 
        n3 =-1.25419766e+03
        n4 = 3.44501576e+01
        return n1 + 2*n2*t + 3*n3*t**2 + 4*n4*t**3

    return h, dh_dt