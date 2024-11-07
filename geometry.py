'''
Copyright (C) November 2024  Alessandro De Santis, alessandro.desantis@roma2.infn.it
'''

import numpy 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from   scipy.integrate import solve_ivp
from   typing  import Any
from   pathlib import Path

class PlanarGeometry:

    def __init__(self,dict,*args,**kargs):

        print('Initialize planar geometry ... \n\n')

        self.Noutputs = 3
        self.exact_results = True
        self.af     = dict['af']     if 'af'      in dict.keys() else 0.1 
        self.b      = dict['b']      if 'b'       in dict.keys() else 1.8 
        self.L1     = dict['L1']     if 'L1'      in dict.keys() else 1
        self.L5     = dict['L5']     if 'L5'      in dict.keys() else 1
        self.E      = dict['E']      if 'E'       in dict.keys() else 1
        
        self.J_phi  = self.b*self.E
        self.rho_0  = dict['rho_0']  if 'rho_0'   in dict.keys() else 10
        self.phi_0  = dict['phi_0']  if 'phi_0'   in dict.keys() else np.arcsin(self.b/self.rho_0)
        self.Prho_0 = dict['Prho_0'] if 'Prho_0'  in dict.keys() else self.get_prho0()
        
        self.bc     = np.array([self.rho_0,self.Prho_0,self.phi_0])
        self.H0     = self.HAMILTONIAN_float(*self.bc)

        self.get_case()
        self.print_geometry()

    def get_case(self):

        if self.b == 1.9:
            self.case = 'Planar geometry, critical case'
        if self.b <  1.9:
            self.case = 'Planar geometry, sub-critical case'
        if self.b > 1.9:
            self.case = 'Planar geometry, over-critical case'

    def get_prho0(self):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        rho   = self.rho_0  
        J_phi = self.J_phi
        return -numpy.sqrt(E**2*L1**2*L5**2 + E**2*L1**2*af**2 + E**2*L1**2*rho**2 + E**2*L5**2*af**2 + E**2*L5**2*rho**2 + E**2*af**2*rho**2 + E**2*rho**4 - 2*E*J_phi*L1*L5*af - J_phi**2*rho**2)/(af**2 + rho**2)


    def print_geometry(self):

        print('<%s>\n' % (self.case))
        print(' af       =   %+16.15f' % (self.af))
        print(' b        =   %+16.15f' % (self.b))
        print(' L1       =   %+16.15f' % (self.L1))
        print(' L5       =   %+16.15f' % (self.L5))
        print(' E        =   %+16.15f' % (self.E))
        print(' J_phi    =   %+16.15f = b*E' % (self.J_phi))
        print(' rho_0    =   %+16.15f' % (self.rho_0))
        print(' phi_0    =   %+16.15f' % (self.phi_0))
        print(' Prho_0   =   %+16.15f' % (self.Prho_0))
        print(' H0       =   %+16.15f' % (self.H0))


    def write_to_file(self,fp):
        
        fp.write('\n[GEOMETRY] \n\n')
        fp.write('%s\n' % (self.case))
        fp.write('af       =   %+16.15f \n' % (self.af))
        fp.write('b        =   %+16.15f \n' % (self.b))
        fp.write('L1       =   %+16.15f \n' % (self.L1))
        fp.write('L5       =   %+16.15f \n' % (self.L5))
        fp.write('E        =   %+16.15f \n' % (self.E))
        fp.write('J_phi    =   %+16.15f = b*E\n' % (self.J_phi))
        fp.write('rho_0    =   %+16.15f\n' % (self.rho_0))
        fp.write('phi_0    =   %+16.15f\n' % (self.phi_0))
        fp.write('Prho_0   =   %+16.15f\n' % (self.Prho_0))
        fp.write('H0       =   %+16.15f\n' % (self.H0))
        fp.write('\n')

    def HAMILTONIAN_float(self,rho,Prho,phi):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi        
        return (1/2)*(E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**2*numpy.sqrt(L1**2/rho**2 + 1)*numpy.sqrt(L5**2/rho**2 + 1))

    def HAMILTONIAN(self,rho,Prho,phi):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi        
        return (1/2)*(E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**2*tf.math.sqrt(L1**2/rho**2 + 1)*tf.math.sqrt(L5**2/rho**2 + 1))

    def RHO_DOT_float(self,rho,Prho,phi):
        L1 = self.L1
        L5 = self.L5
        af = self.af
        return Prho*(af**2 + rho**2)/(rho**2*numpy.sqrt(L1**2/rho**2 + 1)*numpy.sqrt(L5**2/rho**2 + 1))

    def RHO_DOT(self,rho,Prho,phi):
        L1 = self.L1
        L5 = self.L5
        af = self.af
        return Prho*(af**2 + rho**2)/(rho**2*tf.math.sqrt(L1**2/rho**2 + 1)*tf.math.sqrt(L5**2/rho**2 + 1))

    def PRHO_DOT_float(self,rho,Prho):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi  
        return -  ((1/2)*L1**2*(E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**5*(L1**2/rho**2 + 1)**(3/2)*numpy.sqrt(L5**2/rho**2 + 1)) + (1/2)*L5**2*(E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**5*numpy.sqrt(L1**2/rho**2 + 1)*(L5**2/rho**2 + 1)**(3/2)) + (1/2)*(-2*E**2*rho + 2*Prho**2*rho + 2*rho*(-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2)**2)/(rho**2*numpy.sqrt(L1**2/rho**2 + 1)*numpy.sqrt(L5**2/rho**2 + 1)) - (E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**3*numpy.sqrt(L1**2/rho**2 + 1)*numpy.sqrt(L5**2/rho**2 + 1))  )

    def PRHO_DOT(self,rho,Prho,phi):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi  
        return -  ((1/2)*L1**2*(E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**5*(L1**2/rho**2 + 1)**(3/2)*tf.math.sqrt(L5**2/rho**2 + 1)) + (1/2)*L5**2*(E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**5*tf.math.sqrt(L1**2/rho**2 + 1)*(L5**2/rho**2 + 1)**(3/2)) + (1/2)*(-2*E**2*rho + 2*Prho**2*rho + 2*rho*(-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2)**2)/(rho**2*tf.math.sqrt(L1**2/rho**2 + 1)*tf.math.sqrt(L5**2/rho**2 + 1)) - (E**2*af**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2 + Prho**2*(af**2 + rho**2) - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(rho**3*tf.math.sqrt(L1**2/rho**2 + 1)*tf.math.sqrt(L5**2/rho**2 + 1))  )

    def PHI_DOT_float(self,rho,phi):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi           
        return (1/2)*(2*J_phi - 2*af*(-E*L1*L5 + J_phi*af)/(af**2 + rho**2))/(rho**2*numpy.sqrt(L1**2/rho**2 + 1)*numpy.sqrt(L5**2/rho**2 + 1))

    def PHI_DOT(self,rho):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi           
        return (1/2)*(2*J_phi - 2*af*(-E*L1*L5 + J_phi*af)/(af**2 + rho**2))/(rho**2*tf.math.sqrt(L1**2/rho**2 + 1)*tf.math.sqrt(L5**2/rho**2 + 1))


    def Predict(self,Training,time_range,input: Any = None):

        prediction = Training.model.predict(time_range)
        prediction = np.transpose(prediction)
        N0 = prediction[0]
        N1 = prediction[1]
        N2 = prediction[2]

        rho   = np.array([])
        Prho  = np.array([])
        phi   = np.array([])
        for n in range(len(time_range)):
            rho   = np.append(rho,  self.rho_0   + (1-np.exp(-time_range[n]))* N0[n])
            Prho  = np.append(Prho, self.Prho_0  + (1-np.exp(-time_range[n]))*(N1[n] + Training.h(time_range[n])))
            phi   = np.append(phi,  self.phi_0   + (1-np.exp(-time_range[n]))* N2[n])

        H      = [self.HAMILTONIAN_float(rho[i],Prho[i],phi[i]) for i in range(len(rho))]
        x      = rho*np.cos(phi)
        y      = rho*np.sin(phi)


        if input != None:
            
            fp = open(input.destination_path / Path(f'Prediction.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  s      \n')
            fp.write('# $3  H(s)   \n')
            fp.write('# $4  H(0)   \n')
            fp.write('# $5  rho(s) \n')
            fp.write('# $6  Prho(s)\n')
            fp.write('# $7  phi(s) \n')
            fp.write('# $8  x(s)   \n')
            fp.write('# $9  y(s)   \n')
            fp.write('# $10 N_rho(s)   \n')
            fp.write('# $11 N_Prho(s)  \n')
            fp.write('# $12 N_phi(s)   \n')

            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e      %+12.10e      %+12.10e      %+12.10e' % (rho[i],Prho[i],phi[i],x[i],y[i]))
                fp.write('      %12.10e      %+12.10e      %+12.10e' % (N0[i],N1[i],N2[i]))
                fp.write('\n')
            fp.close()

    def Numerical_integration(self,time_range,method: str = 'Radau',input: Any = None):

        def system(s,y):

            rho, prho, phi = y
            drho_ds   = self.RHO_DOT_float(rho,prho)
            dprho_ds  = self.PRHO_DOT_float(rho,prho)
            dphi_ds   = self.PHI_DOT_float(rho)
            return [drho_ds,dprho_ds, dphi_ds]
        
        sol     = solve_ivp(system, t_span=(time_range[0],time_range[-1]), y0=self.bc, t_eval=time_range,method=method)
        rho    = sol.y[0]
        Prho   = sol.y[1]
        phi    = sol.y[2]
        H      = [self.HAMILTONIAN_float(rho[i],Prho[i],phi[i]) for i in range(len(rho))]
        x      = rho*np.cos(phi)
        y      = rho*np.sin(phi)

        if input != None:
            
            fp = open(input.destination_path / Path(f'results_{method}.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  s      \n')
            fp.write('# $3  H(s)   \n')
            fp.write('# $4  H(0)   \n')
            fp.write('# $5  rho(s) \n')
            fp.write('# $6  Prho(s)\n')
            fp.write('# $7  phi(s) \n')
            fp.write('# $8  x(s)   \n')
            fp.write('# $9  y(s)   \n')
            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e      %+12.10e      %+12.10e      %+12.10e\n' % (rho[i],Prho[i],phi[i],x[i],y[i]))
            fp.close()

    def gnuplot_scripts(self,input,training,methods):
        
        fp = open(input.destination_path /  Path('plot_loss.gp'),'w+')
        fp.write('set xlabel "Log(epochs)" font ",14"\n')
        fp.write('set ylabel "Log(Loss)" font ",14"\n')
        fp.write('set grid\n')
        fp.write('set logscale y\n')
        fp.write('set logscale x\n')
        fp.write('set title "%s, b=%4.3f" font ",14"\n' % (self.case,self.b))
        fp.write('  plot "loss.dat" every 2 u 1:2 w l lc "black" title "(1-{/Symbol l})L^{dyn} + {/Symbol l}L^{E}, {/Symbol l} = %4.3f" \n' % (training.lambda_))
        fp.write('replot "loss.dat" every 2 u 1:3 w l lc "red"  title "L^{E}\n')
        fp.write('replot "loss.dat" every 2 u 1:4 w l lc "blue" title "L^{dyn}\n')
        fp.close()

        fp = open(input.destination_path /  Path('plot_energy.gp'),'w+')
        fp.write('set xlabel "s" font ",14"\n')
        fp.write('set ylabel "H(s)" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 2:3 w l lc "red" title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 2:3 w l title "method %s"\n' % (method,method))
        fp.write('replot %6.5f lc "black" title "H(0)"\n' % (self.H0))
        fp.close()

        fp = open(input.destination_path /  Path('plot_cartesian_results.gp'),'w+')
        fp.write('set xlabel "x" font ",14"\n')
        fp.write('set ylabel "y" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 8:9 w l lc "red" title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 8:9 w l title "method %s"\n' % (method,method))
        fp.close()







class SeparableGeometry:

    def __init__(self,dict,*args,**kargs):

        print('Initialize Separable (planar) geometry ... \n\n')

        self.Noutputs = 2
        self.exact_results = True

        self.af     = dict['af']     if 'af'      in dict.keys() else 0.1 
        self.b      = dict['b']      if 'b'       in dict.keys() else 1.8 
        self.L1     = dict['L1']     if 'L1'      in dict.keys() else 1
        self.L5     = dict['L5']     if 'L5'      in dict.keys() else 1
        self.E      = dict['E']      if 'E'       in dict.keys() else 1
        self.rho_0  = dict['rho_0']  if 'rho_0'   in dict.keys() else 10
        self.J_phi  = self.b*self.E 
        self.phi_0  = np.arcsin(self.b/self.rho_0)        
        self.bc     = np.array([self.rho_0,self.phi_0])
        self.L      = self.L1

        self.get_case()
        self.print_geometry()

    def get_case(self):

        if self.b == 1.9:
            self.case = 'Separable (planar) geometry, critical case'
        if self.b <  1.9:
            self.case = 'Separable (planar) geometry, sub-critical case'
        if self.b > 1.9:
            self.case = 'Separable (planar) geometry, over-critical case'

    def print_geometry(self):

        print('<%s>\n' % (self.case))
        print(' af       =   %+16.15f' % (self.af))
        print(' b        =   %+16.15f' % (self.b))
        print(' L1       =   %+16.15f' % (self.L1))
        print(' L5       =   %+16.15f' % (self.L5))
        print(' L        =   %+16.15f' % (self.L))
        print(' E        =   %+16.15f' % (self.E))
        print(' J_phi    =   %+16.15f = b*E' % (self.J_phi))
        print(' rho_0    =   %+16.15f' % (self.rho_0))
        print(' phi_0    =   %+16.15f' % (self.phi_0))

    def write_to_file(self,fp):
        
        fp.write('\n[GEOMETRY] \n\n')
        fp.write('%s\n' % (self.case))
        fp.write('af       =   %+16.15f \n' % (self.af))
        fp.write('b        =   %+16.15f \n' % (self.b))
        fp.write('L1       =   %+16.15f \n' % (self.L1))
        fp.write('L5       =   %+16.15f \n' % (self.L5))
        fp.write('L        =   %+16.15f \n' % (self.L))
        fp.write('E        =   %+16.15f \n' % (self.E))
        fp.write('J_phi    =   %+16.15f = b*E\n' % (self.J_phi))
        fp.write('rho_0    =   %+16.15f\n' % (self.rho_0))
        fp.write('phi_0    =   %+16.15f\n' % (self.phi_0))
        fp.write('\n')

    def RHO_DOT_float(self,rho):
        L  = self.L
        af = self.af
        b  = self.b
        return    -(rho**2+af**2)*np.sqrt(  (rho**2+L**2)**2+(af-b)*(rho**2 *(af+b)+2*af*L**2)    )/(af**2*(rho**2+2*L*L)+(rho**2+L**2)**2-af*b*L*L  )

    def RHO_DOT(self,rho):
        L  = self.L
        af = self.af
        b  = self.b
        return    -(rho**2+af**2)*tf.math.sqrt(  (rho**2+L**2)**2+(af-b)*(rho**2 *(af+b)+2*af*L**2)    )/(af**2*(rho**2+2*L*L)+(rho**2+L**2)**2-af*b*L*L  )

    def PHI_DOT_float(self,rho):
        L     = self.L   
        af    = self.af    
        b     = self.b
        return (b*rho*rho+af*L*L)/((rho**2+L**2)**2+(af**2)*(rho**2+2*L*L)-af*b*L*L)

    def PHI_DOT(self,rho):
        L     = self.L   
        af    = self.af    
        b     = self.b
        return (b*rho*rho+af*L*L)/((rho**2+L**2)**2+(af**2)*(rho**2+2*L*L)-af*b*L*L)

    def Predict(self,Training,time_range,input: Any = None):

        prediction = Training.model.predict(time_range)
        prediction = np.transpose(prediction)
        N0 = prediction[0]
        N1 = prediction[1]

        rho   = np.array([])
        phi   = np.array([])
        for n in range(len(time_range)):
            rho   = np.append(rho,  self.rho_0   + (1-np.exp(-time_range[n]))* (N0[n] + Training.h(time_range[n])))
            phi   = np.append(phi,  self.phi_0   + (1-np.exp(-time_range[n]))* N1[n])

        x      = rho*np.cos(phi)
        y      = rho*np.sin(phi)

        if input != None:
            
            fp = open(input.destination_path / Path(f'Prediction.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  t      \n')
            fp.write('# $3  rho(t) \n')
            fp.write('# $4  phi(t) \n')
            fp.write('# $5  x(t)   \n')
            fp.write('# $6  y(t)   \n')
            fp.write('# $7  N_rho(t)   \n')
            fp.write('# $8  N_phi(t)   \n')

            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e' % (i+1,time_range[i]))
                fp.write('      %12.10e      %+12.10e      %+12.10e      %+12.10e' % (rho[i],phi[i],x[i],y[i]))
                fp.write('      %12.10e      %+12.10e' % (N0[i],N1[i]))
                fp.write('\n')
            fp.close()

    def Numerical_integration(self,time_range,method: str = 'Radau',input: Any = None):

        def system(s,y):

            rho, phi = y
            drho_ds   = self.RHO_DOT_float(rho)
            dphi_ds   = self.PHI_DOT_float(rho)
            return [drho_ds, dphi_ds]
        
        sol     = solve_ivp(system, t_span=(time_range[0],time_range[-1]), y0=self.bc, t_eval=time_range,method=method)
        rho    = sol.y[0]
        phi    = sol.y[1]
        x      = rho*np.cos(phi)
        y      = rho*np.sin(phi)

        if input != None:
            
            fp = open(input.destination_path / Path(f'results_{method}.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  t      \n')
            fp.write('# $3  rho(t) \n')
            fp.write('# $4  phi(t) \n')
            fp.write('# $5  x(t)   \n')
            fp.write('# $6  y(t)   \n')
            fp.write('# $7  N_rho(t)   \n')
            fp.write('# $8  N_phi(t)   \n')

            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e' % (i+1,time_range[i]))
                fp.write('      %12.10e      %+12.10e      %+12.10e      %+12.10e' % (rho[i],phi[i],x[i],y[i]))
                fp.write('\n')
            fp.close()

    def gnuplot_scripts(self,input,training,methods):
        
        fp = open(input.destination_path /  Path('plot_loss.gp'),'w+')
        fp.write('set xlabel "Log(epochs)" font ",14"\n')
        fp.write('set ylabel "Log(Loss)" font ",14"\n')
        fp.write('set grid\n')
        fp.write('set logscale y\n')
        fp.write('set logscale x\n')
        fp.write('set title "%s, b=%4.3f" font ",14"\n' % (self.case,self.b))
        fp.write('  plot "loss.dat" every 2 u 1:2 w l lc "black" title "L^{dyn}" \n')
        fp.close()

        fp = open(input.destination_path /  Path('plot_cartesian_results.gp'),'w+')
        fp.write('set xlabel "x" font ",14"\n')
        fp.write('set ylabel "y" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 5:6 w l lc "red" title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 5:6 w l title "method %s"\n' % (method,method))
        fp.close()




class NonlinearOscillator:

    def __init__(self,dict,*args,**kargs):

        print('Initialize Non-linear unidimensional oscillator ... \n\n')

        self.Noutputs = 2
        self.exact_results = False

        self.q_0     = dict['q_0']     if 'q_0'      in dict.keys() else 1.3
        self.p_0     = dict['p_0']     if 'p_0'      in dict.keys() else 1.0
        self.bc      = np.array([self.q_0,self.p_0])
        self.H0     = self.HAMILTONIAN(*self.bc)
        self.case    = 'Non-linear unidimensional oscillator'
        self.print_geometry()

    def print_geometry(self):

        print('<%s>\n' % (self.case))
        print(' q_0  =   %+16.15f' % (self.q_0))
        print(' p_0  =   %+16.15f' % (self.p_0))
        print(' H_0  =   %+16.15f' % (self.H0))

    def write_to_file(self,fp):
        
        fp.write('\n[GEOMETRY] \n\n')
        fp.write('%s\n' % (self.case))
        fp.write('q_0  =   %+16.15f\n' % (self.q_0))
        fp.write('p_0  =   %+16.15f\n' % (self.p_0))
        fp.write('H_0  =   %+16.15f\n' % (self.H0))
        fp.write('\n')

    def HAMILTONIAN(self,q,p):
        return p*p/2 + q*q/2  + q*q*q*q/4

    def Q_DOT(self,p):
        return p

    def P_DOT(self,q):
        return -q -q*q*q

    def Predict(self,Training,time_range,input: Any = None):

        prediction = Training.model.predict(time_range)
        prediction = np.transpose(prediction)
        N0 = prediction[0]
        N1 = prediction[1]

        q   = np.array([])
        p   = np.array([])
        for n in range(len(time_range)):
            q   = np.append(q,  self.q_0   + (1-np.exp(-time_range[n]))* (N0[n]+ Training.h(time_range[n])))
            p   = np.append(p,  self.p_0   + (1-np.exp(-time_range[n]))* N1[n])
        H       = [self.HAMILTONIAN(q[i],p[i]) for i in range(len(q))]

        if input != None:
            
            fp = open(input.destination_path / Path(f'Prediction.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  t      \n')
            fp.write('# $3  H(t)   \n')
            fp.write('# $4  H(0)   \n')
            fp.write('# $5  q(t)   \n')
            fp.write('# $6  p(t)   \n')
            fp.write('# $7  N_q(t) \n')
            fp.write('# $8  N_p(t) \n')
            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e' % (q[i],p[i]))
                fp.write('      %12.10e      %+12.10e' % (N0[i],N1[i]))
                fp.write('\n')
            fp.close()




    def Numerical_integration(self,time_range,method: str = 'Radau',input: Any = None):

        def system(s,y):
            q, p = y
            dq_dt   = self.Q_DOT(p)
            dp_dt   = self.P_DOT(q)
            return [dq_dt,dp_dt]
        
        sol   = solve_ivp(system, t_span=(time_range[0],time_range[-1]), y0=self.bc, t_eval=time_range,method=method)
        q     = sol.y[0]
        p     = sol.y[1]
        H     = [self.HAMILTONIAN(q[i],p[i]) for i in range(len(q))]

        if input != None:
            
            fp = open(input.destination_path / Path(f'results_{method}.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  t      \n')
            fp.write('# $3  H(t)   \n')
            fp.write('# $4  H(0)   \n')
            fp.write('# $5  q(t)   \n')
            fp.write('# $6  p(t)   \n')
            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e' % (q[i],p[i]))
                fp.write('\n')
            fp.close()

    def gnuplot_scripts(self,input,training,methods):
        
        fp = open(input.destination_path /  Path('plot_loss.gp'),'w+')
        fp.write('set xlabel "Log(epochs)" font ",14"\n')
        fp.write('set ylabel "Log(Loss)" font ",14"\n')
        fp.write('set grid\n')
        fp.write('set logscale y\n')
        fp.write('set logscale x\n')
        fp.write('set title "%s" font ",14"\n' % (self.case))
        fp.write('  plot "loss.dat" every 2 u 1:2 w l lc "black" title "(1-{/Symbol l})L^{dyn} + {/Symbol l}L^{E}, {/Symbol l} = %4.3f" \n' % (training.lambda_))
        fp.write('replot "loss.dat" every 2 u 1:3 w l lc "red"  title "L^{E}\n')
        fp.write('replot "loss.dat" every 2 u 1:4 w l lc "blue" title "L^{dyn}\n')
        fp.close()

        fp = open(input.destination_path /  Path('plot_energy.gp'),'w+')
        fp.write('set xlabel "s" font ",14"\n')
        fp.write('set ylabel "H(s)" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 2:3 w l lc "red" title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 2:3 w l title "method %s"\n' % (method,method))
        fp.write('replot %6.5f lc "black" title "H(0)"\n' % (self.H0))
        fp.close()

        fp = open(input.destination_path /  Path('plot_results1.gp'),'w+')
        fp.write('set xlabel "t" font ",14"\n')
        fp.write('set ylabel "results" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 2:5 w l lc "red"  title "Prediction q(t)"\n')
        fp.write('replot "Prediction.dat" u 2:6 w l lc "blue" title "Prediction p(t)"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 2:5 w l title "method %s q(t)"\n' % (method,method))
            fp.write('replot "results_%s.dat" u 2:6 w l title "method %s p(t)"\n' % (method,method))
        fp.close()

        fp = open(input.destination_path /  Path('plot_results2.gp'),'w+')
        fp.write('set xlabel "q" font ",14"\n')
        fp.write('set ylabel "p" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 5:6 w l lc "red"  title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 5:6 w l title "method %s"\n' % (method,method))
        fp.close()




class ChaoticOscillator:

    def __init__(self,dict,*args,**kargs):

        print('Initialize chaotic bidimensional oscillator ... \n\n')
        self.Noutputs = 4
        self.exact_results = False

        self.x_0  = dict['x_0']  if 'x_0'  in dict.keys() else  0.3
        self.y_0  = dict['y_0']  if 'y_0'  in dict.keys() else -0.3
        self.px_0 = dict['px_0'] if 'px_0' in dict.keys() else  0.3
        self.py_0 = dict['py_0'] if 'py_0' in dict.keys() else  0.15

        self.bc     = np.array([self.x_0,self.y_0,self.px_0,self.py_0])
        self.H0     = self.HAMILTONIAN(*self.bc)
        self.case   = 'chaotic oscillator'
        self.print_geometry()

    def print_geometry(self):

        print('<%s>\n' % (self.case))
        print(' x_0   =   %+16.15f' % (self.x_0))
        print(' y_0   =   %+16.15f' % (self.y_0))
        print(' px_0  =   %+16.15f' % (self.px_0))
        print(' py_0  =   %+16.15f' % (self.py_0))
        print(' H_0   =   %+16.15f' % (self.H0))

    def write_to_file(self,fp):
        
        fp.write('\n[GEOMETRY] \n\n')
        fp.write('%s\n' % (self.case))
        fp.write('x_0   =   %+16.15f\n' % (self.x_0))
        fp.write('y_0   =   %+16.15f\n' % (self.y_0))
        fp.write('px_0  =   %+16.15f\n' % (self.px_0))
        fp.write('py_0  =   %+16.15f\n' % (self.py_0))
        fp.write('H_0   =   %+16.15f\n' % (self.H0))
        fp.write('\n')

    def HAMILTONIAN(self,x,y,px,py):
        return 0.5*(px*px+py*py) + 0.5*(x*x+y*y) + (x*x*y-y*y*y/3)

    def X_DOT(self,px):
        return px

    def Y_DOT(self,py):
        return py
    
    def PX_DOT(self,x,y):
        return -(x+2*x*y)

    def PY_DOT(self,x,y):
        return -(y+x*x-y*y)


    def Predict(self,Training,time_range,input: Any = None):

        prediction = Training.model.predict(time_range)
        prediction = np.transpose(prediction)
        N0 = prediction[0]
        N1 = prediction[1]
        N2 = prediction[2]
        N3 = prediction[3]

        x   = np.array([])
        y   = np.array([])
        px   = np.array([])
        py   = np.array([])

        for n in range(len(time_range)):
            x   = np.append(x,  self.x_0   + (1-np.exp(-time_range[n]))* (N0[n]+ Training.h(time_range[n])))
            y   = np.append(y,  self.y_0   + (1-np.exp(-time_range[n]))* N1[n])
            px   = np.append(px,  self.px_0   + (1-np.exp(-time_range[n]))* N2[n])
            py   = np.append(py,  self.py_0   + (1-np.exp(-time_range[n]))* N3[n])

        H       = [self.HAMILTONIAN(x[i],y[i],px[i],py[i]) for i in range(len(x))]

        if input != None:
            
            fp = open(input.destination_path / Path(f'Prediction.dat'),'w+')
            fp.write('# $1   n      \n')
            fp.write('# $2   t      \n')
            fp.write('# $3   H(t)   \n')
            fp.write('# $4   H(0)   \n')
            fp.write('# $5   x(t)   \n')
            fp.write('# $6   y(t)   \n')
            fp.write('# $7   px(t)   \n')
            fp.write('# $8   py(t)   \n')
            fp.write('# $9   N_x(t) \n')
            fp.write('# $10  N_y(t) \n')
            fp.write('# $11  N_px(t) \n')
            fp.write('# $12  N_py(t) \n')

            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e      %12.10e      %+12.10e' % (x[i],y[i],px[i],py[i]))
                fp.write('      %12.10e      %+12.10e      %12.10e      %+12.10e' % (N0[i],N1[i],N2[i],N3[i]))
                fp.write('\n')
            fp.close()




    def Numerical_integration(self,time_range,method: str = 'Radau',input: Any = None):

        def system(s,z):
            x, y, px, py = z
            dx_dt   = self.X_DOT(px)
            dy_dt   = self.Y_DOT(py)
            dpx_dt  = self.PX_DOT(x,y)
            dpy_dt  = self.PY_DOT(x,y)
            return [dx_dt,dy_dt,dpx_dt,dpy_dt]
        
        sol   = solve_ivp(system, t_span=(time_range[0],time_range[-1]), y0=self.bc, t_eval=time_range,method=method)
        x     = sol.y[0]
        y     = sol.y[1]
        px    = sol.y[2]
        py    = sol.y[3]
        H       = [self.HAMILTONIAN(x[i],y[i],px[i],py[i]) for i in range(len(x))]

        if input != None:
            
            fp = open(input.destination_path / Path(f'results_{method}.dat'),'w+')
            fp.write('# $1   n      \n')
            fp.write('# $2   t      \n')
            fp.write('# $3   H(t)   \n')
            fp.write('# $4   H(0)   \n')
            fp.write('# $5   x(t)   \n')
            fp.write('# $6   y(t)   \n')
            fp.write('# $7   px(t)   \n')
            fp.write('# $8   py(t)   \n')

            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e      %12.10e      %+12.10e' % (x[i],y[i],px[i],py[i]))
                fp.write('\n')
            fp.close()

    def gnuplot_scripts(self,input,training,methods):
        
        fp = open(input.destination_path /  Path('plot_loss.gp'),'w+')
        fp.write('set xlabel "Log(epochs)" font ",14"\n')
        fp.write('set ylabel "Log(Loss)" font ",14"\n')
        fp.write('set grid\n')
        fp.write('set logscale y\n')
        fp.write('set logscale x\n')
        fp.write('set title "%s" font ",14"\n' % (self.case))
        fp.write('  plot "loss.dat" every 2 u 1:2 w l lc "black" title "(1-{/Symbol l})L^{dyn} + {/Symbol l}L^{E}, {/Symbol l} = %4.3f" \n' % (training.lambda_))
        fp.write('replot "loss.dat" every 2 u 1:3 w l lc "red"  title "L^{E}\n')
        fp.write('replot "loss.dat" every 2 u 1:4 w l lc "blue" title "L^{dyn}\n')
        fp.close()

        fp = open(input.destination_path /  Path('plot_energy.gp'),'w+')
        fp.write('set xlabel "s" font ",14"\n')
        fp.write('set ylabel "H(s)" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 2:3 w l lc "red" title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 2:3 w l title "method %s"\n' % (method,method))
        fp.write('replot %6.5f lc "black" title "H(0)"\n' % (self.H0))
        fp.close()

        fp = open(input.destination_path /  Path('plot_cartesian_results.gp'),'w+')
        fp.write('set xlabel "x" font ",14"\n')
        fp.write('set ylabel "y" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 5:6 w l lc "red"  title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 5:6 w l title "method %s"\n' % (method,method))
        fp.close()



class NonPlanarGeometry:

    def __init__(self,dict,*args,**kargs):

        print('Initialize non-planar geometry ... \n\n')

        self.Noutputs = 5
        self.exact_results = False

        self.af     = dict['af']     if 'af'      in dict.keys() else 0.1 
        self.b      = dict['b']      if 'b'       in dict.keys() else 1.8 
        self.bz     = dict['bz']      if 'bz'       in dict.keys() else 1.8 
        self.L1     = dict['L1']     if 'L1'      in dict.keys() else 1
        self.L5     = dict['L5']     if 'L5'      in dict.keys() else 1
        self.E      = dict['E']      if 'E'       in dict.keys() else 1
        self.rho_0  = dict['rho_0']  if 'rho_0'   in dict.keys() else 10
        self.Ptheta_0 = dict['Ptheta_0'] if 'Ptheta_0' in dict.keys() else 0

        self.theta_0  = np.arccos(self.bz/self.rho_0)
        self.phi_0    = np.arcsin(self.b/(self.rho_0*np.sin(self.theta_0)))
        self.J_phi    = self.b*self.E 
        self.Prho_0   = self.get_prho0()
        
        self.bc     = np.array([self.rho_0,self.Prho_0,self.phi_0,self.theta_0,self.Ptheta_0])
        self.H0     = self.HAMILTONIAN_float(*self.bc)

        self.case   = 'Non-planar Geometry'

        self.print_geometry()

    def get_prho0(self):
        E      = self.E     
        L1     = self.L1   
        L5     = self.L5   
        af     = self.af    
        rho    = self.rho_0
        theta  = self.theta_0
        Ptheta = self.Ptheta_0  
        J_phi  = self.J_phi
        return - np.sqrt( (1/(rho**2+af**2)) * ( (J_phi*af-E*L1*L5)**2/(rho**2+af**2) +E**2*(rho**2+af**2+L1**2+L5**2) -E**2*af**2*np.sin(theta)**2-J_phi**2/(np.sin(theta)**2) - Ptheta**2 ))


    def print_geometry(self):

        print('<%s>\n' % (self.case))
        print(' af       =   %+16.15f' % (self.af))
        print(' b        =   %+16.15f' % (self.b))
        print(' L1       =   %+16.15f' % (self.L1))
        print(' L5       =   %+16.15f' % (self.L5))
        print(' E        =   %+16.15f' % (self.E))
        print(' J_phi    =   %+16.15f = b*E' % (self.J_phi))
        print(' rho_0    =   %+16.15f' % (self.rho_0))
        print(' Prho_0   =   %+16.15f' % (self.Prho_0))
        print(' phi_0    =   %+16.15f' % (self.phi_0))
        print(' theta_0  =   %+16.15f' % (self.theta_0))
        print(' Ptheta_0 =   %+16.15f' % (self.Ptheta_0))
        print(' H0       =   %+16.15f' % (self.H0))


    def write_to_file(self,fp):
        
        fp.write('\n[GEOMETRY] \n\n')
        fp.write('%s\n' % (self.case))
        fp.write('af       =   %+16.15f \n' % (self.af))
        fp.write('b        =   %+16.15f \n' % (self.b))
        fp.write('L1       =   %+16.15f \n' % (self.L1))
        fp.write('L5       =   %+16.15f \n' % (self.L5))
        fp.write('E        =   %+16.15f \n' % (self.E))
        fp.write('J_phi    =   %+16.15f = b*E\n' % (self.J_phi))
        fp.write('rho_0    =   %+16.15f\n' % (self.rho_0))
        fp.write('Prho_0   =   %+16.15f\n' % (self.Prho_0))
        fp.write('phi_0    =   %+16.15f\n' % (self.phi_0))
        fp.write('theta_0  =   %+16.15f\n' % (self.theta_0))
        fp.write('Ptheta_0 =   %+16.15f\n' % (self.Ptheta_0))
        fp.write('H0       =   %+16.15f\n' % (self.H0))
        fp.write('\n')

    def HAMILTONIAN_float(self,rho,Prho,phi,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi        
        return (E**2*af**2*numpy.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/numpy.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2))

    def HAMILTONIAN(self,rho,Prho,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi        
        return (E**2*af**2*tf.math.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/tf.math.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2))

    def RHO_DOT_float(self,rho,Prho,phi,theta,Ptheta):
        L1 = self.L1
        L5 = self.L5
        af = self.af
        return 2*Prho*(af**2 + rho**2)/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2))

    def RHO_DOT(self,rho,Prho,theta):
        L1 = self.L1
        L5 = self.L5
        af = self.af
        return 2*Prho*(af**2 + rho**2)/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2))

    def PHI_DOT_float(self,rho,Prho,phi,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi           
        return (2*J_phi/numpy.sin(theta)**2 - 2*af*(-E*L1*L5 + J_phi*af)/(af**2 + rho**2))/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2))

    def PHI_DOT(self,rho,theta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi           
        return (2*J_phi/tf.math.sin(theta)**2 - 2*af*(-E*L1*L5 + J_phi*af)/(af**2 + rho**2))/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2))

    def THETA_DOT_float(self,rho,Prho,phi,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi          
        return 2*Ptheta/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2))

    def THETA_DOT(self,rho,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi          
        return 2*Ptheta/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2))

    def PTHETA_DOT_float(self,rho,Prho,phi,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi         
        return 2*Ptheta/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2))-L1**2*af**2*(E**2*af**2*numpy.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/numpy.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))*numpy.sin(theta)*numpy.cos(theta)/((L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)**(3/2)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(af**2*numpy.cos(theta)**2 + rho**2)**2*(2*af**2*numpy.cos(theta)**2 + 2*rho**2)) - L5**2*af**2*(E**2*af**2*numpy.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/numpy.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))*numpy.sin(theta)*numpy.cos(theta)/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)**(3/2)*(af**2*numpy.cos(theta)**2 + rho**2)**2*(2*af**2*numpy.cos(theta)**2 + 2*rho**2)) + 4*af**2*(E**2*af**2*numpy.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/numpy.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))*numpy.sin(theta)*numpy.cos(theta)/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2)**2) + (2*E**2*af**2*numpy.sin(theta)*numpy.cos(theta) - 2*J_phi**2*numpy.cos(theta)/numpy.sin(theta)**3)/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2))

    def PTHETA_DOT(self,rho,Prho,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi         
        return 2*Ptheta/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2))-L1**2*af**2*(E**2*af**2*tf.math.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/tf.math.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))*tf.math.sin(theta)*tf.math.cos(theta)/((L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)**(3/2)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(af**2*tf.math.cos(theta)**2 + rho**2)**2*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2)) - L5**2*af**2*(E**2*af**2*tf.math.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/tf.math.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))*tf.math.sin(theta)*tf.math.cos(theta)/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)**(3/2)*(af**2*tf.math.cos(theta)**2 + rho**2)**2*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2)) + 4*af**2*(E**2*af**2*tf.math.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/tf.math.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))*tf.math.sin(theta)*tf.math.cos(theta)/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2)**2) + (2*E**2*af**2*tf.math.sin(theta)*tf.math.cos(theta) - 2*J_phi**2*tf.math.cos(theta)/tf.math.sin(theta)**3)/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2))

    def PRHO_DOT_float(self,rho,Prho,phi,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi            
        return - (L1**2*rho*(E**2*af**2*numpy.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/numpy.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/((L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)**(3/2)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(af**2*numpy.cos(theta)**2 + rho**2)**2*(2*af**2*numpy.cos(theta)**2 + 2*rho**2)) + L5**2*rho*(E**2*af**2*numpy.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/numpy.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)**(3/2)*(af**2*numpy.cos(theta)**2 + rho**2)**2*(2*af**2*numpy.cos(theta)**2 + 2*rho**2)) - 4*rho*(E**2*af**2*numpy.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/numpy.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2)**2) + (-2*E**2*rho + 2*Prho**2*rho + 2*rho*(-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2)**2)/(numpy.sqrt(L1**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*numpy.sqrt(L5**2/(af**2*numpy.cos(theta)**2 + rho**2) + 1)*(2*af**2*numpy.cos(theta)**2 + 2*rho**2)))

    def PRHO_DOT(self,rho,Prho,theta,Ptheta):
        E     = self.E     
        L1    = self.L1   
        L5    = self.L5   
        af    = self.af    
        J_phi = self.J_phi            
        return - (L1**2*rho*(E**2*af**2*tf.math.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/tf.math.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/((L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)**(3/2)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(af**2*tf.math.cos(theta)**2 + rho**2)**2*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2)) + L5**2*rho*(E**2*af**2*tf.math.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/tf.math.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)**(3/2)*(af**2*tf.math.cos(theta)**2 + rho**2)**2*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2)) - 4*rho*(E**2*af**2*tf.math.sin(theta)**2 - E**2*(L1**2 + L5**2 + af**2 + rho**2) + J_phi**2/tf.math.sin(theta)**2 + Prho**2*(af**2 + rho**2) + Ptheta**2 - (-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2))/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2)**2) + (-2*E**2*rho + 2*Prho**2*rho + 2*rho*(-E*L1*L5 + J_phi*af)**2/(af**2 + rho**2)**2)/(tf.math.sqrt(L1**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*tf.math.sqrt(L5**2/(af**2*tf.math.cos(theta)**2 + rho**2) + 1)*(2*af**2*tf.math.cos(theta)**2 + 2*rho**2)))



    def Predict(self,Training,time_range,input: Any = None):

        prediction = Training.model.predict(time_range)
        prediction = np.transpose(prediction)
        N0 = prediction[0]
        N1 = prediction[1]
        N2 = prediction[2]
        N3 = prediction[3]
        N4 = prediction[4]

        rho    = np.array([])
        Prho   = np.array([])
        phi    = np.array([])
        theta  = np.array([])
        Ptheta = np.array([])

        for n in range(len(time_range)):
            rho    = np.append(rho,    self.rho_0    + (1-np.exp(-time_range[n]))* N0[n])
            Prho   = np.append(Prho,   self.Prho_0   + (1-np.exp(-time_range[n]))*(N1[n] + Training.h(time_range[n])))
            phi    = np.append(phi,    self.phi_0    + (1-np.exp(-time_range[n]))* N2[n])
            theta  = np.append(theta,  self.theta_0  + (1-np.exp(-time_range[n]))* N3[n])
            Ptheta = np.append(Ptheta, self.Ptheta_0 + (1-np.exp(-time_range[n]))* N4[n])

        H      = [self.HAMILTONIAN_float(rho[i],Prho[i],phi[i],theta[i],Ptheta[i]) for i in range(len(rho))]
        x      = rho*np.sin(theta)*np.cos(phi)
        y      = rho*np.sin(theta)*np.sin(phi)
        z      = rho*np.cos(theta)

        if input != None:
            
            fp = open(input.destination_path / Path(f'Prediction.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  s      \n')
            fp.write('# $3  H(s)   \n')
            fp.write('# $4  H(0)   \n')
            fp.write('# $5  rho(s) \n')
            fp.write('# $6  Prho(s)\n')
            fp.write('# $7  phi(s) \n')
            fp.write('# $8  theta(s) \n')
            fp.write('# $9  Ptheta(s) \n')
            fp.write('# $10  x(s)   \n')
            fp.write('# $11  y(s)   \n')
            fp.write('# $12  z(s)   \n')
            fp.write('# $13 N_rho(s)   \n')
            fp.write('# $14 N_Prho(s)  \n')
            fp.write('# $15 N_phi(s)   \n')
            fp.write('# $16 N_theta(s)   \n')
            fp.write('# $17 N_Ptheta(s)   \n')

            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e      %+12.10e      %+12.10e      %+12.10e' % (rho[i],Prho[i],phi[i],theta[i],Ptheta[i]))
                fp.write('      %12.10e      %+12.10e      %+12.10e' % (x[i],y[i],z[i]))
                fp.write('      %12.10e      %+12.10e      %+12.10e      %+12.10e      %+12.10e' % (N0[i],N1[i],N2[i],N3[i],N4[i]))
                fp.write('\n')
            fp.close()

    def Numerical_integration(self,time_range,method: str = 'Radau',input: Any = None):
        
        def system(s,y):

            rho, Prho, phi, theta, Ptheta = y
            drho_ds    = self.RHO_DOT_float(   rho,Prho,phi,theta,Ptheta)
            dPrho_ds   = self.PRHO_DOT_float(  rho,Prho,phi,theta,Ptheta)
            dphi_ds    = self.PHI_DOT_float(   rho,Prho,phi,theta,Ptheta)
            dtheta_ds  = self.THETA_DOT_float( rho,Prho,phi,theta,Ptheta)
            dPtheta_ds = self.PTHETA_DOT_float(rho,Prho,phi,theta,Ptheta)

            return [drho_ds,dPrho_ds, dphi_ds,dtheta_ds,dPtheta_ds]
        
        sol    = solve_ivp(system, t_span=(time_range[0],time_range[-1]), y0=self.bc, t_eval=time_range,method=method)
        rho    = sol.y[0]
        Prho   = sol.y[1]
        phi    = sol.y[2]
        theta  = sol.y[3]
        Ptheta = sol.y[4]

        H      = [self.HAMILTONIAN_float(rho[i],Prho[i],phi[i],theta[i],Ptheta[i]) for i in range(len(rho))]
        x      = rho*np.sin(theta)*np.cos(phi)
        y      = rho*np.sin(theta)*np.sin(phi)
        z      = rho*np.cos(theta)

        if input != None and len(H) == len(time_range):
            
            fp = open(input.destination_path / Path(f'results_{method}.dat'),'w+')
            fp.write('# $1  n      \n')
            fp.write('# $2  s      \n')
            fp.write('# $3  H(s)   \n')
            fp.write('# $4  H(0)   \n')
            fp.write('# $5  rho(s) \n')
            fp.write('# $6  Prho(s)\n')
            fp.write('# $7  phi(s) \n')
            fp.write('# $8  theta(s) \n')
            fp.write('# $9  Ptheta(s) \n')
            fp.write('# $10  x(s)   \n')
            fp.write('# $11  y(s)   \n')
            fp.write('# $12  z(s)   \n')
            fp.write('\n\n')
            for i in range(len(time_range)):
                fp.write('%-10i      %+12.10e      %+12.10e      %+12.10e' % (i+1,time_range[i],H[i],self.H0))
                fp.write('      %12.10e      %+12.10e      %+12.10e      %+12.10e      %+12.10e' % (rho[i],Prho[i],phi[i],theta[i],Ptheta[i]))
                fp.write('      %12.10e      %+12.10e      %+12.10e' % (x[i],y[i],z[i]))
                fp.write('\n')
            fp.close()

    def gnuplot_scripts(self,input,training,methods):
        
        fp = open(input.destination_path /  Path('plot_loss.gp'),'w+')
        fp.write('set xlabel "Log(epochs)" font ",14"\n')
        fp.write('set ylabel "Log(Loss)" font ",14"\n')
        fp.write('set grid\n')
        fp.write('set logscale y\n')
        fp.write('set logscale x\n')
        fp.write('set title "%s, b=%4.3f" font ",14"\n' % (self.case,self.b))
        fp.write('  plot "loss.dat" every 2 u 1:2 w l lc "black" title "(1-{/Symbol l})L^{dyn} + {/Symbol l}L^{E}, {/Symbol l} = %4.3f" \n' % (training.lambda_))
        fp.write('replot "loss.dat" every 2 u 1:3 w l lc "red"  title "L^{E}\n')
        fp.write('replot "loss.dat" every 2 u 1:4 w l lc "blue" title "L^{dyn}\n')
        fp.close()

        fp = open(input.destination_path /  Path('plot_energy.gp'),'w+')
        fp.write('set xlabel "s" font ",14"\n')
        fp.write('set ylabel "H(s)" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('set grid\n')        
        fp.write('plot "Prediction.dat" u 2:3 w l lc "red" title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 2:3 w l title "method %s"\n' % (method,method))
        fp.write('replot %6.5f lc "black" title "H(0)"\n' % (self.H0))
        fp.close()

        fp = open(input.destination_path /  Path('plot_cartesian_3D.gp'),'w+')
        fp.write('set xlabel "x" font ",14"\n')
        fp.write('set ylabel "y" font ",14"\n')
        fp.write('set zlabel "z" font ",14"\n')
        fp.write('unset logscale x\n')
        fp.write('unset logscale y\n')
        fp.write('unset logscale z\n')
        fp.write('set grid\n')        
        fp.write('splot "Prediction.dat" u 10:11:12 w l lc "red" title "Prediction"\n')
        for method in methods:
            fp.write('replot "results_%s.dat" u 10:11:12 w l title "method %s"\n' % (method,method))
        fp.close()


