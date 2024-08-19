"""
Created on Fri Aug  2 19:22:27 2024
This code solves the steady state 2D Rayleigh-Benard problem for a rectangular domain. The state variables are velocity, pressure and temperature. It uses Central difference 1st order scheme
@author: Murugappan.Ramanathan
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#%% Material parameters, Geometric parameters,Mesh control and Boundary Conditions 

R = 1300                                                                  # The critical Rayleigh number
Ar = 3.495                                                                
Lhorizontal=Ar                                                            # Non dimensionalized length of the rectangular domain
Lvertical=1                                                                # Non dimensionalized height of the rectangular domain
nH = 24                                                                   # no. of grid elements in X axis
nV = 23                                                                  # no. of grid elements in Y axis

ux=0
uz=0
T0=1
T1=0
#%% Mesh
                          
nGrids = (nV+1) * (nH+1)                                               # no. of grid points
gridH = Lhorizontal/nH
gridV = Lvertical/nV

#%%  update equations

def CDsolver(x,ux,uz,T0,T1):
    
    fullN =  [y for y in range((nV+1)*(nH+1))]                                       # total no. of grid points
    N_skip = [0,nV,(nV+1)*nH,(nV+1)*(nH+1)-1]                                        # grid points to skip, i.e. 4 corners of rectangle
    filtered_elements = list(set(fullN)^set(N_skip))                                 #  skip 4 corners of rectanglr
    eq = np.zeros(len(4*fullN))                                                        # total no. of equations that we have, including the BC
    face_L = [y for y in range(1,(nV),1)]                                              # grid points of the left face of the domain
    face_R =  [y for y in range((nH+1)*(nV+1)-2,(nH+1)*(nV+1)-1-nV,-1)]                # grid points of the right face of the domain                            
    face_top = [y for y in range(2*nV+1,(nV+1)*nH,nV+1)]                              # grid points of the top face of the domain
    face_bottom = [y for y in range(nV+1,(nV+1)*nH,nV+1)]                           # grid points of the bottom face of the domain
    AllBC =  face_L  + face_R + face_top + face_bottom       
    grid_points = list(set(filtered_elements)^set(AllBC))                         # grid points without BC
  
    for N in grid_points:                                                           # the 4 NS equations(in order : energy, contunity, momentum in x,z directions) using the Central difference 1st order scheme                                                                    
        eq[4*N] = -(x[4*(N+nV+1)+2] - 2*x[4*N+2] + x[4*(N-nV-1)+2])/gridH**2  -(x[4*(N+1)+2] - 2*x[4*N+2] + x[4*(N-1)+2])/gridV**2 + (x[4*N]*(x[4*(N+nV+1)+2] - x[4*(N-nV-1)+2])/(2*gridH)) + (x[4*N+1]*(x[4*(N+1)+2] - x[4*(N-1)+2])/(2*gridV))
        eq[4*N+1] = ( x[4*(N+nV+1)] - x[4*(N-nV-1)] )/(2*gridH) + ( x[4*(N+1)+1] - x[4*(N-1)+1] )/(2*gridV)
        eq[4*N+2] = -(x[4*(N+nV+1)] - 2*x[4*N] + x[4*(N-nV-1)])/(gridH**2) -(x[4*(N+1)] - 2*x[4*N] + x[4*(N-1)])/(gridV**2) + (x[4*(N+nV+1)+3] - x[4*(N-nV-1)+3])/(2*gridH)
        eq[4*N+3] = -( x[4*(N+1)+1] - 2*x[4*N+1] + x[4*(N-1)+1] )/(gridV**2) -( x[4*(N+nV+1)+1] - 2*x[4*N+1] + x[4*(N-nV-1)+1] )/(gridH**2) + ( x[4*(N+1)+3] - x[4*(N-1)+3] )/(2*gridV) - R*x[4*N+2]
    
    for N in face_L:                                                                 # the 4 NS equations along with BC
        eq[4*N] =  x[4*N+2] - x[4*(N+nV+1)+2] 
        eq[4*N+1] =  x[4*N] -ux 
        eq[4*N+2] = -(x[4*(N+2*nV+2)] - 2*x[4*(N+nV+1)] + x[4*N])/( gridH**2) -(x[4*(N+1)] - 2*x[4*N] + x[4*(N-1)])/(gridV**2) + (x[4*(N+nV+1)+3] - x[4*(N)+3])/(gridH)
        eq[4*N+3] =  x[4*(N+nV+1)+1] - x[4*N+1] 
        
    for N in face_R:
        eq[4*N] =  x[4*N+2] - x[4*(N-nV-1)+2] 
        eq[4*N+1] =  x[4*N] -ux 
        eq[4*N+2] = -(x[4*(N)] - 2*x[4*(N-nV-1)] + x[4*(N-2*nV-2)])/( gridH**2) -(x[4*(N+1)] - 2*x[4*N] + x[4*(N-1)])/(gridV**2) + (x[4*(N)+3] - x[4*(N-nV-1)+3])/(gridH)
        eq[4*N+3] = x[4*(N-nV-1)+1] - x[4*N+1] 
        
    for N in face_top:
        eq[4*N] =  x[4*N+2] - T1 
        eq[4*N+1] =  x[4*N] -x[4*(N-1)] 
        eq[4*N+2] =  -( x[4*N+1] - 2*x[4*(N-1)+1] + x[4*(N-2)+1] )/(gridV**2) -( x[4*N+1] - 2*x[4*(N-nV-1)+1] + x[4*(N-2*nV-2)+1] )/(gridH**2) + ( x[4*N+3] - x[4*(N-1)+3] )/(gridV) - R*x[4*N+2]
        eq[4*N+3] =  x[4*N+1] -uz 
             
    for N in face_bottom:                                                                                                                
        eq[4*N] =  x[4*N+2] - T0 
        eq[4*N+1] =   x[4*N] - ux
        eq[4*N+2] = ( x[4*(N+nV+1)] - x[4*(N-nV-1)] )/(2*gridH) + ( x[4*(N+1)+1] - x[4*N+1] )/(gridV)
        eq[4*N+3] =   x[4*N+1] -uz 
        
     
    N=N_skip[0]                                                                                  
    eq[4*N] =  x[4*N+2] - T0 
    eq[4*N+1] =   x[4*N] -ux 
    eq[4*N+2] = ( x[4*(N+nV+1)] - x[4*N] )/(gridH) + ( x[4*(N+1)+1] - x[4*N+1] )/(gridV)
    eq[4*N+3] =   x[4*N+1] -uz 
        
    N=N_skip[1]    
    eq[4*N] =  x[4*N+2] - T1 
    eq[4*N+1] =  x[4*N] -x[4*(N-1)] 
    eq[4*N+2] =  -( x[4*N+1] - 2*x[4*(N-1)+1] + x[4*(N-2)+1] )/(gridV**2) -( x[4*N+1] - 2*x[4*(N+nV+1)+1] + x[4*(N+2*nV+2)+1] )/(gridH**2) + ( x[4*N+3] - x[4*(N-1)+3] )/(gridV) - R*x[4*N+2]
    eq[4*N+3] =  x[4*N+1] -uz 
        
    N=N_skip[2]
    eq[4*N] =  x[4*N+2] - T0 
    eq[4*N+1] =   x[4*N] -ux 
    eq[4*N+2] = ( x[4*N] - x[4*(N-nV-1)] )/(gridH) + ( x[4*(N+1)+1] - x[4*N+1] )/(gridV)
    eq[4*N+3] =   x[4*N+1] -uz 
        
    N=N_skip[3]
    eq[4*N] =  x[4*N+2] - T1 
    eq[4*N+1] =  x[4*N] -x[4*(N-1)] 
    eq[4*N+2] =  -( x[4*N+1] - 2*x[4*(N-1)+1] + x[4*(N-2)+1] )/(gridV**2) -( x[4*N+1] - 2*x[4*(N-nV-1)+1] + x[4*(N-2*nV-2)+1] )/(gridH**2) + ( x[4*N+3] - x[4*(N-1)+3] )/(gridV) - R*x[4*N+2]
    eq[4*N+3] =  x[4*N+1] -uz 

    return eq

x0=np.zeros(4*nGrids)
for j in range(nGrids):
    x0[4*j +2]=T0
X = fsolve(CDsolver, x0,(ux,uz,T0,T1))

Ux = np.zeros((nV+1,nH+1))                                                      # the solution initialization , Velocity in X
Uz = np.zeros((nV+1,nH+1))                                                      #  Velocity in Z
Th = np.zeros((nV+1,nH+1))                                                      # Temperature 
P = np.zeros((nV+1,nH+1))                                                       # Pressure
z=0
for i in range(nH+1):
    for j in range(nV,-1,-1):
        Ux[j,i] = X[4*z]
        Uz[j,i] = X[4*z +1]
        Th[j,i] = X[4*z +2]
        P[j,i] = X[4*z +3]
        z +=1
        
#%% Plotting
x = np.linspace(0,Ar,nH+1)
z = np.linspace(1,0,nV+1)
X,Z =np.meshgrid(x,z)
p1=plt.figure()
plt.contourf(X,Z,Th)
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('z')

z1 = np.linspace(0,1,nV+1)
X,Z1 =np.meshgrid(x,z1)
p2=plt.figure()
plt.streamplot(X, Z1, -Ux, Uz)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Velocity Field')
