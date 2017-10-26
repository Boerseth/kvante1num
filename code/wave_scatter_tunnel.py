# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 04:36:45 2017

@author: frode
"""
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#==============================================================================
# Defining parameters
#==============================================================================
start_time = time.clock()

hbar = 1.0
m = 1.0
k0 = 20.0
L = 20.0
xs = 5
vg = hbar*k0/m # dw/dk = d/dk ( hbar k^2 / 2m ) = hbar k / m

E = hbar**2 * k0**2 / (2*m)
w = E / hbar

Nx = int(2.5*k0*L) + 1 # 1001
Dx = L/(Nx-1)
rho = 0.1  # For scaling Dt


#==============================================================================
# Defining mathematical functions
#
# V_func:
# - x is an array
# - c gives us Vmax by  Vmax = cE
# - b is width of potential
#
# Psi_func:  as described in problem text, only for starting values
#==============================================================================
def V_func(x,c,b):
    l = int(b/Dx)
    V_list = [0.0]*(Nx/2 + Nx%2 - l/2 -l%2)
    V_list += [c*E]*(l)
    V_list += [0.0]*(Nx/2 - l/2)
    return np.array(V_list)
    
def Psi_func(x, sigmax, Dt):
    gaussian = np.exp( - (x-xs)**2 / (2*sigmax**2) )/(np.pi*sigmax**2)**0.25
    PsiR = gaussian * np.cos( k0*x )
    PsiI = gaussian * np.sin( k0*x - w*Dt/2 )
    PsiR[0] = 0.0
    PsiR[-1] = 0.0
    PsiI[0] = 0.0
    PsiI[-1] = 0.0
    return PsiR, PsiI
    
    
#==============================================================================
# Plotting wave function
#==============================================================================
def Psi_plot( x, Psi_R, Psi_I, plot_squared=False):
    matplotlib.rcParams.update({'font.size': 22})
    
    plt.figure(figsize=(15,10))
    plt.plot( x  , Psi_R , label='$\Psi_R$')
    plt.plot( x  , Psi_I , label='$\Psi_I$')
    plt.legend()
    plt.show()
    
    if plot_squared:
        Psi = Psi_R**2 + Psi_I**2
        plt.figure(figsize=(15,10))
        plt.plot( x , Psi , label='$|\Psi(x,t=T)|^2$')
        plt.legend()
        plt.show()


#==============================================================================
# Propagate Psi, given square potential of height cE and width b
#==============================================================================
def Psi_propagate(c,b,sigmax,plot=False,plot_squared=False):
    # Define the last remaining variables
    Dt = rho  *  2*m*hbar*Dx**2 / (2*m*c*E*Dx**2 + hbar**2)
    T = L/(2*vg)
    Nt = int(T/Dt)
    
    # Defining arrays
    x = np.linspace(0.0, L, Nx)
    V = V_func(x,c,b)
    PsiR0, PsiI0 =  Psi_func(x,sigmax, Dt)
    PsiR, PsiI = Psi_func(x,sigmax, Dt) #To be iterated over
    
    # Defining matrices
    D = -2*np.diagflat([0.0] + [1.0]*(Nx-2) + [0.0])
    D += np.diagflat([0.0] + [1.0]*(Nx-2), 1)
    D += np.diagflat([1.0]*(Nx-2) + [0.0],-1)
    D = D * (hbar * Dt / (2*m*Dx**2))
    P = np.diagflat(V) #I will assume that V is always zero at the endpoints.
    P = P * (Dt/hbar)
    Matrix = - D + P
    
    # Propagating wave function
    for t in range(Nt):
        PsiR += Matrix.dot(PsiI)
        PsiI += - Matrix.dot(PsiR)
        if t % 100 == 0:
            C = np.sqrt(sum(PsiR**2 + PsiI**2)*Dx)
            PsiR = PsiR/C
            PsiI = PsiI/C # Just to preserve normalization
    
    if plot:
        Psi_plot( x, PsiR0, PsiI0, plot_squared=plot_squared)
        Psi_plot( x, PsiR, PsiI, plot_squared=plot_squared)
    return PsiR, PsiI


#==============================================================================
# Calculate probabilities of reflection, transmission
#==============================================================================
def probabilities_ref_tra(PsiR, PsiI):
    Psi_squared = PsiR**2 + PsiI**2
    if Nx%2 == 1:
        p_ref = (sum(Psi_squared[:Nx/2]) + 0.5*Psi_squared[Nx/2])*Dx
        p_tra = (sum(Psi_squared[Nx/2+1:]) + 0.5*Psi_squared[Nx/2])*Dx
    else:
        p_ref = sum(Psi_squared[:Nx/2])*Dx
        p_tra = sum(Psi_squared[Nx/2:])*Dx
    return p_ref, p_tra


#==============================================================================
# Problem 1
#==============================================================================
def problem_1():
    Psi_propagate(0,0,1.0,plot=True,plot_squared=True)


#==============================================================================
# Problem 2
#==============================================================================
def problem_2():
    for sigmax in [0.1, 0.2, 0.5, 1.0, 2.0]:
        Psi_propagate(0,0,sigmax,plot=True,plot_squared=False)
        

#==============================================================================
# Problem 3
#==============================================================================
def problem_3():
    c = 0.5
    b = L/50
    sigmax = 1.0
    PsiR, PsiI = Psi_propagate(c,b,sigmax, plot=False,plot_squared=False)
    
    x = np.linspace(0.0, L, Nx)
    V = V_func(x,c,b)
    
    plt.figure(figsize=(15,10))
    plt.plot( x  , PsiR , label='$\Psi_R$')
    plt.plot( x  , PsiI , label='$\Psi_I$')
    plt.plot( x  , V/E , label='$V$', linewidth=2)
    plt.legend()
    plt.show()
    
    p_ref, p_tra = probabilities_ref_tra(PsiR, PsiI)
    
    print "Probabilities of "
    print "reflection: \t", p_ref
    print "transmission: \t", p_tra
    print "total: \t\t", p_ref + p_tra
    
    
#==============================================================================
# Problem 4
#==============================================================================
def problem_4():
    p_ref_list = []
    p_tra_list = []
    
    b = L/50
    sigmax = 1.0
    c_list = np.linspace(0.0, 1.5, 50)
    for c in c_list:
        PsiR, PsiI = Psi_propagate(c,b,sigmax, plot=False,plot_squared=False)
        p_ref, p_tra = probabilities_ref_tra(PsiR, PsiI)
        p_ref_list.append(p_ref)
        p_tra_list.append(p_tra)
    
    plt.figure(figsize=(15,10))
    plt.plot( c_list  , p_ref_list , label='$p_r$')
    plt.plot( c_list  , p_tra_list , label='$p_t$')
    plt.xlabel("$V_{max} / E$")
    plt.legend()
    plt.show()
    
    
#==============================================================================
# Problem 5
#==============================================================================
def problem_5():
    p_ref_list = []
    p_tra_list = []
    
    c = 0.9
    sigmax = 1.0
    b_list = np.linspace(0.0, L/20, 50)
    for b in b_list:
        PsiR, PsiI = Psi_propagate(c,b,sigmax, plot=False,plot_squared=False)
        p_ref, p_tra = probabilities_ref_tra(PsiR, PsiI)
        p_ref_list.append(p_ref)
        p_tra_list.append(p_tra)
    
    plt.figure(figsize=(15,10))
    plt.plot( b_list  , p_ref_list , label='$p_r$')
    plt.plot( b_list  , p_tra_list , label='$p_t$')
    plt.xlabel("potential width $b$")
    plt.legend()
    plt.show()


#==============================================================================
# Pick your poison
#==============================================================================

problems = sys.argv[1:]

for number in problems:
    if int(number) in [1,2,3,4,5]:
        eval("problem_"+str(number)+"()")
    else:
        print "Not a valid problem number!"
        print "So I will just do problem 3 instead."
        problem_3()
        
if len(problems) == 0:
    print "No arguments => problem 3"
    print "..."
    problem_3()


#problem_1() # 6 seconds
#problem_2() # 30 seconds
#problem_3() # 6 seconds
#problem_4() # 300 seconds
#problem_5() # 310 seconds

print "\nTime:", time.clock() - start_time









