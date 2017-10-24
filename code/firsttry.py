# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:41:18 2017

@author: frode
"""

import numpy as np
import matplotlib.pyplot as plt


N = 100
L = 20.0
b = 0.5
V_val = 10.0
x_list = list(np.linspace(0,L,N))
x0 = 5.0
sigma = 1.0
k0 = 5.0
w = 5.0
hbar = 1.0
m = 1.0

Dx = L/(N-1)
Dt = 0.00001*Dx**2

def V_func(x):
    if abs(x-L/2) < b:
        return V_val
    else:
        return 0.0

psiR = [ np.exp( -(x-x0)**2/(2*sigma**2) ) * np.cos( k0*x + w*Dt ) for x in x_list]
psiI = [ np.exp( -(x-x0)**2/(2*sigma**2) ) * np.sin( k0*x ) for x in x_list]
psiR[0] = 0.0
psiR[-1] = 0.0
psiI[0] = 0.0
psiI[-1] = 0.0


psi = psiR + psiI

I = np.identity(N)

D = np.zeros((N,N))
D += np.diagflat([-2]*N, 0)
D += np.diagflat([1]*(N-1), 1)
D += np.diagflat([1]*(N-1), -1)

V_list = [ V_func(x) for x in x_list ]

V = np.diagflat(V_list, 0)

D = ((hbar*Dt)/(2*m*Dx**2))*D
V = (Dt/hbar)*V

M = np.vstack([np.hstack([ I, V-D ]),
               np.hstack([ V+D, I ])])
               
O = np.vstack([np.hstack([ I, D-V ]),
               np.hstack([ -D-V, I ])])
O = np.linalg.inv(O)
               
plt.plot(x_list, psi[:N])
plt.plot(x_list, psi[N:])
plt.show()

Psi = np.array(psi)
for i in range(40):
    for j in range(20000):
        Psi = O.dot(Psi)
        psi = list(Psi)
    plt.plot(x_list, psi[:N])
    plt.plot(x_list, psi[N:])
    plt.show()
    hei = raw_input()
         



psiR = psi[:N]
psiI = psi[N:]

absPsi = [ (r**2 + i**2) for (r,i) in zip(psiR,psiI)]
print sum(absPsi)








