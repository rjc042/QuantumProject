#!/usr/bin/env python2.7

import numpy as np
from scipy import fftpack

def get_xns(dx, N):
    ''' Discretize x-space '''
    x = dx * (np.arange(N) - 0.5 * N)
    return x

def gauss_x(x, a, x0, k0):
    ''' Gaussian wave packet of width a, centered at x0, with momentum k0 '''
    return ((a * np.sqrt(np.pi)) ** (-0.5) * np.exp(-0.5 * ((x - x0) * 1.0 / a) ** 2 + 1j * x * k0))

def gauss_k(k, a, x0, k0):
    ''' (Analytical) Fourier transform of Gaussian '''
    return ((a / np.sqrt(np.pi)) ** 0.5 * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))

def square_well(x,h,w):
    '''
    Square potential of width w and height h centered at x=0
    Return V(x) as an array
    '''
    V_x = np.zeros(x.shape)
    V_x[abs(x) < w/2.0] = h
    V_x[x < -98] = 1E6
    V_x[x > 98] = 1E6
    return V_x

def get_k0(x, dk, k0=None):
    ''' Set momentum scale '''
    if k0 == None:
        k0 = -0.5 * len(x) * dk
    else:
        assert k0 < 0
    return k0

def get_k(x, dk, k0=None):
    ''' Discretize k-space '''
    k0 = get_k0(x, dk, k0)
    k = k0 + dk * np.arange(len(x))
    return k
