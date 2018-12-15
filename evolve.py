#!/usr/bin/env python2.7

import numpy as np
from scipy import fftpack

def get_psi_mod_x0(psi_x, k, x, dx):
    psi_mod_x = psi_x * np.exp(-1j * k[0] * x) * dx / np.sqrt(2 * np.pi)
    return psi_mod_x

def get_psi_mod_x(psi_mod_k):
    psi_mod_x = fftpack.ifft(psi_mod_k)
    return psi_mod_x

def get_psi_mod_k(psi_mod_x):
    psi_mod_k = fftpack.fft(psi_mod_x)
    return psi_mod_k


def get_psi_x(psi_mod_x, k, x, dx):
    ''' Returns wavefunction in x-space '''
    psi_x = psi_mod_x * np.exp(1j * k[0] * x) * np.sqrt(2 * np.pi) / dx
    return psi_x

def get_psi_k(psi_mod_k, x, dk):
    ''' Returns wavefunction in k-space '''
    psi_k = psi_mod_k * np.exp(-1j * x[0] * dk * np.arange(len(x)))
    return psi_k



def norm_wavefunction(wavefunc, dx):
    norm_wf = np.sqrt((abs(wavefunc) ** 2).sum() * 2 * np.pi / dx)
    return norm_wf

def evolve_x_half_dt(psi_mod_x, V_x, dt,hbar=1):
    prop_x = np.exp(-0.5 * 1j * V_x / hbar * dt)
    return psi_mod_x * prop_x

def evolve_k_dt(psi_mod_k, k, dt, hbar=1.0, m=1.0):
    prop_k = np.exp(-0.5 * 1j * hbar / m * k ** 2 * dt)
    return psi_mod_k * prop_k


def advance_mod_dt(psi_mod_x, psi_mod_k, V_x, k, dt, dx):
    psi_mod_x = evolve_x_half_dt(psi_mod_x, V_x, dt)
    psi_mod_k = get_psi_mod_k(psi_mod_x)
    psi_mod_k = evolve_k_dt(psi_mod_k, k, dt)
    psi_mod_x = get_psi_mod_x(psi_mod_k)
    psi_mod_x = evolve_x_half_dt(psi_mod_x, V_x, dt)

    norm_factor = norm_wavefunction(psi_mod_x, dx)
    psi_mod_x /= norm_factor
    psi_mod_k = get_psi_mod_k(psi_mod_x)
    return (psi_mod_x, psi_mod_k)



def advance_dt(new_psi_mod_x, new_psi_mod_k, x, k, dx, dk):
    ''' Advance x-space and k-space wavefunctions by dt '''
    psi_x = get_psi_x(new_psi_mod_x, k, x, dx)
    psi_k = get_psi_k(new_psi_mod_k, x, dk)
    return (psi_x, psi_k)
