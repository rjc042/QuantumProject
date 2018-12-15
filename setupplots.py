#!/usr/bin/env python2.7

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# Initialize plots
fig = plt.figure()

def init_plot_x(xlimx, ylimx):
    ''' Initialize position space plot '''
    ax1 = fig.add_subplot(211, xlim=xlimx,ylim=ylimx)
    line_psix, = ax1.plot([], [], c='r', label=r'$|\psi(x)|$')
    line_V, = ax1.plot([], [], c='k', label=r'$V(x)$')

    ax1.set_title("Position space wavefunction")
    ax1.legend(prop=dict(size=12))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel(r'$|\psi(x)|$')
    return (line_psix, line_V)


def init_plot_k(xlimk, ylimk):
    ''' Initialize momentum space plot '''
    ax2 = fig.add_subplot(212, xlim=xlimk, ylim=ylimk)
    line_psik, = ax2.plot([], [], c='r', label=r'$|\psi(k)|$')

    ax2.set_title("Momentum space wavefunction")
    ax2.set_xlabel('$k$')
    ax2.set_ylabel(r'$|\psi(k)|$')
    return line_psik
