#!/usr/bin/env python2.7

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from evolve import *
from initialize import *
from setupplots import *



# Constants
hbar = 1.0   # Planck's constant
m = 1.0      # Particle mass

# Position axis
dx = 0.1                    # Increments in x-space
N = 2 ** 11                 # Number of steps in x-axis
x = get_xns(dx, N)          # x-axis

dk = 2 * np.pi / (dx * N)   # Increments in k-space

# Potential barrier
V0 = 1.0
L = hbar / np.sqrt(2 * m * V0)
a = 3 * L
x0 = -60 * L
V_x = square_well(x, a, V0)


# Initialize momentum
p0 = np.sqrt(2 * m * 0.2 * V0)
dp2 = p0 * p0 * 1.0 / 80
d = hbar / np.sqrt(2 * dp2)
k0 = p0 / hbar
v0 = p0 / m


# Initial wavefunctions
psi_x = gauss_x(x, d, x0, k0)
k0 = -28
k = get_k(x, dk, k0)
psi_mod_x = get_psi_mod_x0(psi_x, k, x, dx)
psi_mod_k = get_psi_mod_k(psi_mod_x)
psi_k = get_psi_k(psi_mod_k,x,dk)


# Initialize position plot
(line_psix, line_V) = init_plot_x((x[0], x[-1]), (-0.2, V0+0.1))
line_V.set_data(x, V_x)

# Initialize momentum plot
xlimk = (-5, 5)
ymin, ymax = abs(psi_k).min(), abs(psi_k).max()
ylim_k = (ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin))
line_psik = init_plot_k(xlimk, ylim_k)



dt = 0.1                # Time step
t_max = 20              # Runs over time t=0 to t=20
N_steps = 20            # Advance wavefunctions by N_steps time steps before their plots are drawn again
t = 0
for i in range(int(t_max / dt)):
    for j in range(N_steps):
        psi_mod_x, psi_mod_k = advance_mod_dt(psi_mod_x, psi_mod_k, V_x, k, dt, dx)
        psi_x, psi_k = advance_dt(psi_mod_x, psi_mod_k, x, k, dx, dk)

    # Update plot data
    line_psix.set_data(x, abs(psi_x))
    line_psik.set_data(k, abs(psi_k))

    # Draw plot
    plt.draw()
    plt.pause(0.0001)
    t += dt
