import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel as Elementwise

n = 2**10
block_size  = 32
blocks      = n / block_size

code = open('mandelbrot.cpp', 'r').read()
mod = SourceModule(code)
func = mod.get_function('mandelbrot')

x0 = -2.0
y0 = -1.5
x1 = 1.0
y1 = 1.5
dx = (x1 - x0) / n
dy = (y1 - y0) / n

def static_plot(size = 4.0):
    m = np.zeros((n,n)).astype(np.uint8)

    fig = plt.figure()
    fig.set_size_inches(1. * size, 1. * size, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    func(drv.Out(m), 
         np.float64(x0), 
         np.float64(y0), 
         np.float64(dx), 
         np.float64(dy), 
         np.float64(2.0), grid=(blocks,blocks), block=(block_size,block_size,1))

    mplot = ax.imshow(np.log(m+1.0),
                      origin='lower',
                      cmap='Blues_r',
                      vmin=0.0, vmax=np.log(255))

    plt.savefig('mandelbrot.png', dpi=400)
    plt.show()

def evolution_gif(size = 4.0):
    m = np.zeros((n,n)).astype(np.uint8)

    fig = plt.figure()
    fig.set_size_inches(1. * size, 1. * size, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    mplot = ax.imshow(np.log(m+1.0),
                      origin='lower',
                      cmap='Blues_r',
                      vmin=0.0, vmax=np.log(255))

    def update(p):
        label = 'power %f' % (p)
        print label
        m = np.zeros((n,n)).astype(np.uint8)
        func(drv.Out(m), 
             np.float64(x0), 
             np.float64(y0), 
             np.float64(dx), 
             np.float64(dy), 
             np.float64(p), grid=(blocks,blocks), block=(block_size,block_size,1))
        mplot.set_data(np.log(m+1.0))

    anim = FuncAnimation(fig, update, frames=np.arange(1.0, 10.0, 0.05), interval=50)
    anim.save('evolution.gif', writer='imagemagick')
    plt.show()

def zoom_gif(size = 4.0):
    m = np.zeros((n,n)).astype(np.uint8)

    Lx = 3.0
    Ly = 3.0
    Cx = -0.7125009
    Cy = 0.251

    fig = plt.figure()
    fig.set_size_inches(1. * size, 1. * size, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    mplot = ax.imshow(m,
                      origin='lower',
                      cmap='Blues_r',
                      vmin=0.0, vmax=255)

    def update(p,
               Lx = Lx,
               Ly = Ly,
               Cx = Cx,
               Cy = Cy):
        label = 'power %f' % (p)
        print label
        m = np.zeros((n,n)).astype(np.uint8)

        Lx = 0.9 ** p * Lx
        Ly = 0.9 ** p * Ly

        x0 = Cx - Lx/2
        y0 = Cy - Ly/2

        dx = Lx / n
        dy = Ly / n

        func(drv.Out(m), 
             np.float64(x0), 
             np.float64(y0), 
             np.float64(dx), 
             np.float64(dy), 
             np.float64(2.0), grid=(blocks,blocks), block=(block_size,block_size,1))
        mplot.set_data(m)
        mplot.set_clim(np.min(m), np.max(m))

    anim = FuncAnimation(fig, update, frames=np.arange(0.0, 180.0, 0.5), interval=50)
    anim.save('zoom.gif', writer='imagemagick')
    plt.show()

static_plot()
evolution_gif()
zoom_gif()
