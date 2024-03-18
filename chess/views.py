import argparse
import os
import copy
import matplotlib.colors as mcolors

import torch
from utils import concat_home_dir

from chess.data import get_classification, get_train_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_implicit(model, ax, bbox=(-0.0, 1.0), colored=True):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    model = model.cpu()
    xmin, xmax, ymin, ymax = bbox*2
    zmin, zmax = (-0.0, 1.0)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(zmin, xmax, 100) # number of slices
    # Bz = np.linspace(zmin, zmax, 100)  # number of slices
    A1, A2 = np.meshgrid(A, A) # grid on which the contour is plotted
    if colored:
        Bz = np.linspace(zmin, zmax, 100)  # number of slices
        norm = matplotlib.colors.Normalize(vmin=zmin+.35, vmax=zmax-0.35)
        # norm = matplotlib.colors.Normalize(vmin=zmin+.1, vmax=zmax-.1)
        cmap = plt.cm.jet #matplotlib.cm.get_cmap('viridis')

    for z in Bz: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(model, X,Y,z)
        if colored:
            cset = ax.contour(X, Y, Z+z, [z], zdir='z', alpha=.2, antialiased=True, colors=matplotlib.colors.to_hex(cmap(norm(z))), linewidths=10)
        else:
            cset = ax.contour(X, Y, Z+z, [z], zdir='z', alpha=.2, antialiased=True)
# [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = np.meshgrid(A,Bz)
        Y = fn(model, X,y,Z)
        # if colored:
        #     cset = ax.contour(X, Y + y, Z, [y], zdir='y', alpha=.2, antialiased=True, colors=matplotlib.colors.to_hex(cmap(norm(y))), linewidths=5)
        # else:
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', alpha=.2, antialiased=True, colors='black')

    for x in B: # plot contours in the YZ plane
        Y,Z = np.meshgrid(A,Bz)
        X = fn(model, x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', alpha=.2, antialiased=True, colors='black')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)
    ax.elev = 30
    ax.azim = 60
    return ax
    # plt.show()

def fn(model,x,y,z):
    def m(x,y,z):
        # x,y,z = torch.tensor(x), torch.tensor(y), torch.tensor(z)
        if len(list(x.shape)) < 2:
            x = x*np.ones_like(y)
        if len(list(y.shape)) < 2:
            y = y*np.ones_like(x)
        if len(list(z.shape)) < 2:
            z = z*np.ones_like(y)
        return x,y,z

    X, Y, Z = m(x,y,z)
    input = torch.tensor(np.c_[X.ravel(), Y.ravel(), Z.ravel()], dtype=torch.float)

    Z1 = model(input).round().detach().cpu()
    Z1 = Z1.reshape(X.shape)
    return Z1


def plot_function(models, train_data, xmin=0, xmax=1):
    chess = train_data.clone()
    chess = chess[chess[:, 0] <= xmax]
    chess = chess[chess[:, 0] >= xmin]
    Z_target = get_classification(chess).round().cpu()
    pos = (Z_target == 1)
    # print(pos)
    chess_pos = chess[pos.view(-1)].cpu()
    neg = (Z_target == 0)
    chess_neg = chess[neg.view(-1)].cpu()

    chess = chess.detach().cpu()
    length = len(models)
    print(length)
    total_rows = max(length//4, 2)
    total_col = max((length // total_rows),4)
    (line, col) = (0, 0)
    fig = plt.figure()
    for i in range(length):
        epoch, model = models[i]
        ax = fig.add_subplot(total_rows, total_col, i+1, projection='3d')
        ax.plot(chess_pos[:, 0], chess_pos[:, 1], chess_pos[:, 2], 'ro')
        ax.plot(chess_neg[:, 0], chess_neg[:, 1], chess_neg[:, 2], 'bo')
        ax.grid()
        ax.set_title("epoch: {}".format(epoch))
        if epoch == 0:
            ax.set_title("Random initialization")
        plot_implicit(model, ax)


        for i in range(length):

            if col % 9 == 0 and col > 0:
                line += 1
                col = 0
            else:
                col += 1

    plt.show()
