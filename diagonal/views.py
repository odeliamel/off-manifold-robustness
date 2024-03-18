import numpy as np
import torch
from matplotlib import pyplot as plt
import math
from diagonal.data import get_classification


def plot_function(models, train_data, xmin=-1.2, xmax=1.2):
    x = np.linspace(xmin, xmax, 500)
    y = np.linspace(xmin, xmax, 500)

    X, Y = np.meshgrid(x, y)

    input = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float)
    diagonal = train_data.clone()
    diagonal = diagonal[diagonal[:, 0] <= xmax]
    diagonal = diagonal[diagonal[:, 0] >= xmin]
    Z_target = get_classification(diagonal).round().cpu()
    pos = (Z_target == 1)
    neg = (Z_target == 0)
    diagonal = diagonal.detach().cpu()

    length = len(models)
    print(length)
    total_rows = max(math.ceil(length/5), 2)
    total_col = max(math.ceil(length/total_rows), 3)
    print(total_rows, total_col)
    fig, ax = plt.subplots(total_rows, total_col)
    (line, col) = (0, 0)
    for i in range(length):
        epoch, model = models[i]
        Z1 = model(input).round().detach().cpu()
        Z1 = Z1.reshape(X.shape)

        ax[line, col].contour(X, Y, Z1, cmap=plt.cm.gray)
        ax[line, col].plot(diagonal[pos.view(-1), 0], diagonal[pos.view(-1), 1], 'ro')
        ax[line, col].plot(diagonal[neg.view(-1), 0], diagonal[neg.view(-1), 1], 'bo')
        ax[line, col].grid()
        ax[line, col].set_title("epoch: {}".format(epoch))
        if epoch == 0:
            ax[line, col].set_title("Random initialization")

        ax[line, col].set_xlim(xmin, xmax)
        ax[line, col].set_ylim(xmin, xmax)
        ax[line, col].set_aspect(1)

        if col % (total_col-1) == 0 and col > 0:
            line += 1
            col = 0
        else:
            col += 1

    plt.show()
