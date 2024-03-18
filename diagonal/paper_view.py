import argparse
import os

from torch import nn

# from diagonal.views import add_adversarial_vectors

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import os

from chess.views import plot_implicit

import torch
import numpy as np
from utils import concat_home_dir

from diagonal.data import get_classification, get_train_data, init_data, get_test_iterator
from diagonal.model import OneLayerClassifier
from diagonal.deepmodel import DeepClassifier

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='diagonal classifier')

    # models_names = [classifier_classname, exp_name1, exp_name2, exp_name3]
    # models_names = ["DeepClassifier", "", "", ""]
    models_names = ["OneLayerClassifier", "", "", ""]

    # experiment data
    parser.add_argument('--folder-name', default="trained_models", help='experiment name')
    parser.add_argument('--exp-name1', default=models_names[1], help='experiment name')
    parser.add_argument('--exp-name2', default=models_names[2], help='experiment name')
    parser.add_argument('--exp-name3', default=models_names[3], help='experiment name')
    parser.add_argument('--model', default=models_names[0])


    # general
    parser.add_argument('--batch-size', type=int, default=7, metavar='N', help='input batch size for training')
    parser.add_argument('--data-size', type=int, default=7, metavar='N', help='input data size for training')
    parser.add_argument('--test-data-size', type=int, default=7, metavar='N', help='input data size for training')

    parser.add_argument('--epochs', type=int, default=40000, metavar='N', help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=int, default=400, help='save model every k epochs')
    parser.add_argument('--test-loss', type=int, default=100, help='view loss every k epochs')
    parser.add_argument('--load-model', default=False, help='Load saved model')
    parser.add_argument('--plot-interval',  type=int, default=10000)
    # parser.add_argument('--plot-boundary', type=list, default=[10,100,500,1000,1500,2000,3000,4000], help='view loss every k epochs')
    parser.add_argument('--plot-boundary', type=list, default=[2, 10, 20, 50, 80, 100, 120, 170],
                        help='view loss every k epochs')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument('--adv-training', type=bool, default=False)
    parser.add_argument('--to-save', type=bool, default=False)


    # model parameters
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    init_data(data_size=args.data_size, test_data_size=args.test_data_size)

    paths = []
    paths.append(concat_home_dir(os.path.join(args.folder_name, 'exps', args.exp_name1, 'checkpoints')))
    paths.append(concat_home_dir(os.path.join(args.folder_name, 'exps', args.exp_name2, 'checkpoints')))
    paths.append(concat_home_dir(os.path.join(args.folder_name, 'exps', args.exp_name3, 'checkpoints')))

    models = [0,0,0]

    for i in range(3):
        models[i] = eval(args.model)(out_features=100)
        models[i].cuda()

        dir = sorted([int(j.split('_')[-1][:-3]) for j in os.listdir(paths[i])])[-1]
        check_pt = torch.load(os.path.join(paths[i], 'epoch_{}.pt'.format(dir)))
        models[i].load_state_dict(check_pt['model_state_dict'])
        eval_model(models[i])

    plot_function(models, get_train_data())


def eval_model(model):
    test_data = get_test_iterator(batch_size=1)
    total_loss = 0
    total_len = 0
    correct_label_count = 0
    model.eval()
    loss_fn = nn.BCELoss().requires_grad_(False)
    for batch_idx, (data, target) in enumerate(test_data):
        data = data
        target = target
        total_len += target.shape[0]
        output = model(data)
        loss = loss_fn(output, target)
        total_loss += loss.detach().item()
        for i in range(len(output)):
            margin = ((target[i] - 0.5) * 2) * ((output[i] - 0.5) * 2)
            if (margin >= 0):
                correct_label_count += 1
    print(' Loss: ', total_loss / total_len, ' Correct label: ', correct_label_count, '/', total_len)
    return correct_label_count / total_len, total_loss / total_len


def plot_function(models, train_data, xmin=-1.2, xmax=1.2):
    x = np.linspace(xmin, xmax, 500)
    y = np.linspace(xmin, xmax, 500)

    X, Y = np.meshgrid(x, y)

    all_inputs = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float)
    diagonal = train_data.clone()
    diagonal = diagonal[diagonal[:, 0] <= xmax]
    diagonal = diagonal[diagonal[:, 0] >= xmin]
    z_target = get_classification(diagonal).round().cpu()
    pos = (z_target == 1)
    neg = (z_target == 0)
    diagonal = diagonal.detach().cpu()

    length = len(models)
    print(length)
    total_rows = 1
    total_col = 3
    print(total_rows, total_col)
    (line, col) = (0, 0)
    titles = ["Normal Initialization", "Small Initialization", "Explicit Regularization"]
    fig, axs = plt.subplots(total_rows, total_col)

    for i in range(length):
        ax = axs[col]
        model = models[i]
        z = model(all_inputs.cuda()).round().detach().cpu()
        z = z.reshape(X.shape)

        ax.contour(X, Y, z, cmap=plt.cm.gray)
        ax.plot(diagonal[pos.view(-1), 0], diagonal[pos.view(-1), 1], 'ro')
        ax.plot(diagonal[neg.view(-1), 0], diagonal[neg.view(-1), 1], 'bo')
        ax.grid()

        ax.set_title(titles[i])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)

        ax.set_aspect(1)
        col += 1

    plt.show()


if __name__ == '__main__':
    main()
