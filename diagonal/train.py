import copy
import random
from sched import scheduler
from time import time

import torch
import torch.nn as nn
from torch import optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from diagonal.data import init_data, get_test_iterator, get_train_iterator, get_train_data
from diagonal.model import OneLayerClassifier
from diagonal.deepmodel import DeepClassifier
from diagonal.views import plot_function
import argparse
import matplotlib.pyplot as plt
from utils import concat_home_dir, mkdir_ifnotexists, Logger
from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=8)

logger = Logger()
logger.init("train.log", 'training_log.txt')
print = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='diagonal classifier')

    # experiment data
    parser.add_argument('--folder-name', default="trained_models", help='experiment name')
    parser.add_argument('--exp-name', default="diag_clean", help='experiment name')
    # parser.add_argument('--model', default='OneLayerClassifier')
    parser.add_argument('--model', default='DeepClassifier')

    # general
    parser.add_argument('--batch-size', type=int, default=7, metavar='N', help='input batch size for training')
    parser.add_argument('--data-size', type=int, default=7, metavar='N', help='input data size for training')
    parser.add_argument('--test-data-size', type=int, default=7, metavar='N', help='input data size for testing')

    parser.add_argument('--epochs', type=int, default=40000, metavar='N', help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='learning rate')
    parser.add_argument('--margin', type=float, default=0.3, metavar='margin', help='margin to stop when reaching')
    parser.add_argument('--factor', type=float, default=1, metavar='factor', help='the factor in which we devide the initialization weights')
    parser.add_argument('--weights_decay', type=float, default=0.0, metavar='weights decay', help='weights decay factor')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--plot-interval',  type=int, default=10000)
    parser.add_argument('--plot-boundary', type=list, default=[10, 20, 30, 100], help='view loss every k epochs')

    # model parameters
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    print("=================ARGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    init_data(data_size=args.data_size, test_data_size=args.test_data_size)

    mkdir_ifnotexists(concat_home_dir(os.path.join(args.folder_name)))
    mkdir_ifnotexists(concat_home_dir(os.path.join(args.folder_name, 'exps')))
    mkdir_ifnotexists(concat_home_dir(os.path.join(args.folder_name, 'exps', args.exp_name)))
    mkdir_ifnotexists(concat_home_dir(os.path.join(args.folder_name, 'exps', args.exp_name, 'checkpoints')))
    print(concat_home_dir(os.path.join(args.folder_name, 'exps', args.exp_name, 'checkpoints', 'epoch_{}.pt'.format(str(3)))))

    model = eval(args.model)(out_features=100, factor=args.factor)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weights_decay)  # model.parameters()
    model.cuda()
    loss_fn = nn.BCELoss()
    loss_fn.requires_grad_(True)

    end_epoch = 0

    init_model = copy.deepcopy(model)
    models = [(-1, init_model.cpu())]
    anim_models = [(-1, init_model.cpu())]

    for epoch in range(0, args.epochs + 1):
        end_epoch = epoch
        train_len = args.data_size
        model.train()

        for batch_idx, (data, target) in enumerate(get_train_iterator(args.batch_size)):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()

        if epoch % args.log_interval == 0:
            correct = output.reshape(-1).detach().round().eq(target.reshape(-1)).float().sum()
            acc = correct.float() / len(data)
            print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_idx * len(data), train_len,
                loss.data.item(), acc, optimizer.param_groups[0]['lr']))
            print("margin: {:.4f}".format(model.get_margin(get_train_data())))

        if epoch in args.plot_boundary:
            models.append((epoch, copy.deepcopy(model).cpu()))

        c_model = copy.deepcopy(model)
        anim_models.append((epoch, c_model.cpu()))
        if model.get_margin(get_train_data()).item() > args.margin:
            break

    models.append((end_epoch, copy.deepcopy(model).cpu()))
    plot_function(models, get_train_data(), xmin=-1.2, xmax=1.2)
    s = input("save? y/n")
    if s == 'y':
        for e, model in models:
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'args': args,
                        },
                       concat_home_dir(
                           os.path.join(args.folder_name, 'exps', args.exp_name, 'checkpoints',
                                        'epoch_{}.pt'.format(str(e)))))
    return model


if __name__ == '__main__':
    train()
