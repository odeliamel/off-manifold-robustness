import copy
from time import time

import torch
import torch.nn as nn
from torch import optim

from chess.data import init_data, get_test_iterator, get_train_iterator, get_train_data
import argparse
import os
from utils import concat_home_dir, mkdir_ifnotexists, Logger

# from chess.views import plot_function
from chess.model import OneLayerClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=8)


logger = Logger()
logger.init("train.log", 'training_log.txt')
print = logger.info


def parse_args():
    parser = argparse.ArgumentParser(description='diagonal classifier')

    # experiment data
    parser.add_argument('--folder-name', default="trained_models", help='experiment name')
    parser.add_argument('--exp-name', default="chess400_clean", help='experiment name')
    parser.add_argument('--model', default='OneLayerClassifier')
    parser.add_argument('--model_width', type=int, default='400')

    # general
    parser.add_argument('--batch-size', type=int, default=25, metavar='N', help='input batch size for training')
    parser.add_argument('--data-size', type=int, default=25, metavar='N', help='input data size for training')
    parser.add_argument('--test-data-size', type=int, default=25, metavar='N', help='input data size for training')

    parser.add_argument('--epochs', type=int, default=100000, metavar='N', help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate')
    parser.add_argument('--margin', type=float, default=0.01, metavar='margin', help='margin to stop when reaching')
    parser.add_argument('--factor', type=float, default=3, metavar='margin', help='margin to stop when reaching')
    parser.add_argument('--weights_decay', type=float, default=0.0, metavar='margin', help='margin to stop when reaching')

    parser.add_argument('--log-interval', type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--plot-interval',  type=int, default=10000)
    parser.add_argument('--plot-boundary', type=list, default=[0, 2, 4, 500, 700], help='view loss every k epochs')
    parser.add_argument('--first_layer_reg', type=bool, default=False)
    parser.add_argument('--first_layer_reg_lambda', type=float, default=0.8)

    args = parser.parse_args()
    return args


def test(model):
    args = parse_args()
    model.eval()
    test_loader_s = get_test_iterator(batch_size=args.batch_size)
    tst_correct = 0

    for batch_idx, (data, target) in enumerate(test_loader_s):
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            tst_correct += output.reshape(-1).detach().round().eq(target.reshape(-1)).float().sum()

    acc = tst_correct / args.test_data_size
    print('\tTest set: Accuracy: {}/{} ({:.2f}%)'.format(tst_correct, args.test_data_size, 100*acc))
    return acc

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

    model = eval(args.model)(out_features=args.model_width, factor=args.factor)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weights_decay)
    model.cuda()
    loss_fn = nn.BCELoss()
    loss_fn.requires_grad_(True)

    models = [(-1, copy.deepcopy(model).cpu())]
    end_epoch = 0

    for epoch in range(0, args.epochs + 1):
        end_epoch = epoch
        train_len = args.data_size
        model.train()

        for batch_idx, (data, target) in enumerate(get_train_iterator(args.batch_size)):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            if args.first_layer_reg:
                first_layer_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
                reg_loss = args.first_layer_reg_lambda * torch.norm(first_layer_params, 2)
                loss += reg_loss

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

        if model.get_margin(get_train_data()).item() > args.margin:
            break

    models.append((end_epoch, copy.deepcopy(model).cpu()))

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
        print("saved.")

    return model



if __name__ == '__main__':
    train()
