import argparse
import os

from chess.views import plot_implicit

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from utils import concat_home_dir

from chess.data import get_classification, get_train_data, init_data
from chess.model import OneLayerClassifier

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='diagonal classifier')

    # models_names = ["OneLayerClassifier", "chess4000_clean", "chess4000_clean", "chess4000_clean"]
    models_names = ["OneLayerClassifier", "chess400_clean", "chess400_clean", "chess400_clean"]
    # models_names = ["OneLayerClassifier", "chess_clean", "chess_smallinit6_4", "chess_regulr0.8"]

    # models_names = ["OneLayerClassifier", "nchess_clean", "nnchess_smallinit1.5", "nnchess_regular0.001"]
    # models_names = ["OneLayerClassifier", "nnchess400_clean", "nnchess400_smallinit3_2", "nnchess400_regulr0.003_2"]
    # models_names = ["DeepClassifier", "5deep_clean", "5deep_smallinit12_firstlayer",
    #                 "5deep_regulr1.0_firstlayer"]
    # experiment data
    parser.add_argument('--folder-name', default="trained_models", help='experiment name')
    parser.add_argument('--exp-name1', default=models_names[1], help='experiment name')
    parser.add_argument('--exp-name2', default=models_names[2], help='experiment name')
    parser.add_argument('--exp-name3', default=models_names[3], help='experiment name')
    parser.add_argument('--model', default=models_names[0])
    parser.add_argument('--model_width', type=int, default='400')

    # general
    parser.add_argument('--batch-size', type=int, default=25, metavar='N', help='input batch size for training')
    parser.add_argument('--data-size', type=int, default=25, metavar='N', help='input data size for training')
    parser.add_argument('--test-data-size', type=int, default=25, metavar='N', help='input data size for training')

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
        models[i] = eval(args.model)(out_features=args.model_width)
        # print([int(i.split('_')[-1][:-3]) for i in os.listdir(paths[i])])
        # print([int(i.split('_')[-1][:-3]) for i in os.listdir(paths[i])][-1])
        print(i, paths[i])
        all_dirs = sorted([int(j.split('_')[-1][:-3]) for j in os.listdir(paths[i])])
        all_dirs.reverse()
        epoch = 0
        check_pt = torch.load(os.path.join(paths[i], 'epoch_{}.pt'.format(all_dirs[epoch])))
        models[i].load_state_dict(check_pt['model_state_dict'])
        models[i] = models[i].cuda()
        while models[i].get_margin(get_train_data()).item() > 0.301:
            epoch += 1
            check_pt = torch.load(os.path.join(paths[i], 'epoch_{}.pt'.format(all_dirs[epoch])))
            models[i].load_state_dict(check_pt['model_state_dict'])

    plot_function(models, get_train_data())


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
    total_rows = 1
    total_col = length
    # (line, col) = (0, 0)

    titles = ["Normal Initialization", "Small Initialization", "Explicit Regularization"]

    fig = plt.figure()
    for i in range(length):
        ax = fig.add_subplot(total_rows, total_col, i+1, projection='3d')
        # ax = fig.add_subplot(total_rows, total_col, 1, projection='3d')
        ax.plot(chess_pos[:, 0], chess_pos[:, 1], chess_pos[:, 2], 'ro')
        ax.plot(chess_neg[:, 0], chess_neg[:, 1], chess_neg[:, 2], 'bo')
        # ax.grid()

        # margin = model.get_margin(train_data).item()
        # print(margin)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.zaxis.set_tick_params(labelsize=14)
        ax.elev = 30
        ax.azim = 60
        # plt.zticks()
        # if i == 0: continue
        model = models[i]
        # margin = model.get_margin(train_data).item()
        ax.set_title(titles[i])# + "    margin: %.2f" % margin)
        plot_implicit(model, ax)

        # for i in range(length):
        #     if col % 9 == 0 and col > 0:
        #         line += 1
        #         col = 0
        #     else:
        #         col += 1
        # adv_vec = model.get_universal_gradient().cpu()
        # for j in range(chess.shape[0]):
        #     if Z_target[j] == 0:
        #         # print(j, chess[j], adv_vec)
        #         ax.quiver(chess[j][0], chess[j][1], chess[j][2], adv_vec[0], adv_vec[1], adv_vec[2], color='black', label="adv dir")
        #     else:
        #         ax.quiver(chess[j][0], chess[j][1], chess[j][2], -adv_vec[0], -adv_vec[1], -adv_vec[2], color='black', label="adv dir")

    # fig.canvas.manager.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    main()
