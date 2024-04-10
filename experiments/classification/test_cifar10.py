

""" 
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
import conf as settings

sys.path.append(os.path.join('..', '..'))

from utils.utils_train import get_network, dataset_loader
import utils.utils as utils
from operators.scrambler import ImageScrambler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, help='dataset used for training')
    parser.add_argument('-permute', action='store_true', default=False, help='permute test data or not')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-seed', type=int, help='resume training')
    parser.add_argument('-wrong_key', action='store_true', default=False, help='use gpu or not')

    args = parser.parse_args()

    if args.seed:
        utils.setup_seed(args.seed)

    perm = ImageScrambler(32)

    net = get_network(args,perm)

    mean = settings.CIFAR100_TRAIN_MEAN if args.dataset == 'cifar100' else settings.CIFAR10_TRAIN_MEAN
    std = settings.CIFAR100_TRAIN_STD if args.dataset == 'cifar100' else settings.CIFAR10_TRAIN_STD

    if args.wrong_key:
        perm = ImageScrambler(32) # new instance

    _, test_loader, _ = dataset_loader(
        args.dataset,
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        shuffle_pixels = args.permute,
        permkey = perm
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.load_state_dict(torch.load(args.weights, map_location=device))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    correct = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = net(image)
            
            #_, pred = output.topk(5, 1, largest=True, sorted=True)
            _, preds = output.max(1)

            #print(pred == preds)
            #print(pred.shape)
            #print(preds.shape)

            #label = label.view(label.size(0), -1).expand_as(pred)
            correct += preds.eq(label).sum()



    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Acc: ", correct.float() / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))