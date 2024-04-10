""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np 
import utils
from operators.scrambler import ImageScrambler

sys.path.append(os.path.join('..'))

perm = ImageScrambler(32)

def get_network(args, permkey=None):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg16_permuted':
        from models.vgg_deformable import VGG_deformable
        net = VGG_deformable(permkey, args.b)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def dataset_loader(type, mean,std,batch_size,num_workers,shuffle, shuffle_pixels, permkey=None ):
    """ return training dataloader 
    Args:
        type: type of dataset cifar10 or cifar100
        mean: mean of dataset
        std: std of dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        shuffle_pixels: whether to shuffle pixels
    """
    transform_data_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            #transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    dataset  = {
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100
    }

    try:
        loader = dataset[type]
    except:
        print('the dataset name you have entered is not supported yet')
        sys.exit()

    train_transform = transform_data_aug if not shuffle_pixels else transform

    trainset = loader(root='./data', train=True, download=True, transform=train_transform)
    testset = loader(root='./data', train=False, download=True, transform=transform)

    if shuffle_pixels:
        #train
        fake_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
        a = list(fake_loader)
        b = permkey.forward(a[0][0]) if permkey is not None else perm.forward(a[0][0])
        trainset = torch.utils.data.TensorDataset(b,a[0][1])
        #test
        fake_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
        a = list(fake_loader)
        b = permkey.forward(a[0][0]) if permkey is not None else perm.forward(a[0][0])
        testset = torch.utils.data.TensorDataset(b,a[0][1])
    
    trainloader = DataLoader(trainset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)
    testloader = DataLoader(testset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)

    perm = permkey if permkey is not None else perm

    return trainloader, testloader, perm


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]