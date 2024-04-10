#!/usr/bin/env python3

"""
Perform inference on a single image using a trained model.
"""

import sys
import os
sys.path.append(os.path.join('..', '..'))

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import conf as settings
from utils.utils_train import get_network
import utils.utils as utils
from operators.scrambler import ImageScrambler
import numpy as np

import matplotlib.pyplot as plt

def load_image(image_path, transform):
    """Load an image from the file system and apply transformations."""
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def save_feature_maps(layer_name, feature_maps):
    """Save the feature maps for a given layer."""
    for i, fmap in enumerate(feature_maps):
        print(f'Feature map shape: {fmap.shape}')
        for j in range(fmap.shape[1]):
            fmap1 = fmap[0]
            fmap1 = fmap1[j]
            fmap1 = fmap1.unsqueeze(0)
            fmap1 = fmap1.permute(1, 2, 0)

            fmap1 = fmap1.cpu().float().numpy()
            print(fmap1.shape)
            fmap1 = np.array(fmap1, dtype=np.float32)
            plt.imshow(fmap1, cmap='coolwarm')

            plt.savefig(f'feature_map_{layer_name}_{j}.png')
       


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on a single image')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-image', type=str, required=True, help='path to the image file')
    parser.add_argument('-permute', action='store_true', default=False, help='permute image or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')

    args = parser.parse_args()

    mean = settings.CIFAR10_TRAIN_MEAN
    std =  settings.CIFAR10_TRAIN_STD

    # Set up dataset-specific transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    perm = ImageScrambler(32)

    image = load_image(args.image, transform)
    image = image.unsqueeze(0)  # Add batch dimension


    #permute the image
    if args.permute:
        image = perm.forward(image)

    if args.permute:
        net = get_network(args, perm)
    else:
        net = get_network(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.to(device)
    net.eval()


    feature_maps = []
    def hook_function(module, input, output):
        feature_maps.append(output)

    image = image.to(device)

    layer_name = 'conv1'  # You need to specify the actual layer name based on your network architecture
    hook = getattr(net, layer_name).register_forward_hook(hook_function)

    image = image.to(device)
    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
    
    # After inference, save the feature maps
    save_feature_maps(layer_name, feature_maps)

    print(f'Predicted class: {predicted.item()}')

    hook.remove()
