"""
vgg

[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6

"""

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

import sys
import os 

sys.path.append(os.path.join('..'))

from operators.deformable_pooling import DeformMaxPool2d
from operators.offset import calculate_offset

cfg = {
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
}

class VGG_deformable(nn.Module):

    def __init__(self, initial_perm, batch, num_class=100):
        super().__init__()

        # conv1
        self.conv1 =  DeformConv2d(3, 64, 3, padding=1)
        self.offset_1, perm_1 = calculate_offset(32, 1, 1, 3, initial_perm, batch)
        self.bcn1 = nn.BatchNorm2d(64)
        self.conv2 =  DeformConv2d(64, 64, 3, padding=1)
        self.offset_2, perm_2 = calculate_offset(self.offset_1.shape[-1], 1, 1, 3, perm_1, batch)
        self.bcn2 = nn.BatchNorm2d(64)
        self.pool1 = DeformMaxPool2d(self.offset_2.shape[-1], perm_2, 2)
        
        # conv2
        self.conv3 = DeformConv2d(64, 128, 3, padding=1)
        self.offset_3, perm_3 = calculate_offset(16, 1, 1, 3, self.pool1.new_perm, batch)
        self.bcn3 = nn.BatchNorm2d(128)
        self.conv4 = DeformConv2d(128, 128, 3, padding=1)
        self.offset_4, perm_4 = calculate_offset(self.offset_3.shape[-1], 1, 1, 3, perm_3, batch)
        self.bcn4 = nn.BatchNorm2d(128)
        self.pool2 = DeformMaxPool2d(self.offset_4.shape[-1], perm_4, 2)

        # conv3
        self.conv5 = DeformConv2d(128, 256, 3, padding=1)
        self.offset_5, perm_5 = calculate_offset(8, 1, 1, 3,  self.pool2.new_perm, batch)
        self.bcn5 = nn.BatchNorm2d(256)
        self.conv6 = DeformConv2d(256, 256, 3, padding=1)
        self.offset_6, perm_6 = calculate_offset(self.offset_5.shape[-1], 1, 1, 3, perm_5, batch)
        self.bcn6 = nn.BatchNorm2d(256)
        self.conv7 = DeformConv2d(256, 256, 3, padding=1)
        self.offset_7, perm_7 = calculate_offset(self.offset_6.shape[-1], 1, 1, 3, perm_6, batch)
        self.bcn7 = nn.BatchNorm2d(256)
        self.pool3 = DeformMaxPool2d(self.offset_7.shape[-1], perm_7, 2)


        # conv4
        self.conv8 = DeformConv2d(256, 512, 3, padding=1)
        self.offset_8, perm_8 = calculate_offset(4, 1, 1, 3, self.pool3.new_perm, batch)
        self.bcn8 = nn.BatchNorm2d(512)
        self.conv9 = DeformConv2d(512, 512, 3, padding=1)
        self.offset_9, perm_9 = calculate_offset(self.offset_8.shape[-1], 1, 1, 3, perm_8, batch)
        self.bcn9 = nn.BatchNorm2d(512)
        self.conv10 = DeformConv2d(512, 512, 3, padding=1)
        self.offset_10, perm_10 = calculate_offset(self.offset_9.shape[-1], 1, 1, 3, perm_9, batch)
        self.bcn10 = nn.BatchNorm2d(512)
        self.pool4 = DeformMaxPool2d(self.offset_10.shape[-1], perm_10, 2)

        # conv5
        self.conv11 = DeformConv2d(512, 512, 3, padding=1)
        self.offset_11, perm_11 = calculate_offset(2, 1, 1, 3,  self.pool4.new_perm, batch)
        self.bcn11 = nn.BatchNorm2d(512)
        self.conv12 = DeformConv2d(512, 512, 3, padding=1)
        self.offset_12, perm_12 = calculate_offset(self.offset_11.shape[-1], 1, 1, 3, perm_11, batch)
        self.bcn12 = nn.BatchNorm2d(512)
        self.conv13 = DeformConv2d(512, 512, 3, padding=1)
        self.offset_13, perm_13 = calculate_offset(self.offset_12.shape[-1], 1, 1, 3, perm_12, batch)
        self.bcn13 = nn.BatchNorm2d(512)
        self.pool5 = DeformMaxPool2d(self.offset_13.shape[-1], perm_13, 2)


        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        
        # conv1
        x = self.conv1(x, self.offset_1)
        x = torch.relu(self.bcn1(x))
        x = self.conv2(x, self.offset_2)
        x = self.pool1(torch.relu(self.bcn2(x)))

        # conv2
        x = self.conv3(x, self.offset_3)
        x = torch.relu(self.bcn3(x))
        x = self.conv4(x, self.offset_4)
        x = self.pool2(torch.relu(self.bcn4(x)))

        # conv3
        x = self.conv5(x, self.offset_5)
        x = torch.relu(self.bcn5(x))
        x = self.conv6(x, self.offset_6)
        x = torch.relu(self.bcn6(x))
        x = self.conv7(x, self.offset_7)
        x = self.pool3(torch.relu(self.bcn7(x)))

        # conv4
        x = self.conv8(x, self.offset_8)
        x = torch.relu(self.bcn8(x))
        x = self.conv9(x, self.offset_9)
        x = torch.relu(self.bcn9(x))
        x = self.conv10(x, self.offset_10)
        x = self.pool4(torch.relu(self.bcn10(x)))

        # conv5
        x = self.conv11(x, self.offset_11)
        x = torch.relu(self.bcn11(x))
        x = self.conv12(x, self.offset_12)
        x = torch.relu(self.bcn12(x))
        x = self.conv13(x, self.offset_13)
        output = self.pool5(torch.relu(self.bcn13(x)))

        #output = self.pool5.new_perm.ordenar(output) #order latent space

        #torch.save(output, 'output_deformed.pt')

        output = output.view(output.size()[0], -1)
        output = self.classifier(output)


        return output


def vgg16_bn_deformable(key, batch):
    return VGG_deformable(key, batch)