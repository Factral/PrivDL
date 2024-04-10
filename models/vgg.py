"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import math

cfg = {
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
}

class VGG(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()

        # conv1
        self.conv1 =  nn.Conv2d(3, 64, 3, padding=1)
        self.bcn1 = nn.BatchNorm2d(64)
        self.conv2 =  nn.Conv2d(64, 64, 3, padding=1)
        self.bcn2 = nn.BatchNorm2d(64)
        
        # conv2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bcn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bcn4 = nn.BatchNorm2d(128)

        # conv3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bcn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bcn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bcn7 = nn.BatchNorm2d(256)

        # conv4
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bcn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bcn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bcn10 = nn.BatchNorm2d(512)

        # conv5
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bcn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bcn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bcn13 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


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
        x = self.conv1(x)
        x = torch.relu(self.bcn1(x))
        x = self.conv2(x)
        x = self.pool(torch.relu(self.bcn2(x)))

        # conv2
        x = self.conv3(x)
        x = torch.relu(self.bcn3(x))
        x = self.conv4(x)
        x = self.pool(torch.relu(self.bcn4(x)))

        # conv3
        x = self.conv5(x)
        x = torch.relu(self.bcn5(x))
        x = self.conv6(x)
        x = torch.relu(self.bcn6(x))
        x = self.conv7(x)
        x = self.pool(torch.relu(self.bcn7(x)))

        # conv4
        x = self.conv8(x)
        x = torch.relu(self.bcn8(x))
        x = self.conv9(x)
        x = torch.relu(self.bcn9(x))
        x = self.conv10(x)
        x = self.pool(torch.relu(self.bcn10(x)))

        # conv5
        x = self.conv11(x)
        x = torch.relu(self.bcn11(x))
        x = self.conv12(x)
        x = torch.relu(self.bcn12(x))
        x = self.conv13(x)
        output = self.pool(torch.relu(self.bcn13(x)))

        #torch.save(output, 'output.pt')

        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def vgg16_bn():
    return VGG()