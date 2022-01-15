# %%
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchsummary import summary
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np

# %%
from generator import Generator

# %%
device = torch.device('cuda:1')

# %%
generator = Generator(input_nc=6, output_nc=3)
generator = generator.to(device)

# %%
class Decoder(nn.Module):
    def __init__(self, nc=3, nhf=32):
        super(Decoder, self).__init__()
        # input is (3) x 32 x 32
        dropout = 0.3
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output=self.main(input)
        return output

decoder = Decoder().to(device)

# %%
# Build ResNet 18 Module 

# https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

classifier = ResNet18()
classifier = classifier.to(device)

# %%
optG = optim.Adam(generator.parameters(), lr = 1e-4)


# %%
from utils import *
cifar100_train_loader, cifar100_test_loader = prepare_dataloader(dataset=CIFAR100)
cifar10_train_loader, cifar10_test_loader = prepare_dataloader(dataset=CIFAR10)

# %%
label = next(iter(cifar10_train_loader))[1]

# %%
c = torch.rand(128, 10)
l = nn.CrossEntropyLoss()
l(torch.sigmoid(c), label)

# %%
def train_classifier(generator, train_loader, classifier, optim, device):
    generator.eval()
    classifier.train()
    losses = []
    corrected = 0
    total = 0
    for image, labels in train_loader:
        classifier.zero_grad()
        
        # Select half dataset for container images
        pivot = len(image) 
        container_image = torch.zeros(image.shape)

        # Create input of generator (6-dims: batch_size x 6 x 32 x 32)
        gen_input = torch.cat([container_image, image], dim=1).to(device)

        with torch.no_grad():
            privacy_image = generator(gen_input)

        pred_labels = classifier(privacy_image)
        labels = labels.to(device)
        loss = F.cross_entropy(pred_labels, labels)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        _, pred_labels = pred_labels.max(1)
        corrected += (pred_labels == labels).sum()
        total += pivot

    return classifier, np.mean(losses), corrected / total
        

# %%
def train_decoder(generator, train_loader, decoder, optim, scale, device):
    generator.eval()
    decoder.train()
    losses = []

    for img, _ in train_loader:
        decoder.zero_grad()
        container_image = torch.zeros(img.shape)
        gen_input = torch.cat([container_image, img], dim=1).to(device)
        with torch.no_grad():
            privacy_image = generator(gen_input) * scale

        pred_image = decoder(privacy_image)
        loss = F.binary_cross_entropy(pred_image, img.to(device))
        loss.backward()
        optim.step()
        losses.append(loss.item())

    return decoder, np.mean(losses)

# %%
def test_classifier(generator, test_loader, classifier, device):
    generator.eval()
    classifier.eval()
    losses = []
    corrected = 0
    total = 0
    for img, labels in test_loader:
        container = torch.zeros(img.shape)
        gen_input = torch.cat([container, img], dim = 1).to(device)

        privacy_img = generator(gen_input)
        pred_labels = classifier(privacy_img)
        labels = labels.to(device)
        loss = F.cross_entropy(pred_labels, labels)
        losses.append(loss.item())
        _, pred_labels = pred_labels.max(1)
        corrected += (pred_labels == labels).sum()
        total += len(img)

    return np.mean(losses), corrected / total


# %%
def test_decoder(generator, test_loader, decoder, scale, device):
    generator.eval()
    decoder.eval()
    losses = []
    losses_bce = []
    for img, _ in test_loader:
        container = torch.zeros(img.shape)
        gen_input = torch.cat([container, img], dim = 1).to(device)

        privacy_img = generator(gen_input) * scale
        pred_img = decoder(privacy_img)
        loss = F.mse_loss(pred_img, img.to(device))
        bce = F.binary_cross_entropy(pred_img, img.to(device))
        losses.append(loss.item())
        losses_bce.append(bce.item())

    return np.mean(losses), np.mean(losses_bce)

# %%
def train_generator_epoch(generator, train_loader, optim, device):
    generator.train()
    losses = []
    for img, _ in train_loader:
        generator.zero_grad()
        pivot = len(img) // 2 
        container = img[0:pivot, :, :, :].to(device)
        real_image = img[pivot:pivot*2, :, :, :].to(device)

        gen_input = torch.cat([container, real_image], dim=1).to(device)
        privacy_img = generator(gen_input)
        loss = F.mse_loss(privacy_img, container)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    return generator, np.mean(losses)

# %%
import logging
def get_logger(path):
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename= path, filemode='w')
    return logging.getLogger()

from tqdm import tqdm
classifier_epoch = 50
decoder_epoch = 30
iterator = tqdm(range(100))
c_acc_list = []
d_loss_list = []
d_loss_bce_list = []
g_loss_list = []
base_path = 'checkpoints/trade_off_1e-4/'
g_logger = get_logger(base_path + 'generator_log.txt')
d_logger = get_logger(base_path + 'decoder_log.txt')
c_logger = get_logger(base_path + 'classifier_log.txt')

for i in iterator:
    classifier = ResNet18()
    classifier = classifier.to(device)
    decoder = Decoder().to(device)
    optD = optim.Adam(decoder.parameters(), lr = 1e-3, weight_decay= 1e-3)
    optC = optim.Adam(classifier.parameters(), lr = 1e-3)
    
    generator, g_loss = train_generator_epoch(generator, cifar10_train_loader, optG, device)
    g_loss_list.append(g_loss)
    g_logger.info(g_loss)
    c_count = - 1
    c_max_acc = 0
    print(f"Train {i}th classifier ")
    for j in range(classifier_epoch):
        classifier, loss, acc = train_classifier(generator, cifar10_train_loader, classifier, optC, device)
        c_eval_loss, c_eval_acc = test_classifier(generator, cifar10_test_loader, classifier, device)
        if c_eval_acc > c_max_acc:
            torch.save(classifier.state_dict(), f'checkpoints/trade_off_1e-4/classifier_{i}.pth')
            c_count += 1
            c_max_acc = c_eval_acc
        if c_count >= 5:
            break
    # c_eval_loss, c_eval_acc = test_classifier(generator, cifar10_test_loader, classifier, device)
    c_acc_list.append(c_max_acc)
    c_logger.info(c_max_acc.cpu().numpy())

    d_count = -1
    d_min_loss = 100
    d_min_bce = 100
    print(f"Train {i}th decoder ")
    for j in range(decoder_epoch):
        decoder, loss = train_decoder(generator, cifar100_train_loader, decoder, optD, 10, device)
        d_eval_loss, d_eval_loss_bce = test_decoder(generator, cifar100_test_loader, decoder, 10, device)
        if d_eval_loss < d_min_loss:
            d_count += 1
            d_min_loss = d_eval_loss
            d_min_bce = d_eval_loss_bce
            torch.save(decoder.state_dict(), f'checkpoints/trade_off_1e-4/decoder_{i}.pth')
        if d_count >= 5:
            break
    # d_eval_loss, d_eval_loss_bce = test_decoder(generator, cifar100_test_loader, decoder, 10, device)
    d_loss_list.append(d_min_loss)
    d_logger.info(d_min_loss)
    d_logger.info(d_min_bce)

    torch.save(generator.state_dict(), f'checkpoints/trade_off_1e-4/generator_{i}.pth')
    
    
    iterator.set_description(f"Epoch: {i}, Generator's loss: {g_loss}, Decoder's loss: {d_eval_loss}, Classifier's acc: {c_eval_acc}")
