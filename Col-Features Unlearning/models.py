#!/usr/bin/python3

"""models.py Contains an implementation of the LeNet5 model

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init, BasicBlock, unlearn_weights_init
import torch


class LeNetComplete(nn.Module):
    """CNN based on the classical LeNet architecture, but with ReLU instead of
    tanh activation functions and max pooling instead of subsampling."""
    def __init__(self):
        super(LeNetComplete, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Define forward pass of CNN

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = self.block1(x)

        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        x = x.view(-1, 4*4*16)

        # Apply first fully-connected block to input tensor
        x = self.block3(x)

        return F.log_softmax(x, dim=1)



class LeNetClientNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ClientNetwork is used for Split Learning and implements the CNN
    until the first convolutional layer."""
    def __init__(self):
        super(LeNetClientNetwork, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        """Defines forward pass of CNN until the split layer, which is the first
        convolutional layer

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = self.block1(x)

        return x


class LeNetServerNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ServerNetwork is used for Split Learning and implements the CNN
    from the split layer until the last."""
    def __init__(self):
        super(LeNetServerNetwork, self).__init__()

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Defines forward pass of CNN from the split layer until the last

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        #x = x.view(-1, 4*4*16)
        x = x.view(x.size(0), -1)

        # Apply fully-connected block to input tensor
        x = self.block3(x)

        return x


"""
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


def resnet110(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[18, 18, 18], kernel_size=kernel_size, num_classes=num_classes)


def resnet56(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[9, 9, 9], kernel_size=kernel_size, num_classes=num_classes)


class BottomModelForDiabetes(nn.Module):
    def __init__(self, in_size, n_clients):
        super(BottomModelForDiabetes, self).__init__()
        self.fc1 = nn.Linear(in_size, 8)
        self.fc2 = nn.Linear(8, 2)
        self.bn1 = nn.BatchNorm1d(8)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        return x


class TopModelForDiabetes(nn.Module):
    def __init__(self, bottom_output_size=4, n_clients=5):
        super(TopModelForDiabetes, self).__init__()
        self.fc1top = nn.Linear(bottom_output_size*n_clients, 2)
        # self.fc2top = nn.Linear(8, 2)
        # self.fc3top = nn.Linear(32, 16)
        # self.fc4top = nn.Linear(16, 2)
        self.bn1top = nn.BatchNorm1d(bottom_output_size*n_clients)
        # self.bn2top = nn.BatchNorm1d(32)
        # self.bn3top = nn.BatchNorm1d(16)
        # self.dropout = nn.Dropout(0.8)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1top(self.bn1top(x))
        # x = self.dropout(x)
        # x = self.fc2top(F.relu(self.bn1top(x)))
        # x = self.fc3top(F.relu(self.bn2top(x)))
        # x = self.fc4top(F.relu(self.bn3top(x)))
        return x


D_ = 28
class BottomModelForCriteo(nn.Module):

    def __init__(self, in_size, n_clients):
        super(BottomModelForCriteo, self).__init__()
        self.fc1 = nn.Linear(in_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.8)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


class BottomModelForBCW(nn.Module):

    def __init__(self, in_size, n_clients):
        super(BottomModelForBCW, self).__init__()
        self.fc1 = nn.Linear(in_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)
        self.bn1 = nn.BatchNorm1d(20)
        # self.bn2 = nn.BatchNorm1d(20)
        self.apply(weights_init)

    def forward(self, x):
        # x = F.linear(x, self.weight, self.bias)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class BottomModelForAdult(nn.Module):
    def __init__(self, in_size, n_clients):
        super(BottomModelForAdult, self).__init__()
        self.fc1 = nn.Linear(in_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class TopModelForCriteo(nn.Module):
    def __init__(self, bottom_output_size=4, n_clients=5):
        super(TopModelForCriteo, self).__init__()
        self.fc1_top = nn.Linear(n_clients*bottom_output_size, 16)
        self.fc2_top = nn.Linear(16, 8)
        self.fc3_top = nn.Linear(8, 4)
        self.fc4_top = nn.Linear(4, 2)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1_top(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        x = F.relu(x)
        x = self.fc4_top(x)
        return x


class TopModelForBCW(nn.Module):
    def __init__(self, bottom_output_size, n_clients):
        super(TopModelForBCW, self).__init__()
        self.fc1_top = nn.Linear(bottom_output_size*n_clients, 2)
        # self.bn0_top = nn.BatchNorm1d(bottom_output_size*n_clients)
        self.apply(weights_init)

    def forward(self, x):
        # x = self.bn0_top(x)
        # x = F.relu(x)
        x = self.fc1_top(x)
        return x


class TopModelForAdult(nn.Module):
    def __init__(self, bottom_output_size, n_clients):
        super(TopModelForAdult, self).__init__()
        self.fc1_top = nn.Linear(bottom_output_size * n_clients, 2)
        self.bn0_top = nn.BatchNorm1d(bottom_output_size * n_clients)
        self.apply(weights_init)

    def forward(self, x):
        x = self.bn0_top(x)
        x = F.relu(x)
        x = self.fc1_top(x)
        return x


class BottomModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, in_size, n_clients=10):
        if self.dataset_name == 'Diabetes':
            return BottomModelForDiabetes(in_size, n_clients)
        elif self.dataset_name == 'Criteo':
            return BottomModelForCriteo(in_size, n_clients)
        elif self.dataset_name == 'BCW':
            return BottomModelForBCW(in_size, n_clients)
        elif self.dataset_name == 'Adult':
            return BottomModelForAdult(in_size, n_clients)
        else:
            raise Exception('Unknown dataset name!')

    def __call__(self):
        raise NotImplementedError()


class TopModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, n_clients):
        if self.dataset_name == 'Diabetes':
            client_out_size = 2
            return TopModelForDiabetes(client_out_size, n_clients)
        elif self.dataset_name == 'Criteo':
            client_out_size = 16
            return TopModelForCriteo(client_out_size, n_clients)
        elif self.dataset_name == 'BCW':
            client_out_size = 2
            return TopModelForBCW(client_out_size, n_clients)
        elif self.dataset_name == 'Adult':
            client_out_size = 2
            return TopModelForAdult(client_out_size, n_clients)
        else:
            raise Exception('Unknown dataset name!')


def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return


if __name__ == "__main__":
    demo_model = BottomModel(dataset_name='CIFAR10')
    print(demo_model)

