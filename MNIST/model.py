from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, useDropout = True):
        super(Net, self).__init__()
        self.useDropout = useDropout
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        if self.useDropout:
            self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, useDropout=None, isReg = False):
        if useDropout == None:
            useDropout = self.useDropout
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x= self.conv2(x)
        if useDropout:
            x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        else:
            x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if useDropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if isReg:
            return x
        else:
            return F.log_softmax(x)

    def fun_norm(self,x):
        out = self(x,useDropout=False, isReg=True)
        return torch.matmul(out, torch.t(out)).diag().mean()

    def grad_norm(self, x, dtype):
        out = self(x, useDropout=False, isReg=True)
        grad = torch.autograd.grad(out, x, grad_outputs=Variable(torch.ones(out.size()).type(dtype)), create_graph=True)
        grad = grad[0].view(-1, 784)
        return torch.matmul(grad, torch.t(grad)).diag().mean()

    def sobolev_norm(self, x, dtype):
        out = self(x,useDropout=False, isReg=True)
        grad = torch.autograd.grad(out, x, grad_outputs=Variable(torch.ones(out.size()).type(dtype)), create_graph=True)
        grad = grad[0].view(-1, 784)
        return (torch.matmul(out, torch.t(out)) + torch.matmul(grad, torch.t(grad))).diag().mean()


class LeNet(nn.Module):
    def __init__(self, useBN = True, FuncNorm = True):
        super(LeNet, self).__init__()
        self.useBN = useBN
        self.FuncNorm = FuncNorm
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.conv3 = nn.Conv2d(50, 500, kernel_size=4)
        if self.useBN:
            self.BN1 = nn.BatchNorm2d(20)
            self.BN2 = nn.BatchNorm2d(50)
            self.BN3 = nn.BatchNorm2d(500)
            if self.FuncNorm:
                self.BN1_reg = nn.BatchNorm2d(20)
                self.BN2_reg = nn.BatchNorm2d(50)
                self.BN3_reg = nn.BatchNorm2d(500)
        self.fc = nn.Linear(500,10)

    def forward(self, x, isReg = False):
        if self.useBN:
            if isReg:
                x = self.BN1_reg(F.max_pool2d(self.conv1(x), 2))
                x = self.BN2_reg(F.max_pool2d(self.conv2(x), 2))
                x = self.BN3_reg(self.conv3(x))
            else:
                x = self.BN1(F.max_pool2d(self.conv1(x), 2))
                x = self.BN2(F.max_pool2d(self.conv2(x), 2))
                x = self.BN3(self.conv3(x))
        else:
            x = F.max_pool2d(self.conv1(x), 2)
            x = F.max_pool2d(self.conv2(x), 2)
            x = self.conv3(x)
        x = x.view(-1,500)
        x = self.fc(x)
        if isReg:
            return x
        else:
            return F.log_softmax(x)

    def fun_norm(self,x):
        out = self(x, isReg=True)
        return torch.matmul(out, torch.t(out)).diag().mean()

    def grad_norm(self,x, dtype):
        out = self(x, isReg=True)
        grad = torch.autograd.grad(out, x, grad_outputs=Variable(torch.ones(out.size()).type(dtype)), create_graph=True)
        grad = grad[0].view(-1, 784)
        return torch.matmul(grad, torch.t(grad)).diag().mean()

    def sobolev_norm(self,x, dtype):
        out = self(x, isReg=True)
        grad = torch.autograd.grad(out, x, grad_outputs=Variable(torch.ones(out.size()).type(dtype)), create_graph=True)
        grad = grad[0].view(-1, 784)
        return (torch.matmul(out, torch.t(out)) + torch.matmul(grad, torch.t(grad))).diag().mean()