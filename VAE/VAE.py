# Defines VAE architectures
# For our experiments we used only Vanilla-VAE for MNIST
from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def xavier_init(size, dtype = torch.FloatTensor):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    t = torch.randn(*size) * xavier_stddev
    return Variable(t.type(dtype), requires_grad=True)


def sample_z(mu, log_var, dtype = torch.FloatTensor, cst = 1):
    eps = Variable(torch.randn(mu.size()).type(dtype))
    return mu + cst*torch.exp(log_var / 2) * eps


def sample_z_gen(mu, log_var, mb_size, Z_dim, dtype = torch.FloatTensor, cst =3):
    res = Variable(torch.zeros(mb_size,Z_dim).type(dtype))
    for i in range(mb_size):
        eps = Variable(torch.randn(mu.size()).type(dtype))
        res[i,:] =mu + cst*torch.exp(log_var / 2) * eps
    return res


def VAE_loss(X, X_sample, z_mu, z_var):
    recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False) / X.size(0)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss
    return loss


# =====================Vanilla-VAE==================================
class VanillaVAE(nn.Module):

    def __init__(self, X_dim, h_dim, Z_dim, dtype = torch.FloatTensor):
        super(VanillaVAE, self).__init__()
        self.X_dim = X_dim
        self.h_dim = h_dim
        self.Z_dim = Z_dim
        self.dtype = dtype
        # =============================== Q(z|X) ======================================
        self.Wxh = xavier_init(size=[self.X_dim, self.h_dim],dtype=self.dtype)
        self.bxh = Variable(torch.zeros(self.h_dim).type(self.dtype), requires_grad=True)

        self.Whz_mu = xavier_init(size=[self.h_dim, self.Z_dim],dtype=self.dtype)
        self.bhz_mu = Variable(torch.zeros(self.Z_dim).type(self.dtype), requires_grad=True)

        self.Whz_var = xavier_init(size=[self.h_dim, self.Z_dim],dtype=self.dtype)
        self.bhz_var = Variable(torch.zeros(self.Z_dim).type(self.dtype), requires_grad=True)

        # =============================== P(X|z) ======================================
        self.Wzh = xavier_init(size=[self.Z_dim, self.h_dim],dtype=self.dtype)
        self.bzh = Variable(torch.zeros(self.h_dim).type(self.dtype), requires_grad=True)

        self.Whx = xavier_init(size=[self.h_dim, self.X_dim],dtype=self.dtype)
        self.bhx = Variable(torch.zeros(self.X_dim).type(self.dtype), requires_grad=True)
    
    def Q(self, X):
        h = F.relu(torch.matmul(X, self.Wxh) + self.bxh.repeat(X.size(0), 1))
        z_mu = torch.matmul(h, self.Whz_mu) + self.bhz_mu.repeat(h.size(0), 1)
        z_var = torch.matmul(h, self.Whz_var) + self.bhz_var.repeat(h.size(0), 1)
        return z_mu, z_var

    def P_gen(self,z):
        h = F.relu(torch.matmul(z, self.Wzh) + self.bzh.repeat(z.size(0), 1))
        X = torch.matmul(h, self.Whx) + self.bhx.repeat(h.size(0), 1)
        return X

    def P(self, z):
        h = F.relu(torch.matmul(z, self.Wzh) + self.bzh.repeat(z.size(0), 1))
        X = F.sigmoid(torch.matmul(h, self.Whx) + self.bhx.repeat(h.size(0), 1))
        return X

    def parameters(self):
        return self.Wxh, self.bxh, self.Whz_mu, self.bhz_mu, self.Whz_var, self.bhz_var, self.Wzh, self.bzh, self.Whx, self.bhx

    def forward(self, X):
        z_mu, z_var = self.Q(X)
        z = sample_z(z_mu, z_var, self.dtype)
        X_sample = self.P(z)
        return X_sample, z_mu, z_var

    def initialize_from_file(self,file):
        params = torch.load(file)
        self.Wxh, self.bxh, self.Whz_mu, self.bhz_mu, self.Whz_var, self.bhz_var, self.Wzh, self.bzh, self.Whx, self.bhx = params

    def sample(self, mb_size, cst = 3):
        X = Variable(torch.randn(mb_size,self.X_dim).type(self.dtype))
        z_mu, z_var = self.Q(X)
        z = sample_z(z_mu,z_var, self.dtype, cst)
        X_sample = self.P(z)
        return X_sample


# =====================Convolutional-VAE==================================
class Encoder(nn.Module):
    def __init__(self, X_dim, Z_dim):
        super(Encoder, self).__init__()

        self.X_dim = X_dim
        self.Z_dim = Z_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(128, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.ELU(),

            nn.Conv2d(256, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.ELU(),

            nn.Conv2d(256, 512, 4, 2),
            nn.BatchNorm2d(512),
            nn.ELU(),

            nn.Conv2d(512, 512, 4, 2),
            nn.BatchNorm2d(512),
            nn.ELU(),

            nn.Conv2d(512, self.Z_dim, 4, 2),
            nn.BatchNorm2d(self.Z_dim),
            nn.ELU()
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
        :return: An float tensor with shape of [batch_size, latent_variable_size]
        """

        '''
        Transpose input to the shape of [batch_size, embed_size, seq_len]
        '''
        result = self.cnn(input)
        return result.transpose(1,3).transpose(1,2)


class Decoder(nn.Module):
    def __init__(self, X_dim, Z_dim, batch_size):
        super(Decoder, self).__init__()
        self.X_dim = X_dim
        self.Z_dim = Z_dim
        self.batch_size = batch_size
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(self.Z_dim, 512, 4, 2, 0),
            nn.BatchNorm2d(512),
            nn.ELU(),

            nn.ConvTranspose2d(512, 512, 4, 2, 0),
            nn.BatchNorm2d(512),
            nn.ELU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ELU(),

            nn.ConvTranspose2d(256, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ELU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 0,output_padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.ConvTranspose2d(128,1, 4, 2, 0)
        )

    def forward(self, latent):
        return self.cnn(latent.transpose(1,3).transpose(2,3))


class ConvolutionalVAE(nn.Module):
    def __init__(self, X_dim, Z_dim, batch_size):
        super(ConvolutionalVAE, self).__init__()

        self.X_dim = X_dim
        self.Z_dim = Z_dim
        self.batch_size = batch_size

        self.encoder = Encoder(self.X_dim, self.Z_dim)
        self.W_mu = nn.Linear(self.Z_dim, self.Z_dim)
        self.W_var = nn.Linear(self.Z_dim, self.Z_dim)
        self.decoder = Decoder(self.X_dim, self.Z_dim, self.batch_size)

    def forward(self, X, dtype= torch.FloatTensor):
        h = self.encoder(X)
        z_mu = self.W_mu(h)
        z_var = self.W_var(h)
        z = sample_z(z_mu, z_var, dtype)
        X_sample = self.decoder(z)
        return X_sample, z_mu, z_var


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    mb_size = 32
    Z_dim = 100
    X_dim = mnist.train.images.shape[1]
    y_dim = mnist.train.labels.shape[1]
    h_dim = 128
    c = 0
    lr = 1e-3

    model = ConvolutionalVAE(X_dim, Z_dim)
    print(model)







