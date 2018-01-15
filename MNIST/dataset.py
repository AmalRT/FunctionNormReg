# select a random subset of MNIST
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import datasets, transforms, models
import torch.nn.functional as F


def to_VAE(data):
    return F.sigmoid(data.view(-1,784))


class MNIST_subset(data.Dataset):

    def __init__(self, root, size):
        self.root = root
        self.size = size
        self.sampler = data.DataLoader(datasets.MNIST(self.root, train=True, download=True,
                  transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                  batch_size = self.size, shuffle=True)
        self.data = next(iter(self.sampler))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        data = self.data[0][item,:,:,:]
        lab = self.data[1][item]
        return data, lab


if __name__ == '__main__':
    root = '/users/visics/arannen/MNIST_data' #change path
    dataset = MNIST_subset(root, 600)
    trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True)
    train_it = iter(trainloader)
    im, lab = next(train_it)
    im = to_VAE(im)
    im = im.data.numpy()
    plt.imshow(im[0,:].reshape(28,28), cmap='gray')
    print(lab)