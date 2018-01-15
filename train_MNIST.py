# Experiments for proof of concept of stochastic weighted function norm regularization
# See https://arxiv.org/pdf/1710.06703.pdf for more details

from __future__ import print_function
import argparse
from os import path as osp
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils import data

from MNIST import model, dataset
from VAE import VAE
from utils import Tictoc
tic, toc = Tictoc()
import pytorch_utils
import utils


# Training settings
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    def comma_sep_int(s): # comma separated tuple
        h, w = map(int, s.split(','))
        return (h, w)

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--results-dir", type=str, default='/esat/kochab/arannen/Pytorch_fun_regularization_generation1',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--expname", type=str, default='MNIST_100')
    parser.add_argument("--model-type", type = str, default= 'LeNet')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0,
                    help='weight decay')
    parser.add_argument('--dropout', type=bool, default=False,
                    help='use dropout or not')
    parser.add_argument('--batch-normalization', type=bool, default=True,
                        help='use batch normalization or not')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--save-every', type=int, default=1000, metavar='N',
                        help='how many batches to wait before saving')
    parser.add_argument('--data-portion', type=int, default=100,
                    help='size of training data' )
    parser.add_argument('--num-it', type=int, default=10000,
                        help='determines the number of iterations for main training')
    parser.add_argument('--glob-restore', type=str, default=0,
                        help="Restore from: {iter_number, exp_dir/iter_number} (use -1 for last, 0 for init)")
    parser.add_argument('--function-norm', type=bool, default=True,
                    help='use function norm regularization or not')
    parser.add_argument('--reg-batch-size', type=int, default=100,
                        help='input batch size for regularization (default: 64)')
    parser.add_argument('--VAE-expname', type=str, default= 'MNIST-VAE' )
    parser.add_argument('--VAE-x-dim', type=int, default=784,
                        help='dimension of VAE first hidden layer')
    parser.add_argument('--VAE-h-dim', type=int, default=128,
                    help='dimension of VAE first hidden layer')
    parser.add_argument('--VAE-z-dim', type=int, default=100,
                    help='dimension of VAE latent space')
    parser.add_argument('--VAE-lr', type=float, default=0.001,
                        help='VAE learning rate (default: 0.001)')
    parser.add_argument('--VAE-num-it', type=int, default=10000,
                        help='VAE iterarion number (default: 1000)')
    parser.add_argument('--VAE-batch-size', type=int, default=20,
                        help='input batch size for VAE training (default: 64)')
    parser.add_argument('--VAE-restore', type=str, default=0,
                        help="Restore from: {iter_number, exp_dir/iter_number} (use -1 for last, 0 for init)")
    parser.add_argument('--reg-lambda', type=float, default=0.1,
                        help='regularization parameter (default: 0.01)')
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parsed = parser.parse_args()
    return parsed

# global variables
args = get_arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


# Models:
def get_models():
    netVAE = VAE.VanillaVAE(args.VAE_x_dim, args.VAE_h_dim, args.VAE_z_dim, dtype=dtype)
    if args.model_type == 'Net':
        net = model.Net(args.dropout)
    elif args.model_type == 'LeNet':
        net = model.LeNet(args.batch_normalization)
    else:
        raise SystemExit("Unknown model type")
    return net, netVAE


# Data:
def get_data():
    path = osp.join(args.results_dir, 'MNIST_data')
    train_data = dataset.MNIST_subset(path, args.data_portion)
    return train_data


# Some utils for training :
def restore_or_init(net, logger, dest_dir, args):
    from_scratch = False
    if utils.isint(args.restore):
        restore_from, restore_iter = (dest_dir, args.restore)
        restore_fromthis = True
    else:
        restore_from, restore_iter = utils.parent_dir(args.restore)
        if not osp.isabs(restore_from):
            restore_from = osp.join(utils.parent_dir(dest_dir)[0], restore_from)
        restore_fromthis = False
    saved = utils.get_saves(restore_from)
    restore_iter = int(restore_iter)
    if restore_iter == -1:
        if saved:
            start_iter, iter_dir = saved[-1]
        else:
            if restore_fromthis:
                from_scratch = True
            else:
                raise ValueError('No checkpoints found in {}'.format(restore_from))
    else:
        for start_iter, iter_dir in saved:
            if start_iter == restore_iter:
                break
        else:
            if restore_iter == 0:
                from_scratch = True
            else:
                raise ValueError('Checkpoint {} not found in {}'.format(restore_iter, restore_from))
    if from_scratch:
        start_iter = 0
    if not from_scratch:
        snap_dest = osp.join(iter_dir, 'state_dict.pth')  # map to cpu in case the optim was done with different devices
        print("Restoring net and logger state from", snap_dest)
        saved_state_dict = torch.load(snap_dest, map_location=lambda storage, loc: storage)
        if hasattr(saved_state_dict,'_OrderedDict__root'):
            load_weights(net, saved_state_dict)
        else:
            net.initialize_from_file(snap_dest)
        logger.restore(iter_dir)
    return start_iter


def load_weights(net, saved_state_dict):

    net.load_state_dict(saved_state_dict)
    return net


def cast(object, dtype='float'):
    if args.gpu >= 0 and not args.no_cuda:
        object = object.cuda(args.gpu)
    else:
        object = object.cpu()
    return getattr(object, dtype)()


def set_eval(net):
    net.eval()


# Train the VAE:
def trainVAE(netVAE, logger, start_iter, train_iterator, optimizer, next_stop, train_loader):

    netVAE.train()
    for it  in range(start_iter, next_stop+1):
        tic()
        # Get batch
        # Needed because the batch size*number of iterations < data size, as we use a small subset of the data
        try:
            X, _ = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            X, _ = next(train_iterator)
        X = dataset.to_VAE(X).type(dtype)

        # Forward
        X_sample, z_mu, z_var= netVAE(X)
        loss = VAE.VAE_loss(X, X_sample, z_mu, z_var)

        # Backward
        loss.backward()

        # Update
        optimizer.step()
        elapsed = toc()

        # Update logger
        logger.log('train', it, 'train_loss', loss.data[0])

        # Housekeeping
        for p in netVAE.parameters():
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())

        # Logging training status
        if it % args.log_interval== 0:
            print('Iter-{}; Loss: {:.4}; [{:.1f}Hz]'.format(it, loss.data[0], args.VAE_batch_size/elapsed))
    if it == next_stop:
        return it, netVAE, z_mu.mean(0), z_var.mean(0)


# Main train function between start_iter and next_stop
def train(net, logger, start_iter, train_iterator, optimizer, next_stop, train_loader, args, netVAE=None, z_mu_all=None, z_var_all=None):
    net.train()
    for it in range(start_iter, next_stop+1):
        tic()
        # Get training batch
        # Needed because the batch size*number of iterations < data size, as we use a small subset of the data
        try:
            data, target = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            data, target = next(train_iterator)
        data = data.type(dtype)
        target = target.type(torch.cuda.LongTensor) if args.cuda else target.type(torch.LongTensor)
        data, target = Variable(data), Variable(target)
        if data.dim()==2:
            data = data.view(-1,1,28,28)
        # Get regularization batch
        if args.function_norm:
            z= VAE.sample_z_gen(z_mu_all, z_var_all, args.reg_batch_size, args.VAE_z_dim, dtype = dtype)
            dataReg = netVAE.P(z)
            dataReg = Variable(dataReg.data.type(dtype))
            if dataReg.dim() == 2:
                dataReg = dataReg.view(-1, 1, 28, 28)

        # Forward
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        if args.function_norm:
            reg = net.fun_norm(dataReg)
            obj = loss +reg*args.reg_lambda
        else:
            obj = loss

        # Backward
        obj.backward()

        # Update
        optimizer.step()
        elapsed = toc()

        # Update logger
        if args.function_norm:
            logger.log('train', it, 'train_loss', loss.data[0])
            logger.log('train', it, 'estim_norm', reg.data[0])
        else:
            logger.log('train', it, 'train_loss', loss.data[0])

        # Logging training status
        if it % args.log_interval == 0:
            if args.function_norm:
                print('Iter-{}; Loss: {:.4} Est. norm: {:.4}; [{:.1f}Hz]'.format(it, loss.data[0], reg.data[0],  args.batch_size / elapsed))
            else:
                print('Iter-{}; Loss: {:.4}; [{:.1f}Hz]'.format(it, loss.data[0], args.batch_size / elapsed))
    if it == next_stop:
        return it, net


# Test model
def test(net, logger, train_iter):
    net.eval()
    test_loss = 0
    correct = 0
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Test data
    path = osp.join(args.results_dir,'MNIST_data')
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Main loop
    for data, target in test_loader:
        # Get batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # Forward
        output = net(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss

        # Get predictions and number of correct predictions
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    # Update logger
    logger.log('test', train_iter, 'test_loss', test_loss)
    logger.log('test', train_iter, 'Accuracy', 100. * correct / len(test_loader.dataset))

    # Log test loss and accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


# Global training function used in pur experiments
def main_train(train_dataset, weight_decay = args.weight_decay, useFN = args.function_norm, useDropout = args.dropout, useBN = args.batch_normalization):
    args.weight_decay = weight_decay
    args.function_norm = useFN
    args.dropout = useDropout
    args.batch_normalization = useBN
    expname = args.expname
    args.expname += '_'+ str(args.function_norm)+'_'+str(args.batch_normalization)+'_'+str(args.dropout)+'_'+str(args.weight_decay)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Prepare directories, models and loggers
    dest_dir = osp.join(args.results_dir, expname, args.expname)
    utils.ifmakedirs(dest_dir)
    print('saving to ' + dest_dir)
    if useFN:
        dest_dir_VAE = osp.join(args.results_dir, expname, args.VAE_expname)
        utils.ifmakedirs(dest_dir_VAE)
        print('saving VAE to ' + dest_dir_VAE)
        net, netVAE = get_models()
        loggerVAE = pytorch_utils.StatsLogger()
        args.restore = args.VAE_restore
        start_iter_VAE= restore_or_init(netVAE, loggerVAE, dest_dir_VAE, args)
        cast(netVAE)
    else:
        net,_ = get_models()
        netVAE = None
        z_mu_all = None
        z_var_all = None
    logger = pytorch_utils.StatsLogger()
    args.restore = args.glob_restore
    start_iter = restore_or_init(net, logger, dest_dir, args)
    cast(net)

    # Training loader
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, drop_last=True,**kwargs)

    # Train VAE if needed
    if useFN:
        train_loader_VAE = data.DataLoader(train_dataset, batch_size=args.VAE_batch_size, shuffle = True, drop_last=True,**kwargs)
        optimizerVAE = optim.Adam(list(netVAE.parameters()), lr=args.VAE_lr)
        iter_stops = []
        for it in range(start_iter_VAE, args.VAE_num_it):
            if (it + 1) % args.save_every == 0:
                iter_stops.append(it)
        if len(iter_stops) > 0:
            assert start_iter_VAE < iter_stops[0]
            train_iterator = iter(train_loader_VAE)
            next_iter = start_iter_VAE
            it = start_iter_VAE - 1
            for next_stop in iter_stops:
                if next_stop > it:
                        it, netVAE, z_mu_all, z_var_all = trainVAE(netVAE, loggerVAE, next_iter, train_iterator, optimizerVAE, next_stop, train_loader_VAE)
                next_iter = it + 1
                iter_dir = osp.join(dest_dir_VAE, str(next_iter))
                utils.ifmakedirs(iter_dir)
                snap_dest = osp.join(iter_dir, 'state_dict.pth')
                print('saving snapshot at {}...'.format(snap_dest))
                torch.save(netVAE.parameters(), snap_dest)
                mu_dest = osp.join(iter_dir, 'z_mu.npy')
                var_dest = osp.join(iter_dir, 'z_var.npy')
                if args.cuda:
                    np.save(mu_dest, z_mu_all.data.cpu().numpy())
                    np.save(var_dest, z_var_all.data.cpu().numpy())
                else:
                    np.save(mu_dest, z_mu_all.data.numpy())
                    np.save(var_dest, z_var_all.data.numpy())
                loggerVAE.save(iter_dir)
        else:
            mu_path = osp.join(dest_dir_VAE,str(args.VAE_num_it),'z_mu.npy')
            var_path = osp.join(dest_dir_VAE,str(args.VAE_num_it),'z_var.npy')
            z_mu_all = Variable(torch.from_numpy(np.load(mu_path)).type(dtype))
            z_var_all = Variable(torch.from_numpy(np.load(var_path)).type(dtype))
        args.VAE_restore = -1

    # Main tarining
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    iter_stops = []
    for it in range(start_iter, args.num_it):
        if (it + 1) % args.save_every == 0:
            iter_stops.append(it)
    if len(iter_stops) > 0:
        assert start_iter < iter_stops[0]
        train_iterator = iter(train_loader)
        next_iter = start_iter
        it = start_iter - 1
        for next_stop in iter_stops:
            if next_stop > it:
                it, net = train(net, logger, next_iter, train_iterator, optimizer, next_stop, train_loader, args, netVAE, z_mu_all, z_var_all)
            next_iter = it + 1
            iter_dir = osp.join(dest_dir, str(next_iter))
            utils.ifmakedirs(iter_dir)
            snap_dest = osp.join(iter_dir, 'state_dict.pth')
            print('saving snapshot at {}...'.format(snap_dest))
            torch.save(net.state_dict(), snap_dest)
            test(net, logger, it)
            logger.save(iter_dir)
    args.expname = expname


def main():
    expname= args.expname
    for i in range(10):
        train_dataset = get_data()

        args.model_type = 'LeNet' # To test function norm vs. batch norm
        args.expname = expname +'_'+ str(i+1)+ '_'+ args.model_type
        main_train(train_dataset, 0, False, False, False)
        main_train(train_dataset, 0, True, False, False)
        main_train(train_dataset, 0, False, False, True)
        main_train(train_dataset, 0, True, False, True)
        main_train(train_dataset, 0.0005, False, False, False)
        main_train(train_dataset, 0.0005, True, False, False)
        main_train(train_dataset, 0.0005, False, False, True)
        main_train(train_dataset, 0.0005, True, False, True)

        args.model_type = 'Net' # To test function norm vs. dropout
        args.expname = expname+'_'+str(i+1)+'_'+args.model_type
        main_train(train_dataset, 0, False, False, False)
        main_train(train_dataset, 0, True, False, False)
        main_train(train_dataset, 0, False, True, False)
        main_train(train_dataset, 0, True, True, False)
        main_train(train_dataset, 0.0005, False, False, False)
        main_train(train_dataset, 0.0005, True, False, False)
        main_train(train_dataset, 0.0005, False, True, False)
        main_train(train_dataset, 0.0005, True, True, False)


if __name__ == '__main__':
    main()
