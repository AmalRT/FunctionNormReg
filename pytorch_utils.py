#Author: Maxim Berman

from __future__ import division, print_function

import collections
import os.path as osp
import torch
import torch.nn.functional as F
import six
from torch.autograd import Variable
import numpy as np
import utils
import pickle
from datetime import datetime
from collections import OrderedDict, namedtuple



# define softmax with argument for backward compatibility
try:
    F.softmax(Variable(torch.ones(1, 1)), 0)
    softmax = F.softmax
except TypeError:
    def softmax(logits, dim):
        softmax_dim = int((logits.dim() != 0) and (logits.dim() != 3))
        if softmax_dim != dim:
            raise ValueError(("softmax dim unsupported on this "
                              "python version (got {}, should be {})").format(dim, softmax_dim))
        return F.softmax(logits)


def pkl_dump(obj, path):
    with utils.write(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def tensor_like(tensor):
    return type(tensor)(tensor.size())


class IterTimeCurve(namedtuple('IterTimeCurve', ['x', 't', 'y'])):
    __slots__ = ()

    def __repr__(self):
        return 'IterTimeCurve(x=<{} elements>, t=<{} elements>, y=<{} elements>)'.format(
            len(self.x), len(self.t), len(self.y)
        )


class StatsLogger(object):
    def __init__(self, restore=None, filename='stats.pkl'):
        self.stats = utils.NestedDict()
        self.filename = filename
        if restore is not None:
            self.restore(restore)

    def restore(self, directory, source_filename=None):
        if source_filename is None:
            source_filename = self.filename
        save_file = osp.join(directory, source_filename)
        if osp.isfile(save_file):
            self.stats = pkl_load(save_file)
        else:
            print('Warning: no saved stats found in {}.'.format(directory))
        return self

    def save(self, directory, dest_filename=None):
        if dest_filename is None:
            dest_filename = self.filename
        pkl_dump(self.stats, osp.join(directory, dest_filename))

    def log(self, set, iter, name, value):
        timestamp = datetime.now()
        if isinstance(iter, tuple):
            iter, sub_iter = iter
            self.stats[iter][set][name][sub_iter] = (value, timestamp)
        else:
            self.stats[iter][set][name] = (value, timestamp)

    def __str__(self):
        val_iters = self.select_iters('val')
        out = 'StatsLogger with {} iters'.format(len(self.stats))
        if val_iters:
            summary = self.summary('val', val_iters[-1])
            out += '. Last val iter:\n'
            with utils.printoptions(suppress=True, precision=4):
                summary_str = ['{} = {}'.format(k, v) for (k, v) in summary.items()]
                out += '\n'.join(summary_str)
        return out

    def select_iters(self, set='val', name=None):
        if name is None:
            return sorted([i for (i, sets) in self.stats.items() if set in sets])
        else:
            return sorted([i for (i, sets) in self.stats.items() if set in sets and name in sets[set]])

    def curve(self, name, set='train'):
        iters = []
        times = []
        values = []
        available = self.select_iters(set, name)
        for i in available:
            iters.append(i)
            v, t = self.stats[i][set][name]
            times.append(t)
            values.append(v)
        return IterTimeCurve(iters, times, values)


    def summary(self, set, iter):
        out = OrderedDict()
        logged = self.stats[iter][set]
        for key, value in logged.items():
            if logged.is_leaf(key):
                out[key] = value[0]
            else:
                values = [v[0] for v in value.values()]
                try:
                    out[key + '_mean'] = np.array([np.nanmean(col) for col in zip(*values)])
                    out[key + '_std'] = np.array([np.nanstd(col) for col in zip(*values)])
                    out[key + '_mean_mean'] = np.nanmean(out[key + '_mean'])
                except TypeError: # elements are not arrays themselves
                    out[key + '_mean'] = np.mean(values)
                    out[key + '_std'] = np.std(values)
        return out


def one_hot(indices, C, dtype='LongTensor'):
    """
    Convert a 0 <= LongTensor array < C to a one-hot encoding
    """
    P = len(indices)
    module = torch.cuda if indices.is_cuda else torch
    out = getattr(module, dtype)(P, C).fill_(0)
    out.scatter_(1, indices.view(-1, 1), 1)
    return out


def flip(x, dim):
    """
    Flip a tensor in dimension dim
    Taken from https://github.com/pytorch/pytorch/issues/229#issuecomment-328275690
    """
    dim = x.dim() + dim if dim < 0 else dim
    if x.is_cuda:
        out = x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long().cuda()
             for i in range(x.dim()))]
    else:
        out = x[tuple(slice(None, None) if i != dim
                      else torch.arange(x.size(i) - 1, -1, -1).long()
                      for i in range(x.dim()))]
    return out


class ConfusionMatrix(object):
    """Confusion Matrix compatible with pytorch/numpy
    ignores all GT labels >= nclass.
    """
    def __init__(self, nclass):
        self.nclass = nclass
        self.M = np.zeros((nclass, nclass), int)

    def add(self, gt, pred):
        assert(pred.max() < self.nclass)
        for gt_cat in range(self.nclass):
            gt_mask = (gt == gt_cat)
            for pred_cat in range(self.nclass):
                pred_mask = (pred == pred_cat) & (gt < self.nclass)
                self.M[gt_cat, pred_cat] += (gt_mask & pred_mask).sum()

    def __str__(self):
        with utils.printoptions(suppress=True):
            return str(self.M)

    def recall(self):
        recall = 0.0
        for i in xrange(self.nclass):
            recall += self.M[i, i] / self.M[:, i].sum()
        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in xrange(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])
        return accuracy/self.nclass

    def normalized(self):
        """Normalized version: each line sums to 1"""
        MN = self.M.copy()
        for i in range(self.nclass):
            MN[i, :] = MN[i, :]/MN[i, :].sum()
        return MN

    def jaccard_perclass(self):
        perclass = []
        for i in range(self.nclass):
            positives = self.M[i, :].sum()
            if not positives:
                perclass.append(1.)
            else:
                TP = self.M[i, i]
                FP = self.M[:, i].sum() - TP
                perclass.append(
                    TP / (positives + FP))
        return perclass

    def jaccard(self):
        perclass = self.jaccard_perclass()
        return np.mean(perclass)


def iterable(arg):
    return (isinstance(arg, collections.Iterable)
            and not isinstance(arg, six.string_types)
            and not 'torch' in type(arg).__module__)
