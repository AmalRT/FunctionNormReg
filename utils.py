from __future__ import print_function, division

import argparse
import collections
import contextlib
import os
import shutil
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import parse
import yaml
from PIL import Image


def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    https://stackoverflow.com/a/37661854/805502
    """
    return 'ipykernel' in sys.modules


class NestedDict(collections.OrderedDict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]

    def is_leaf(self, key):
        if key not in self:
            raise ValueError('Key {} not found.'.format(key))
        return not isinstance(self[key], NestedDict)


def which_in(obj, *args):
    """
    test if which of the args are in obj
    returns list of args in obj
    """
    out = []
    for a in args:
        if a in obj:
            out.append(a)
    return out


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isint(s):
    """test if string is an int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def mean(l, ignore_nan=False):
    if len(l) == 1:
        if ignore_nan and l[0] == np.nan:
            raise ValueError("Only NaN in mean")
        return l[0]
    else:
        acc = None
        added = 0
        for s in l:
            if ignore_nan and s == np.nan:
                continue
            acc = s if acc is None else acc + s
            added += 1
        return acc / added


# https://stackoverflow.com/questions/2860153/how-do-i-get-the-parent-directory-in-python
def parent_dir(path):
    return os.path.split(os.path.normpath(path))


def get_saves(dest_dir, pattern='{:d}'):
    if not os.path.exists(dest_dir):
        return []
    files = os.listdir(dest_dir)
    p = parse.compile(pattern)
    saves = []
    for s in files:
        parsed = p.parse(s)
        if parsed:
            saves.append((parsed[0], os.path.join(dest_dir, s)))
    return sorted(saves)


def bbox(lbl):
    # https://stackoverflow.com/a/4809040/805502
    B = np.argwhere(lbl)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    return (ystart, ystop), (xstart, xstop)


def ifmakedirs(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def dict_str(data):
    return yaml.dump(data)#, default_flow_style=False)


def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25,
                                                              0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5,
                                                   0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()


# https://stackoverflow.com/questions/2891790/how-to-pretty-printing-a-numpy-array-without-scientific-notation-and-with-given
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


@contextlib.contextmanager
def write(path, flag='w'):
    """Delayed write operation to avoid overwriting data in case of failure."""
    tmp_path = path + '.tmp'
    try:
        f = open(tmp_path, flag)
        yield f
        f.close()
        shutil.move(tmp_path, path)
    finally:
        f.close()


def pil_grid(images, max_horiz=sys.maxint):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * -(-n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


def discrete_matshow(data, cmap=None, cbar=True):
    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    if cmap is None:
        cmap = plt.get_cmap('RdBu', np.max(data)-np.min(data)+1)
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data)-.5,
                      vmax=np.max(data)+.5, fignum=False)
    if cbar:
        cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data)+1),
                       fraction=0.046, pad=0.04)


def Tictoc():
    start_stack = []
    start_named = {}

    def tic(name=None):
        if name is None:
            start_stack.append(time())
        else:
            start_named[name] = time()

    def toc(name=None):
        if name is None:
            start = start_stack.pop()
        else:
            start = start_named.pop(name)
        elapsed = time() - start
        return elapsed
    return tic, toc
