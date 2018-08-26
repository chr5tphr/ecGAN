import numpy as np
import json
import h5py
from mxnet.base import MXNetError
from mxnet import nd
from imageio import imwrite
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from logging import getLogger

from .func import asnumpy


cmaps = {}
def register_cmap(func):
    cmaps[func.__name__] = func
    return func

@register_cmap
def gray(x):
    return np.stack([x]*3, axis=-1).clip(0., 1.)

@register_cmap
def wred(x):
    return np.stack([0.*x+1., 1.-x, 1.-x], axis=-1).clip(0., 1.)

@register_cmap
def wblue(x):
    return np.stack([1.-x, 1.-x, 0*x+1.], axis=-1).clip(0., 1.)

@register_cmap
def hot(x):
    return np.stack([x*3, x*3-1, x*3-2], axis=-1).clip(0., 1.)

@register_cmap
def cold(x):
    return np.stack([0.*x, x*2-1, x*2], axis=-1).clip(0., 1.)

@register_cmap
def coldnhot(x):
    hpos = hot((2*x-1.).clip(0., 1.))
    hneg = cold(-(2*x-1.).clip(-1., 0.))
    return hpos + hneg

@register_cmap
def bwr(x):
    hpos = wred((2*x-1.).clip(0., 1.))
    hneg = wblue(-(2*x-1.).clip(-1., 0.))
    return hpos + hneg - 1.

def draw_heatmap(data, lo=0., hi=1., center=None, cmap='hot'):
    if isinstance(data, nd.NDArray):
        data = asnumpy(data)
    if center is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            lodat = (data.clip(lo, center) - lo) / (center - lo) - 1.
            lodat[~np.isfinite(lodat)] = 0.
            hidat = (data.clip(center, hi) - center) / (hi - center)
            hidat[~np.isfinite(hidat)] = 0.
        ndat = ((hidat + lodat) + 1.) / 2.
    else:
        ndat = ((data - lo)/(hi-lo))

    return cmaps[cmap](ndat.clip(0., 1.))

def align_images(im, H, W, h, w, C=1):
    # color channels must be last axis!
    return im.reshape([H, W, h, w, C]).transpose([0, 2, 1, 3, 4]).reshape([H*h, W*w, C])

def save_colorized_image(data, fpath, center=None, cmap='hot', batchnorm=False, fullcmap=False, what='explanation', outshape=(5, 6)):
    if isinstance(data, nd.NDArray):
        data = asnumpy(data)
    N, C, H, W = data.shape
    data = data.transpose([0, 2, 3, 1])

    if outshape is None:
        outshape = [int(N**0.5)]*2
    crop = np.prod(outshape)
    oH, oW = outshape
    data = data[:crop].mean(axis=3)
    if batchnorm:
        lo, hi = data.min(), data.max()
    else:
        lo = data.min(axis=1, keepdims=True).min(axis=2, keepdims=True)
        hi = data.max(axis=1, keepdims=True).max(axis=2, keepdims=True)
    if not fullcmap:
        hi = np.maximum(np.abs(lo), np.abs(hi))
        lo = -hi
    #getLogger('ecGAN').debug('%s min %f, max %f', what, lo.min(), hi.max())
    data = draw_heatmap(data, lo, hi, center=center, cmap=cmap)
    #getLogger('ecGAN').debug('data min %s, max %s', str(data.min()), str(data.max()))
    data = (data * 255).clip(0, 255).astype(np.uint8)

    data = align_images(data, oH, oW, H, W, 3)
    imwrite(fpath, data)
    getLogger('ecGAN').info('Saved %s in \'%s\'.', what, fpath)

def save_data_h5(relevance, fpath, what='explanation'):
    with h5py.File(fpath, 'w') as fp:
        fp['heatmap'] = asnumpy(relevance)
    getLogger('ecGAN').info('Saved %s in \'%s\'.', what, fpath)

def save_cgan_visualization(noise, cond, fpath, what='visualization'):
    fig = plt.figure(figsize=(16, 9))
    #combo = np.concatenate([noise, cond], axis=1)
    #amin = combo.min()
    #amax = combo.max()
    num, nlen = noise.shape
    _, clen = cond.shape
    for i, (noi, con) in enumerate(zip(noise, cond)):
        ax = fig.add_subplot(num//3, 3, i+1)
        ax.bar(np.arange(nlen), noi, color='b')
        ax.bar(np.arange(nlen, nlen + clen), con, color='r')
        ax.set_xlim(-1, nlen + clen)
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.ylim(amin, amax)
    fig.tight_layout()
    fig.savefig(fpath)
    plt.close(fig)
    getLogger('ecGAN').info('Saved %s in \'%s\'.', what, fpath)

def save_aligned_image(data, fpath, bbox, what='input data', outshape=(5, 6)):
    if isinstance(data, nd.NDArray):
        data = asnumpy(data)
    N, C, H, W = data.shape
    data = data.transpose([0, 2, 3, 1])
    if outshape is None:
        outshape = [int(N**0.5)]*2
    crop = np.prod(outshape)
    data = data[:crop]
    oH, oW = outshape
    indat = ((data - bbox[0]) * 255/(bbox[1]-bbox[0])).clip(0, 255).astype(np.uint8)
    indat = align_images(indat, oH, oW, H, W, C)
    imwrite(fpath, indat)
    getLogger('ecGAN').info('Saved %s in \'%s\'.', what, fpath)

def save_raw_image(data, fpath, what='input data'):
    lo, hi = data.min().asscalar(), data.max().asscalar()
    indat = (draw_heatmap(data, lo, hi)*255).astype(np.uint8)
    imwrite(fpath, indat)
    getLogger('ecGAN').info('Saved %s in \'%s\'.', what, fpath)

def save_predictions(data, fpath):
    with open(fpath, 'w') as fp:
        sdat = asnumpy(data)
        json.dump(sdat.astype(int).tolist(), fp, indent=2)
        #json.dump([int(da.asscalar()) for da in data], fp, indent=2)
    getLogger('ecGAN').info('Saved predicted labels in \'%s\'.', fpath)


def plot_data(data):
    snum = int(len(data)**.5)
    if type(data) is nd.NDArray:
        data = asnumpy(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data[:snum**2].reshape(snum, snum, 28, 28).transpose(0, 2, 1, 3).reshape(snum*28, snum*28), cmap='Greys')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hlines(np.arange(1, snum)*28, 0, snum*28)
    ax.vlines(np.arange(1, snum)*28, 0, snum*28)
    fig.tight_layout()
    ax.set
    return fig

