import numpy as np
import h5py
from imageio import imwrite
import matplotlib as mpl
import matplotlib.pyplot as plt
from logging import getLogger


cmaps = {}
def register_cmap(func):
    cmaps[func.__name__] = func
    return func

@register_cmap
def gray(x):
    return np.stack([x]*3, axis=-1).clip(0., 1.)

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

def draw_heatmap(data, lo=0., hi=1., center=None, cmap='hot'):
    try:
        dat = data.asnumpy()
    except AttributeError:
        dat = data
    if center is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            lodat = (dat.clip(lo, center) - lo) / (center - lo) - 1.
            lodat[~np.isfinite(lodat)] = 0.
            hidat = (dat.clip(center, hi) - center) / (hi - center)
            hidat[~np.isfinite(hidat)] = 0.
        ndat = ((hidat + lodat) + 1.) / 2.
    else:
        ndat = ((dat - lo)/(hi-lo))

    return cmaps[cmap](ndat.clip(0., 1.))

def align_images(im, H, W, h, w, C=1):
    return im.reshape(H, W, h, w, C).transpose(0, 2, 1, 3, 4).reshape(H*h, W*w, C)

def save_explanation_image(relevance, fpath, center=None, align=False, cmap='hot', batchnorm=False):
    rdat = relevance
    if rdat.shape[1] == 1:
        if batchnorm:
            lo, hi = rdat.min().asscalar(), rdat.max().asscalar()
        else:
            lo = rdat.min(axis=(2, 3), keepdims=True).asnumpy()
            hi = rdat.max(axis=(2, 3), keepdims=True).asnumpy()
        getLogger('ecGAN').debug('Explanation min %f, max %f', lo, hi)
        rdat = (draw_heatmap(rdat, lo, hi, center=center, cmap=cmap)*255).astype(np.uint8)
    elif rdat.shape[1] == 3:
        rdat = rdat.transpose([0,2,3,1])
    else:
        raise RuntimeError("Useless number of channels.")

    if align:
        # rdat = align_images(rdat, *align)
        rdat = align_images(rdat, 5, 6, 28, 28, 3)
    imwrite(fpath, rdat)
    getLogger('ecGAN').info('Saved explanation image in \'%s\'.', fpath)

def save_explanation_data(relevance, fpath):
    with h5py.File(fpath, 'w') as fp:
        fp['heatmap'] = relevance.asnumpy()
    getLogger('ecGAN').info('Saved explanation in \'%s\'.', fpath)

def save_source_image(data, fpath, bbox):
    indat = ((data - bbox[0]) * 255/(bbox[1]-bbox[0])).asnumpy().clip(0, 255).astype(np.uint8)
    indat = align_images(indat, 5, 6, 28, 28)
    imwrite(fpath, indat)
    getLogger('ecGAN').info('Saved input data in \'%s\'.', fpath)

def save_source_raw_image(data, fpath):
    lo, hi = data.min().asscalar(), data.max().asscalar()
    indat = (draw_heatmap(data, lo, hi)*255).astype(np.uint8)
    imwrite(fpath, indat)
    getLogger('ecGAN').info('Saved input data in \'%s\'.', fpath)

def plot_data(data):
    snum = int(len(data)**.5)
    if type(data) is nd.NDArray:
        data = data.asnumpy()
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

