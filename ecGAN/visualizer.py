import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import os
import logging
import h5py

from .util import mkfilelogger
from .plot import plot_data, save_data_h5, load_data_h5, save_colorized_image, save_aligned_image, save_raw_image, save_cgan_visualization, save_predictions

getLogger = logging.getLogger

visualizers = {}
def register_visualizer(func):
    visualizers[func.__name__] = func
    return func

def check_write(overwrite, fpath, func, args=[], kwargs={}):
    if not overwrite and os.path.isfile(fpath):
        getLogger('ecGAN').info('File already exits, skipping \'%s\'...', fpath)
        return False
    else:
        func(*args, fpath=fpath, **kwargs)
        return True

def check_read(load, fpath, func, args=[], kwargs={}):
    if load and os.path.isfile(fpath):
        getLogger('ecGAN').info('File exits, loading \'%s\'...', fpath)
        func(*args, fpath=fpath, **kwargs)
        return True
    else:
        return False

class Visualizer(object):
    def __init__(self, config):
        self.config = config
        self.comkw  = None
        self.templ  = None

    def feed(self):
        raise NotImplementedError

    @staticmethod

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

@register_visualizer
class PlainVisualizer(Visualizer):
    def __init__(self, *args, outshape=[5, 6], iterations=None, globlim=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.outshape = outshape
        self.iterations = iterations
        self.globlim = globlim

    def __enter__(self):
        if self.config.model.type == 'Classifier':
            top_net = 'classifier'
        elif self.config.model.type == 'CGAN':
            top_net = 'discriminator'
        else:
            raise NotImplementedError
        net_desc = self.config.nets[self.config.model.kwargs[top_net]]
        self.comkw = {
            'iter'      : 0,
            'net'       : '%s<%s>'%(net_desc.name, net_desc.type),
            'net_epoch' : net_desc.epoch
        }
        if self.config.use_pattern:
            self.templ = self.config.pattern.output
        else:
            self.templ = self.config.explanation.output
        return self

    def feed(self):
        if self.comkw['iter'] >= self.iterations:
            getLogger('ecGAN').info('Maximum visualization iteration reached, skipping.')
            return
        if self.config.model.type == 'CGAN':
            return self._feed_cgan()
        elif self.config.model.type == 'Classifier':
            return self._feed_clss()
        else:
            raise NotImplementedError

    def _save_figure(self, noise, cond, noise_s, cond_s, fpath, outshape=[10,10]):
        num = len(noise)
        hei, wid = self.outshape

        lim = np.maximum(*[np.nanmax(np.abs(arr)) for arr in [noise_s, cond_s]])
        fig = plt.figure(figsize=(9*wid, 3*(num//wid+1)))

        for i, (noi_x, con_x, noi_s, con_s) in enumerate(zip(noise, cond, noise_s, cond_s)):
            nlen = len(noi_x)
            clen = len(con_x)
            ax = fig.add_subplot(num//wid, wid, i+1)
            #ax.set_title('mean with %s: %s'%(cname, key))
            # noise
            ax.bar(0.05 + np.arange(nlen)/10, noi_s, width=0.1, color='b')

            # cond
            color = np.array([1, 0, 0])[None] - con_x[:,None].clip(0,1) * np.array([1, 0, 0])[None]
            ax.bar(0.5 + nlen/10 + np.arange(clen), con_s , width=1.0, color=color)

            if not self.globlim:
                lim = np.maximum(*[np.nanmax(np.abs(arr)) for arr in [noi_s, con_s]])
            ax.set_xlim(0., nlen/10 + clen)
            ax.set_ylim(-lim, lim)
            if (i % wid) != 0:
                ax.set_yticks([])
            if (i // wid) != ((num // wid)-1):
                ax.set_xticks([])
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecgan').info('Saved %s in \'%s\'.', 'plain figure', fpath)

    def _save_figure_noise(self, noise, noise_s, fpath, outshape=[10,10]):
        num = len(noise)
        hei, wid = self.outshape

        lim = np.nanmax(np.abs(noise_s))
        fig = plt.figure(figsize=(16/3*wid, 3*(num//wid+1)))

        for i, (noi_x, noi_s) in enumerate(zip(noise, noise_s)):
            nlen = len(noi_x)
            ax = fig.add_subplot(num//wid, wid, i+1)
            #ax.set_title('mean with %s: %s'%(cname, key))
            # noise
            ax.bar(0.5 + np.arange(nlen), noi_s, width=1.0, color='b')

            if not self.globlim:
                lim = np.nanmax(np.abs(noi_s))
            ax.set_xlim(0., nlen)
            ax.set_ylim(-lim, lim)
            #if (i % wid) != 0:
            #    ax.set_yticks([])
            #if (i // wid) != ((num // wid)-1):
            #    ax.set_xticks([])
            ax.set_xticks(np.arange(0, nlen, 10))
            ax.set_xticklabels(['$%d$'%lab for lab in range(0, nlen, 10)])
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecgan').info('Saved %s in \'%s\'.', 'plain noise figure', fpath)

    def _save_figure_cond(self, cond, cond_s, fpath, outshape=[10,10]):
        num = len(cond)
        hei, wid = self.outshape

        lim = np.nanmax(np.abs(cond_s))
        fig = plt.figure(figsize=(16/3*wid, 3*(num//wid+1)))

        for i, (con_x, con_s) in enumerate(zip(cond, cond_s)):
            clen = len(con_x)
            ax = fig.add_subplot(num//wid, wid, i+1)
            # cond
            color = np.array([1, 0, 0])[None] - con_x[:,None].clip(0,1) * np.array([1, 0, 0])[None]
            ax.bar(0.5 + np.arange(clen), con_s , width=1.0, color=color)

            if not self.globlim:
                lim = np.nanmax(np.abs(con_s))
            ax.set_xlim(0., clen)
            ax.set_ylim(-lim, lim)

            ax.set_xticks(np.arange(clen))
            ax.set_xticklabels(['$%d$'%lab for lab in range(clen)])
            #if (i % wid) != 0:
            #    ax.set_yticks([])
            #if (i // wid) != ((num // wid)-1):
            #    ax.set_xticks([])
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecgan').info('Saved %s in \'%s\'.', 'plain cond figure', fpath)

    def _feed_cgan(self):
        h, w = self.outshape
        n = np.prod(self.outshape)
        info_keys = [
            'input/noise',
            'input/cond',
            'generated',
            'prediction',
            'label',
            'relevance/noise',
            'relevance/cond',
            'relevance/generated',
        ]
        fpath = self.config.exsub(self.templ, data_desc='result<%s>'%self.config.sampler.type, ftype='h5', **self.comkw)
        if os.path.isfile(fpath):
            noise, cond, gen, pred, label, s_noise, s_cond, s_gen = load_data_h5(info_keys, fpath)
        else:
            getLogger('ecGAN').warning('Cannot load file, skipping... \'%s\'', fpath)
            self.comkw['iter'] += 1
            return
        desc = self.config.sampler.type
        files = {
            'pred'      : {
                'fpath'  : self.config.exsub(self.templ, data_desc='prediction<%s>'%desc, ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [pred],
                'kwargs' : {},
            },
            'label'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='truth<%s>'%desc     , ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [label],
                'kwargs' : {},
            },
            'input'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='input<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure,
                'args'   : [noise, cond, noise, cond],
                'kwargs' : {'outshape': self.outshape},
            },
            'relevance' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='relevance<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure,
                'args'   : [noise, cond, s_noise, s_cond],
                'kwargs' : {'outshape': self.outshape},
            },
            'noiserel' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='noiserel<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure_noise,
                'args'   : [noise, s_noise],
                'kwargs' : {'outshape': self.outshape},
            },
            'condrel' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='condrel<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure_cond,
                'args'   : [cond, s_cond],
                'kwargs' : {'outshape': self.outshape},
            },
            'gen'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='gen<%s>'%desc, ftype='png', **self.comkw),
                'func'   : save_aligned_image,
                'args'   : [gen],
                'kwargs' : {'bbox': self.config.data.bbox, 'what': 'generated input data', 'outshape': self.outshape},
            },
            'genrel' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='genrel<%s>'%desc, ftype='png', **self.comkw),
                'func'   : save_colorized_image,
                'args'   : [s_gen],
                'kwargs' : {'center': self.config.get('cmap_center'), 'cmap': self.config.get('cmap', 'bwr'), 'what': 'top explanation', 'outshape': self.outshape},
            },
        }
        for key, val in files.items():
            check_write(self.config.overwrite, val['fpath'], val['func'], val['args'], val['kwargs'])
        self.comkw['iter'] += 1

    def _feed_clss(self):
        info_keys = [
            'input',
            'prediction',
            'label',
            'relevance',
        ]
        fpath = self.config.exsub(self.templ, data_desc='result<%s>'%self.config.data.func, ftype='h5', **self.comkw)
        if os.path.isfile(fpath):
            data, pred, label, relevance = load_data_h5(info_keys, fpath)
        else:
            getLogger('ecGAN').warning('Cannot load file, skipping... \'%s\'', fpath)
            self.comkw['iter'] += 1
            return

        desc = self.config.data.func
        files = {
            'pred'      : {
                'fpath'  : self.config.exsub(self.templ, data_desc='prediction<%s>'%desc, ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [pred],
                'kwargs' : {},
            },
            'label'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='truth<%s>'%desc     , ftype='json', **self.comkw),
                'func'   : save_predictions,
                'args'   : [label],
                'kwargs' : {},
            },
            'input'     : {
                'fpath'  : self.config.exsub(self.templ, data_desc='input<%s>'%desc, ftype='png', **self.comkw),
                'func'   : save_aligned_image,
                'args'   : [data],
                'kwargs' : {'bbox': self.config.data.bbox, 'outshape': self.outshape},
            },
            'relevance' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='relevance<%s>'%desc, ftype='png', **self.comkw),
                'func'   : save_colorized_image,
                'args'   : [relevance],
                'kwargs' : {'center': self.config.get('cmap_center'), 'cmap': self.config.get('cmap', 'bwr'), 'outshape': self.outshape},
            },
        }
        for key, val in files.items():
            check_write(self.config.overwrite, val['fpath'], val['func'], val['args'], val['kwargs'])

        self.comkw['iter'] += 1

@register_visualizer
class SimilarityVisualizer(Visualizer):
    def __enter__(self):
        if self.config.model.type == 'Classifier':
            top_net = 'classifier'
        elif self.config.model.type == 'CGAN':
            top_net = 'discriminator'
        else:
            raise NotImplementedError
        net_desc = self.config.nets[self.config.model.kwargs[top_net]]
        self.comkw = {
            'iter'      : 0,
            'net'       : '%s<%s>'%(net_desc.name, net_desc.type),
            'net_epoch' : net_desc.epoch
        }
        if self.config.use_pattern:
            self.templ = self.config.pattern.output
        else:
            self.templ = self.config.explanation.output
        return self

    def feed(self):
        if self.config.model.type == 'CGAN':
            return self._feed_cgan()
        elif self.config.model.type == 'Classifier':
            return self._feed_clss()
        else:
            raise NotImplementedError

    def _feed_cgan(self):
        info_keys = [
            'input/cond',
            'relevance/noise',
            'relevance/cond',
        ]
        fpath = self.config.exsub(self.templ, data_desc='result<%s>'%self.config.sampler.type, ftype='h5', **self.comkw)
        if os.path.isfile(fpath):
            in_cond, s_noise, s_cond = load_data_h5(info_keys, fpath)
        else:
            getLogger('ecGAN').warning('Cannot load file, skipping... \'%s\'', fpath)
            self.comkw['iter'] += 1
            return

        csqnorm = (s_cond**2).sum(1, keepdims=True)
        nsqnorm = (s_noise**2).sum(1, keepdims=True)
        norm    = (csqnorm + nsqnorm)**.5
        s_noise = s_noise/norm
        s_cond  = s_cond/norm

        ncrate = (csqnorm**.5 / norm)[:,0]

        argmax = []
        for x,y in zip(in_cond, s_cond):
            act = x > 0.
            nact = act.sum()
            if nact > 0:
                argmax += [(act * np.eye(len(x), dtype=bool)[np.argsort(y)[::-1][:nact]].sum(axis=0)).sum() / nact]
        argmax = np.array(argmax)

        euclid = ((in_cond - s_cond)**2).sum(axis=1)
        cosine = (in_cond * s_cond).sum(1) / ((in_cond**2).sum(1) * (s_cond**2).sum(1))**.5

        desc = self.config.sampler.type
        files = {
            'data'   : {
                'fpath'  : self.config.exsub(self.templ, data_desc='sim<%s>'%desc, ftype='h5', **self.comkw),
                'func'   : self._save_data,
                'args'   : [ncrate, argmax, euclid, cosine],
                'kwargs' : {},
            },
            'figure' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='sim<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure,
                'args'   : [ncrate, argmax, euclid, cosine],
                'kwargs' : {},
            },
        }
        for key, val in files.items():
            check_write(self.config.overwrite, val['fpath'], val['func'], val['args'], val['kwargs'])
        self.comkw['iter'] += 1

    def _feed_clss(self):
        pass

    def _save_data(self, ncrate, argmax, euclid, cosine, fpath):
        info = {
            'ncrate' : ncrate,
            'argmax' : argmax,
            'euclid' : euclid,
            'cosine' : cosine,
        }
        save_data_h5(info, fpath)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'similarities result', fpath)

    def _save_figure(self, ncrate, argmax, euclid, cosine, fpath):
        num = len(ncrate)

        fig = plt.figure(figsize=(num/10, 8))

        for i, (val, nam, col) in enumerate(zip([ncrate, argmax, euclid, cosine], ['ncrate', 'argmax', 'euclid', 'cosine'], ['r', 'g', 'b', 'm'])):
            ax = fig.add_subplot(4, 1, i+1)
            vlen = len(val)

            ax.set_title('Similarity to input condition using %s'%nam)
            ax.bar(0.5 + np.arange(vlen), val, width=1.0, color=col)

            ax.set_xlim(0., vlen)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond figure', fpath)


@register_visualizer
class MeanVisualizer(Visualizer):
    def __init__(self, *args, cond='label', load=True, center=True, globlim=False, normalize=True, outshape=[5,2], **kwargs):
        super().__init__(*args, **kwargs)
        self.cond = cond
        self.load = load
        self._loaded = False
        self.center = center
        self.outshape = outshape
        self.globlim = globlim
        self.normalize = normalize

    def __enter__(self):
        if self.config.model.type == 'Classifier':
            top_net = 'classifier'
            desc = self.config.data.func
        elif self.config.model.type == 'CGAN':
            desc = self.config.sampler.type
            top_net = 'discriminator'
        else:
            raise NotImplementedError
        net_desc = self.config.nets[self.config.model.kwargs[top_net]]
        self.comkw = {
            'iter'      : 0,
            'net'       : '%s<%s>'%(net_desc.name, net_desc.type),
            'net_epoch' : net_desc.epoch
        }
        if self.config.use_pattern:
            self.templ = self.config.pattern.output
        else:
            self.templ = self.config.explanation.output
        self._acc = {}

        if self.load:
            ckw = self.comkw.copy()
            ckw['iter'] = self.config.explanation.iterations
            fpath = self.config.exsub(self.templ, data_desc='mean<%s>'%desc, ftype='h5', **ckw)
            if os.path.isfile(fpath):
                self._load_data(fpath)
                self.comkw = ckw

        return self

    def feed(self):
        if self._loaded:
            getLogger('ecGAN').info('Loaded data, skipping feed.')
            return
        if self.config.model.type == 'CGAN':
            return self._feed_cgan()
        elif self.config.model.type == 'Classifier':
            return self._feed_clss()
        else:
            raise NotImplementedError

    def _feed_cgan(self):
        info_keys = [
            'prediction',
            'label',
            'input/cond',
            'relevance/noise',
            'relevance/cond',
        ]
        fpath = self.config.exsub(self.templ, data_desc='result<%s>'%self.config.sampler.type, ftype='h5', **self.comkw)
        if os.path.isfile(fpath):
            vals = load_data_h5(info_keys, fpath)
        else:
            getLogger('ecGAN').warning('Cannot load file, skipping... \'%s\'', fpath)
            self.comkw['iter'] += 1
            return
        for ind, (pred, label, in_cond, s_noise, s_cond) in enumerate(zip(*vals)):
            noise_all = self._acc.setdefault('noise_a', np.zeros_like(s_noise))
            cond_all  = self._acc.setdefault('cond_a', np.zeros_like(s_cond))
            count_all = self._acc.setdefault('count_a', np.zeros([1]))

            if self.normalize:
                norm = ((s_noise**2).sum() + (s_cond**2).sum())**.5
                s_noise = s_noise/norm
                s_cond = s_cond/norm

            noise_all += s_noise
            cond_all  += s_cond
            count_all += 1

            alpha = {
                'label': label,
                'prediction': pred,
                'index': ind,
            }[self.cond]
            noise_one = self._acc.setdefault('noise', {}).setdefault('%02d'%alpha, np.zeros_like(s_noise))
            cond_one  = self._acc.setdefault('cond',  {}).setdefault('%02d'%alpha, np.zeros_like(s_cond))
            input_one = self._acc.setdefault('input', {}).setdefault('%02d'%alpha, np.zeros_like(in_cond))
            count_one = self._acc.setdefault('count', {}).setdefault('%02d'%alpha, np.zeros([1], dtype=int))

            noise_one += s_noise
            cond_one  += s_cond
            input_one += in_cond
            count_one += 1
        self.comkw['iter'] += 1


    def _feed_clss(self):
        pass

    def _save_data(self, fpath):
        if self._loaded:
            getLogger('ecGAN').info('File has been loaded for this visualization, not saving %s in \'%s\'.', 'mean cond result', fpath)
            return

        info = {
            'noise/all': self._acc['noise_a'],
            'cond/all' : self._acc['cond_a' ],
            'count/all': self._acc['count_a'],
        }
        for (key, noise), (_, cond), (_, count), (_, incond) in zip(*[self._acc[mk].items() for mk in ['noise', 'cond', 'count', 'input']]):
            info['noise/%s'%key] = noise
            info['cond/%s'%key]  = cond
            info['input/%s'%key] = incond
            info['count/%s'%key] = count
        save_data_h5(info, fpath)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond result', fpath)

    def _load_data(self, fpath):
        with h5py.File(fpath, 'r') as fp:
            for base, top in fp.items():
                for num, val in top.items():
                    if num == 'all':
                        self._acc[base + '_a'] = val[:]
                    else:
                        self._acc.setdefault(base, {})[num] = val[:]
        self._loaded = True
        getLogger('ecGAN').info('Loaded %s from \'%s\'.', 'mean cond result', fpath)


    def _save_figure(self, fpath):
        num = len(self._acc['noise'])
        hei, wid = self.outshape


        anoi = self._acc['noise_a']
        acon = self._acc['cond_a']
        acnt = self._acc['count_a']
        manoi = anoi/acnt
        macon = acon/acnt

        nlen = len(anoi)
        clen = len(acon)

        fig = plt.figure(figsize=(9*wid, 3*(num//wid+1)))
        ax = fig.add_subplot(num//wid+1, 1, 1)
        ax.set_title('Global Mean')
        ax.bar(0.05 + np.arange(nlen)/10      , manoi, width=0.1, color='b')
        ax.bar(0.5 + nlen/10 + np.arange(clen), macon, width=1.0, color='r')
        ax.set_xlim(0., nlen/10 + clen)
        lim = np.maximum(*[np.abs(arr).max() for arr in [manoi, macon]])
        lims = [lim]

        for i, ((key, noise), (_, cond), (_, count), (_, incond)) in enumerate(zip(*[sorted(self._acc[mk].items()) for mk in ['noise', 'cond', 'count', 'input']])):
            mnoise = noise/count
            mcond  = cond/count
            minput = incond/count
            if self.center:
                mnoise -= anoi/acnt
                mcond  -= acon/acnt
            nlen = len(noise)
            clen = len(cond)
            ax = fig.add_subplot(num//wid+1, wid, i+1+wid)
            cname = {
                'label'     : 'label',
                'prediction': 'predicted label',
                'index'     : 'index',
            }[self.cond]
            ax.set_title('Mean with %s: %s'%(cname, key))
            # noise
            ax.bar(0.05 + np.arange(nlen)/10, mnoise, width=0.1, color='b')

            # cond
            color = np.array([1, 0, 0])[None] - minput[:,None].clip(0,1) * np.array([1, 0, 0])[None]
            ax.bar(0.5 + nlen/10 + np.arange(clen), mcond , width=1.0, color=color)

            ax.set_xlim(0., nlen/10 + clen)
            lim = np.maximum(*[np.abs(arr).max() for arr in [mcond, mnoise]])
            ax.set_ylim(-lim, lim)
            lims += lim
            #if (i % wid) != 0:
            #    ax.set_yticks([])
            #if (i // wid) != ((num // wid)-1):
            #    ax.set_xticks([])
        if self.globlim:
            mlim = np.max(lims)
            for ax in fig.axes:
                ax.set_ylim(-mlim, mlim)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond figure', fpath)

    def _save_figure_cond(self, fpath):
        num = len(self._acc['cond'])
        hei, wid = self.outshape

        acon = self._acc['cond_a']
        acnt = self._acc['count_a']
        macon = acon/acnt

        clen = len(acon)

        fig = plt.figure(figsize=(9*wid, 3*(num//wid+1)))
        ax = fig.add_subplot(num//wid+1, 1, 1)
        ax.set_title('Global Mean')
        ax.bar(0.5 + np.arange(clen), macon, width=1.0, color='r')
        ax.set_xlim(0., clen)
        lim = np.nanmax(np.abs(macon))
        ax.set_ylim(-lim, lim)
        lims = [lim]

        for i, ((key, cond), (_, count), (_, incond)) in enumerate(zip(*[sorted(self._acc[mk].items()) for mk in ['cond', 'count', 'input']])):
            mcond  = cond/count
            minput = incond/count
            if self.center:
                mcond  -= macon
            clen = len(cond)
            ax = fig.add_subplot(num//wid+1, wid, i+1+wid)
            cname = {
                'label'     : 'label',
                'prediction': 'predicted label',
                'index'     : 'index',
            }[self.cond]
            ax.set_title('Mean with %s: %s'%(cname, key))

            # cond
            color = np.array([1, 0, 0])[None] - minput[:,None].clip(0,1) * np.array([1, 0, 0])[None]
            ax.bar(0.5 +np.arange(clen), mcond , width=1.0, color=color)

            ax.set_xticks(np.arange(clen))
            ax.set_xticklabels(['$%d$'%lab for lab in range(clen)])

            ax.set_xlim(0., clen)
            lim = np.nanmax(np.abs(mcond))
            ax.set_ylim(-lim, lim)
            lims += lim
        if self.globlim:
            mlim = np.max(lims)
            for ax in fig.axes:
                ax.set_ylim(-mlim, mlim)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond figure', fpath)

    def _save_figure_noise(self, fpath):
        num = len(self._acc['noise'])
        hei, wid = self.outshape


        anoi = self._acc['noise_a']
        acnt = self._acc['count_a']
        manoi = anoi/acnt

        nlen = len(anoi)

        fig = plt.figure(figsize=(9*wid, 3*(num//wid+1)))
        ax = fig.add_subplot(num//wid+1, 1, 1)
        ax.set_title('Global Mean')
        ax.bar(0.5 + np.arange(nlen), manoi, width=1.0, color='b')
        ax.set_xlim(0., nlen)
        lim = np.nanmax(np.abs(manoi))
        lims = [lim]

        for i, ((key, noise), (_, count)) in enumerate(zip(*[sorted(self._acc[mk].items()) for mk in ['noise', 'count']])):
            mnoise = noise/count
            if self.center:
                mnoise -= anoi/acnt
            nlen = len(noise)
            ax = fig.add_subplot(num//wid+1, wid, i+1+wid)
            cname = {
                'label'     : 'label',
                'prediction': 'predicted label',
                'index'     : 'index',
            }[self.cond]
            ax.set_title('Mean with %s: %s'%(cname, key))
            # noise
            ax.bar(0.5 + np.arange(nlen), mnoise, width=1.0, color='b')

            ax.set_xticks(np.arange(0, nlen, 10))
            ax.set_xticklabels(['$%d$'%lab for lab in range(0, nlen, 10)])

            ax.set_xlim(0., nlen)
            lim = np.nanmax(np.abs(mnoise))
            ax.set_ylim(-lim, lim)
            lims += lim
        if self.globlim:
            mlim = np.max(lims)
            for ax in fig.axes:
                ax.set_ylim(-mlim, mlim)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond figure', fpath)

    def _save_all(self):
        desc = self.config.sampler.type
        files = {
            'data'   : {
                'fpath'  : self.config.exsub(self.templ, data_desc='mean<%s>'%desc, ftype='h5', **self.comkw),
                'func'   : self._save_data,
                'args'   : [],
                'kwargs' : {},
            },
            'noise' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='noisemean<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure_noise,
                'args'   : [],
                'kwargs' : {},
            },
            'cond' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='condmean<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure_cond,
                'args'   : [],
                'kwargs' : {},
            },
            'figure' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='mean<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure,
                'args'   : [],
                'kwargs' : {},
            },
        }
        for key, val in files.items():
            check_write(self.config.overwrite, val['fpath'], val['func'], val['args'], val['kwargs'])

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        if self.config.model.type == 'CGAN':
            self._save_all()
        else:
            getLogger('ecGAN').info('MeanVisualizer is not defined for other model than CGAN.')

@register_visualizer
class QuantitativeVisualizer(Visualizer):
    def __init__(self, *args, cond='label', load=True, center=True, globlim=False, outshape=[5,2], **kwargs):
        super().__init__(*args, **kwargs)
        self.cond = cond
        self.load = load
        self._loaded = False
        self.outshape = outshape

    def __enter__(self):
        if self.config.model.type == 'Classifier':
            top_net = 'classifier'
            desc = self.config.data.func
        elif self.config.model.type == 'CGAN':
            desc = self.config.sampler.type
            top_net = 'discriminator'
        else:
            raise NotImplementedError
        net_desc = self.config.nets[self.config.model.kwargs[top_net]]
        self.comkw = {
            'iter'      : 0,
            'net'       : '%s<%s>'%(net_desc.name, net_desc.type),
            'net_epoch' : net_desc.epoch
        }
        if self.config.use_pattern:
            self.templ = self.config.pattern.output
        else:
            self.templ = self.config.explanation.output
        self._acc = {}

        if self.load:
            ckw = self.comkw.copy()
            ckw['iter'] = self.config.explanation.iterations
            fpath = self.config.exsub(self.templ, data_desc='quant<%s>'%desc, ftype='h5', **ckw)
            if os.path.isfile(fpath):
                self._load_data(fpath)
                self.comkw = ckw

        return self

    def feed(self):
        if self._loaded:
            getLogger('ecGAN').info('Loaded data, skipping feed.')
            return
        if self.config.model.type == 'CGAN':
            return self._feed_cgan()
        elif self.config.model.type == 'Classifier':
            return self._feed_clss()
        else:
            raise NotImplementedError

    def _feed_cgan(self):
        info_keys = [
            'prediction',
            'label',
            'input/cond',
            'relevance/noise',
            'relevance/cond',
        ]
        fpath = self.config.exsub(self.templ, data_desc='result<%s>'%self.config.sampler.type, ftype='h5', **self.comkw)
        if os.path.isfile(fpath):
            vals = load_data_h5(info_keys, fpath)
        else:
            getLogger('ecGAN').warning('Cannot load file, skipping... \'%s\'', fpath)
            self.comkw['iter'] += 1
            return
        for ind, (pred, label, in_cond, s_noise, s_cond) in enumerate(zip(*vals)):

            alpha = ';'.join([val%'%.2f' for val in in_cond])
            self._acc.setdefault('incond', {}).setdefault('%s'%alpha, in_cond)

            ncrate = self._acc.setdefault('ncrate', {}).setdefault('%s'%alpha, np.zeros([1], dtype=float))
            argmax = self._acc.setdefault('argmax', {}).setdefault('%s'%alpha, np.zeros([1], dtype=float))
            euclid = self._acc.setdefault('euclid', {}).setdefault('%s'%alpha, np.zeros([1], dtype=float))
            cosine = self._acc.setdefault('cosine', {}).setdefault('%s'%alpha, np.zeros([1], dtype=float))
            count  = self._acc.setdefault('count' , {}).setdefault('%s'%alpha, np.zeros([1], dtype=int))

            csqnorm = (s_cond**2).sum()
            nsqnorm = (s_noise**2).sum()
            norm    = (csqnorm + nsqnorm)**.5
            s_noise = s_noise/norm
            s_cond  = s_cond/norm

            ncrate += csqnorm**.5 / norm
            active_cond = in_cond > 0.
            if active_cond.any():
                nact = active_cond.sum()
                argmax += (active_cond == np.eye(len(in_cond), dtype=bool)[np.argsort[:nact]].sum(axis=1)).sum() / nact
            euclid += ((in_cond - s_cond)**2).sum()
            cosine += np.inner(in_cond, s_cond) / ((in_cond**2).sum() * (s_cond**2).sum())**.5
            count  += 1
        self.comkw['iter'] += 1


    def _feed_clss(self):
        pass

    def _save_data(self, fpath):
        if self._loaded:
            getLogger('ecGAN').info('File has been loaded for this visualization, not saving %s in \'%s\'.', 'mean cond result', fpath)
            return

        info = {
            'noise/all': self._acc['noise_a'],
            'cond/all' : self._acc['cond_a' ],
            'count/all': self._acc['count_a'],
        }
        for (key, noise), (_, cond), (_, count), (_, incond) in zip(*[self._acc[mk].items() for mk in ['noise', 'cond', 'count', 'input']]):
            info['noise/%s'%key] = noise
            info['cond/%s'%key]  = cond
            info['input/%s'%key] = incond
            info['count/%s'%key] = count
        save_data_h5(info, fpath)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond result', fpath)

    def _load_data(self, fpath):
        with h5py.File(fpath, 'r') as fp:
            for base, top in fp.items():
                for num, val in top.items():
                    if num == 'all':
                        self._acc[base + '_a'] = val[:]
                    else:
                        self._acc.setdefault(base, {})[num] = val[:]
        self._loaded = True
        getLogger('ecGAN').info('Loaded %s from \'%s\'.', 'mean cond result', fpath)


    def _save_figure(self, fpath):
        num = len(self._acc['noise'])
        hei, wid = self.outshape


        anoi = self._acc['noise_a']
        acon = self._acc['cond_a']
        acnt = self._acc['count_a']
        manoi = anoi/acnt
        macon = acon/acnt

        nlen = len(anoi)
        clen = len(acon)

        fig = plt.figure(figsize=(9*wid, 3*(num//wid+1)))
        ax = fig.add_subplot(num//wid+1, 1, 1)
        ax.set_title('Global Mean')
        ax.bar(0.05 + np.arange(nlen)/10      , manoi, width=0.1, color='b')
        ax.bar(0.5 + nlen/10 + np.arange(clen), macon, width=1.0, color='r')
        ax.set_xlim(0., nlen/10 + clen)
        lim = np.maximum(*[np.abs(arr).max() for arr in [manoi, macon]])
        lims = [lim]

        for i, ((key, noise), (_, cond), (_, count), (_, incond)) in enumerate(zip(*[sorted(self._acc[mk].items()) for mk in ['noise', 'cond', 'count', 'input']])):
            mnoise = noise/count
            mcond  = cond/count
            minput = incond/count
            if self.center:
                mnoise -= anoi/acnt
                mcond  -= acon/acnt
            nlen = len(noise)
            clen = len(cond)
            ax = fig.add_subplot(num//wid+1, wid, i+1+wid)
            cname = {
                'label'     : 'label',
                'prediction': 'predicted label',
                'index'     : 'index',
            }[self.cond]
            ax.set_title('Mean with %s: %s'%(cname, key))
            # noise
            ax.bar(0.05 + np.arange(nlen)/10, mnoise, width=0.1, color='b')

            # cond
            color = np.array([1, 0, 0])[None] - minput[:,None].clip(0,1) * np.array([1, 0, 0])[None]
            ax.bar(0.5 + nlen/10 + np.arange(clen), mcond , width=1.0, color=color)

            ax.set_xlim(0., nlen/10 + clen)
            lim = np.maximum(*[np.abs(arr).max() for arr in [mcond, mnoise]])
            ax.set_ylim(-lim, lim)
            lims += lim
            #if (i % wid) != 0:
            #    ax.set_yticks([])
            #if (i // wid) != ((num // wid)-1):
            #    ax.set_xticks([])
        if self.globlim:
            mlim = np.max(lims)
            for ax in fig.axes:
                ax.set_ylim(-mlim, mlim)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond figure', fpath)

    def _save_figure_cond(self, fpath):
        num = len(self._acc['cond'])
        hei, wid = self.outshape

        acon = self._acc['cond_a']
        acnt = self._acc['count_a']
        macon = acon/acnt

        clen = len(acon)

        fig = plt.figure(figsize=(9*wid, 3*(num//wid+1)))
        ax = fig.add_subplot(num//wid+1, 1, 1)
        ax.set_title('Global Mean')
        ax.bar(0.5 + np.arange(clen), macon, width=1.0, color='r')
        ax.set_xlim(0., nlen/10 + clen)
        lim = np.maximum(*[np.abs(arr).max() for arr in [manoi, macon]])
        ax.set_ylim(-lim, lim)
        lims = [lim]

        for i, ((key, cond), (_, count), (_, incond)) in enumerate(zip(*[sorted(self._acc[mk].items()) for mk in ['cond', 'count', 'input']])):
            mcond  = cond/count
            minput = incond/count
            if self.center:
                mcond  -= macon
            clen = len(cond)
            ax = fig.add_subplot(num//wid+1, wid, i+1+wid)
            cname = {
                'label'     : 'label',
                'prediction': 'predicted label',
                'index'     : 'index',
            }[self.cond]
            ax.set_title('Mean with %s: %s'%(cname, key))

            # cond
            color = np.array([1, 0, 0])[None] - minput[:,None].clip(0,1) * np.array([1, 0, 0])[None]
            ax.bar(0.5 +np.arange(clen), mcond , width=1.0, color=color)

            ax.set_xlim(0., nlen/10 + clen)
            lim = np.nanmax(np.abs(mcond))
            ax.set_ylim(-lim, lim)
            lims += lim
        if self.globlim:
            mlim = np.max(lims)
            for ax in fig.axes:
                ax.set_ylim(-mlim, mlim)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond figure', fpath)

    def _save_figure_noise(self, fpath):
        num = len(self._acc['noise'])
        hei, wid = self.outshape


        anoi = self._acc['noise_a']
        acnt = self._acc['count_a']
        manoi = anoi/acnt

        nlen = len(anoi)

        fig = plt.figure(figsize=(9*wid, 3*(num//wid+1)))
        ax = fig.add_subplot(num//wid+1, 1, 1)
        ax.set_title('Global Mean')
        ax.bar(0.5 + np.arange(nlen), manoi, width=1.0, color='b')
        ax.set_xlim(0., nlen)
        lim = np.nanmax(np.abs(manoi))
        lims = [lim]

        for i, ((key, noise), (_, count)) in enumerate(zip(*[sorted(self._acc[mk].items()) for mk in ['noise', 'count']])):
            mnoise = noise/count
            if self.center:
                mnoise -= anoi/acnt
            nlen = len(noise)
            ax = fig.add_subplot(num//wid+1, wid, i+1+wid)
            cname = {
                'label'     : 'label',
                'prediction': 'predicted label',
                'index'     : 'index',
            }[self.cond]
            ax.set_title('Mean with %s: %s'%(cname, key))
            # noise
            ax.bar(0.5 + np.arange(nlen), mnoise, width=1.0, color='b')

            ax.set_xlim(0., nlen)
            lim = np.nanmax(np.abs(mnoise))
            ax.set_ylim(-lim, lim)
            lims += lim
        if self.globlim:
            mlim = np.max(lims)
            for ax in fig.axes:
                ax.set_ylim(-mlim, mlim)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)
        getLogger('ecGAN').info('Saved %s in \'%s\'.', 'mean cond figure', fpath)

    def _save_all(self):
        desc = self.config.sampler.type
        files = {
            'data'   : {
                'fpath'  : self.config.exsub(self.templ, data_desc='mean<%s>'%desc, ftype='h5', **self.comkw),
                'func'   : self._save_data,
                'args'   : [],
                'kwargs' : {},
            },
            'noise' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='noisemean<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure_noise,
                'args'   : [],
                'kwargs' : {},
            },
            'cond' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='condmean<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure_cond,
                'args'   : [],
                'kwargs' : {},
            },
            'figure' : {
                'fpath'  : self.config.exsub(self.templ, data_desc='mean<%s>'%desc, ftype='svg', **self.comkw),
                'func'   : self._save_figure,
                'args'   : [],
                'kwargs' : {},
            },
        }
        for key, val in files.items():
            check_write(self.config.overwrite, val['fpath'], val['func'], val['args'], val['kwargs'])

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        if self.config.model.type == 'CGAN':
            self._save_all()
        else:
            getLogger('ecGAN').info('MeanVisualizer is not defined for other model than CGAN.')
