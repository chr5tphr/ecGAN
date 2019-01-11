from mxnet import nd
import numpy as np
import imageio

from ecGAN.util import Config, make_ctx, load_module_file
from ecGAN.func import linspace
from ecGAN.model import models
from ecGAN.plot import save_aligned_image, align_images

config = Config()
config.update_from_file('./result/cgan/CGAN.yaml')
#config.update({'device_id': 5, 'data': {'func': 'cifar10'}, 'nets':{'generator': {'epoch': 400, 'kwargs': {'outnum': 3}}, 'discriminator': {'epoch': 400}}})
net_module = load_module_file(config.sub('net_file'), 'net_module')
ctx = make_ctx(config.device, config.device_id)
model = models[config.model](ctx=ctx, config=config)

# noise,_ = model.sample_noise(K=10, num=30)
#
# cond = nd.ones((30,10,1,1), ctx=ctx)/10.
# gen = model.generate(noise=noise, cond=cond)
# save_aligned_image(data=gen, fpath='gen/gen%04d.png'%100,bbox=config.data.bbox)
#
# cond = nd.concat(nd.ones((30,2,1,1),ctx=ctx)/2.,nd.zeros((30,8,1,1),ctx=ctx),dim=1)
# gen = model.generate(noise=noise, cond=cond)
# save_aligned_image(data=gen, fpath='gen/gen%04d.png'%101,bbox=config.data.bbox)
#
# # see how scaling condtions changes output
# noise = np.tile(np.random.normal(size=(1,100,1,1)), (30,1,1,1))
# noise = nd.array(noise, ctx=ctx)
# bcond = nd.one_hot(linspace(0, 10, 30, ctx=ctx, dtype='int32'), 10).reshape(30, 10, 1, 1)
# for i in range(10):
#     cond = nd.one_hot(nd.array([i]*30, ctx=ctx), 10).reshape(30,10,1,1) * linspace(0.,1.,30,ctx=ctx).reshape(30,1,1,1)
#     gen = model.generate(noise=noise, cond=cond)
#     save_aligned_image(data=gen, fpath='gen/gen%04d.png'%i,bbox=config.data.bbox)

# random walk through latent space
def render(gfunc, stepsize=0.1, momentum=0.9, maxstep=24000):
    K = 10
    num = 30
    bbox = config.data.bbox
    cond = nd.one_hot(nd.repeat(nd.arange(K, ctx=ctx), (num-1)//K+1)[:num], K).reshape((num, K, 1, 1))
    anoi = nd.random.normal(shape=(num,100,1,1), ctx=ctx)
    bnoi = nd.random.normal(shape=(num,100,1,1), ctx=ctx)
    slast = 0.
    for step in range(maxstep):
        snoi = anoi - bnoi

        sdist = snoi.norm(axis=1,keepdims=True)
        if sdist.min().asscalar() < .5:
            anoi = nd.random.normal(shape=(30,100,1,1), ctx=ctx)
        snoi /= sdist
        slast = stepsize*snoi + momentum*slast
        bnoi += slast

        gen = gfunc(noise=bnoi, cond=cond)
        indat = ((gen - bbox[0]) * 255/(bbox[1]-bbox[0])).asnumpy().clip(0, 255).astype(np.uint8)
        indat = align_images(indat, 5, 6, 32, 32, 3)
        yield indat


with imageio.get_writer('gen/latentwalk.%s.mp4'%config.data.func,
                        macro_block_size=32, quality=5., fps=24,
                        ffmpeg_params=['-sws_flags','neighbor','-s', '%dx%d'%(4*6*32,4*5*32)]) as writer:
    for dat in render(model.generate):
        writer.append_data(dat)
