#train
import h5py
from os.path import isfile
from keras.datasets import mnist
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam

from innvestigate.utils.tests.networks.base import mlp_3dense

def rescale(X, bbox=None, scale=[-1.,1.]):
    tlo, thi = scale
    lo, hi = (X.min(), X.max()) if bbox is None else bbox
    return (X - lo) / (hi - lo) * (thi - tlo) + tlo

K       = 10
inshape = (None, 1, 28, 28)

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()
# is already channel_first due to there being only a single channel!
x_train = rescale(x_train_raw.reshape([60000, 1, 28, 28]).astype('float32'), bbox=[0,255])
x_test  = rescale(x_test_raw .reshape([10000, 1, 28, 28]).astype('float32'), bbox=[0,255])
y_train = to_categorical(y_train_raw, K)
y_test  = to_categorical(y_test_raw , K)

net        = mlp_3dense(inshape, K)
model      = Model(inputs=net['in'], outputs=net['out'])

fname   = '/home/chrstphr/tmp/mlp_3dense.h5'
retrain = False

if retrain or not isfile(fname):
    bsize   = 64
    nepochs = 10

    model_smax = Model(inputs=net['in'], outputs=net['sm_out'])
    model_smax.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    hist  = model_smax.fit(x_train, y_train, batch_size=bsize, epochs=nepochs, verbose=1)
    score = model_smax.evaluate(x_test, y_test, verbose=0)
    print('Test loss: %s\nTest accuracy: %s'%(str(score[0]), str(score[1])))

    weights = model_smax.get_weights()
    with h5py.File(fname, 'w') as fp:
        fp.update([('weights/%02d'%i,x) for i,x in enumerate(weights)])
else:
    with h5py.File(fname, 'r') as fp:
        weights = [arr[1][:] for arr in sorted(fp['weights'].items())]
model.set_weights(weights)

#analysis
from innvestigate import create_analyzer
from innvestigate.utils import BatchSequence
from innvestigate.tools.pattern import PatternComputer

analyzer = create_analyzer('pattern.attribution', model)
#analyzer.fit(x_train, pattern_type='relu', batch_size=bsize, verbose=1)
#rel = analyzer.analyze(x_test[:30])
bsize = 256
generator = BatchSequence(x_train, bsize)
computer = PatternComputer(model, pattern_type='relu')
patterns = computer.compute_generator(generator, keep_pattern_instances=True)

#comparison
import mxnet
from mxnet import nd
from ecGAN.net import mlp_3dense as mlp_3dense_mx
from ecGAN.explain.pattern.estimator import estimators

net_mx = mlp_3dense_mx(outnum=K, numhid=512, droprate=0.25, use_bias=True, patest={'relu': 'relu', 'out': 'linear'})
net_mx.collect_params().initialize()
net_mx(nd.zeros([1] + list(inshape)[1:]))

for param, weightval in zip(net_mx.collect_params().values(), weights):
    param.set_data(weightval.T)

bsize = 512
dset_test = mxnet.gluon.data.ArrayDataset(x_test, y_test_raw)
data_iter = mxnet.gluon.data.DataLoader(dset_test, bsize)
metric = mxnet.metric.Accuracy()
for X, Y in data_iter:
    out = net_mx(X)
    metric.update([Y], [out])
print("Test %s: %s"%metric.get())

net_mx.init_pattern()
net_mx.collect_pparams().initialize()
dset_train = mxnet.gluon.data.ArrayDataset(x_train, y_train_raw)
data_iter = mxnet.gluon.data.DataLoader(dset_train, bsize)
for X, Y in data_iter:
    net_mx.forward_logged(X)
    net_mx.learn_pattern()
net_mx.compute_pattern()

kmean, mmean = computer._pattern_instances['relu'][0].mean_x.get_weights()[0], net_mx[1]._regimes[0].mean_x.data().asnumpy()
