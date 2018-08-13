#patterntest
import numpy as np
from mxnet import nd
from ecGAN.layer import Conv2D
from ecGAN.explain.pattern.estimator import estimators

lay = Conv2D(20, 2, strides=2, padding=0, regimes=estimators['linear']())
lay.initialize()

data = nd.random.normal(5,shape=[1000,3,8,8])
out = lay(data)
lay.init_pattern()
lay.collect_pparams().initialize()

for mdat in [data[i::100] for i in range(100)]:
    lay.forward_logged(mdat)
    lay.learn_pattern()
lay.compute_pattern()
resdat = data.reshape([1000,3,4,2,4,2]).transpose([0,2,4,1,3,5]).reshape([1000*4*4,3*4])
resout = out.transpose([0,2,3,1]).reshape([1000*4*4,20])
rescov = nd.dot((resout - resout.mean(0)).T, (resdat - resdat.mean(0))) / resout.shape[0]

#TODO check whether correlation is correct!
var_y = (lay.weight.data().flatten() * rescov).mean(1, keepdims=True)
std_y = (resout - resout.mean(0)).mean(0)
