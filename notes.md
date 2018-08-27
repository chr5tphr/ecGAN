## Problems
### Problem: Float precision when updating mean:
If the mantissa of the float is 23 bit, there is a precision of up to 1677000 during addition with a single digit number.
Above that threshold, adding a single digit number does not have any effect on the sum anymore.
When updating means, the mean simply does not change anymore, which is not that much of a problem after more than 16,000,000 samples.

### Pattern: Mean of Signal cannot be restored
since noise-mean + signal-mean cannot be decomposed

### Zero-corners in MNIST conv PatternNet
 - Solution: because corners are always zero in dataset

### Attribution gives wrong predictions
- Obsolete:
    - Problem : because we use relu where actually clipping (or something else) is used, output gets lost
    - Solution: use clipping regimes
    - Problem : Still inaccurate
    - Solution: add bias in pattern pass
- New Solution:
    - Have a Null-Pattern that forwards outputs for every output that does not lie in any regime


### Broken PatternNet
- Problem : I only computed one global mean of X
- Solution: you have to compute the X's means wrt. every output, resulting in a mean of size out x in

### Biases are not negative (LRP,DTD)
- Problem : LRP and DTD need negative biases, but we do not enforce those for sake of a more performant model
- Solution: introduce bias in denominator, effectively treating biases as weights and appending a '1' to X. Note that this results in the DTD methods not being conservative anymore!

### LRP/DTD gives empty explanations
- Problem : since we append a ReLU to networks before explaining, when the highest activation is negative, the explanation will be all zero.
- Discuss : does negative highest activations, since we trained with softmax, correspond to low confidence?
- Solution: cherry-pick?!

### PatternAttribution is negative on Generated Data but positive on Test Data
- Problem : possibly, confidence for the input data is low, resulting in negative output, which in effect flips the sign

## Questions

### Why use even filters instead of odd ones?
When deconvoluting odd filters with stride 2, result will always be odd, and output will have different amounts of connections:

|  ksize 5, stride 2  |
|---------------------|
|    1   1   1   1    |
|1 1 2 2 3 2 3 2 2 1 1|
-----------------------

however, this is not the case for even kernel sizes:

| ksize 4, stride 2 |
|-------------------|
|   1   1   1   1   |
|1 1 2 2 2 2 2 2 1 1|
---------------------

the problem of lower activations at the border remains.

#### REMEMBER
when padding with strides, pads are multiplied by stride-width!

### Why use strides instead of MaxPool?
When using MaxPool, gradients are one where the maximum was and zero elsewhere.
Thus, gradients become very sparse, which in case of a discriminator, makes it very hard for the underlying generator to learn.

## TODO
## Experiments

### Datasets
- MNIST
- Toy  (think of a better one?)
- CIFAR10

### Nets
- w/ clipping
- w/o clipping (bad performance)

### Weights
- trained weights
- constant weights
- random weights?

### Ideas
- test explanations with same noise but different conds

## Questions

### Does no clipping result in better explanations?
- (not shown)
- but clipping results in better generator

### Tanh vs. Clipping
- Tanh: no sparse gradients -> inabillity to learn cifar10
- Clip: does not have the 'noisy pixels', since it can easily reach full colors

- Cifar10 only works with Tanh! (sparse gradients)


## pattern tangens: clipped linear
