## Problems
### Problem: Float precision when updating mean:
If the mantissa of the float is 23 bit, there is a precision of up to 1677000 during addition with a single digit number.
Above that threshold, adding a single digit number does not have any effect on the sum anymore.
When updating means, the mean simply does not change anymore, which is not that much of a problem after more than 16,000,000 samples.

### Pattern: Mean of Signal cannot be restored
since noise-mean + signal-mean cannot be decomposed

### Problem: Zero-corners in MNIST conv PatternNet
because corners are always zero in dataset

### Problem: Attribution gives wrong predictions
because we use relu where actually clipping (or something else) is used, output gets lost
Solution: use clipping regimes
Problem: Still inaccurate
Solution: add bias in pattern pass

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

## Questions

### Does no clipping result in better explanations?
- (not shown)
- but clipping results in better generator

### Tanh vs. Clipping
- Tanh: no sparse gradients -> inabillity to learn cifar10
- Clip: does not have the 'noisy pixels', since it can easily reach full colors


## pattern tangens: clipped linear
