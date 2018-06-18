### Problem: Float precision when updating mean:
If the mantissa of the float is 23 bit, there is a precision of up to 1677000 during addition with a single digit number.
Above that threshold, adding a single digit number does not have any effect on the sum anymore.
When updating means, the mean simply does not change anymore, which is not that much of a problem after more than 16,000,000 samples.

### Pattern: Mean of Signal cannot be restored
since noise-mean + signal-mean cannot be decomposed

### Problem: Zero-corners in MNIST conv PatternNet
because corners are always zero in dataset

