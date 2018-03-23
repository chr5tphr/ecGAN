from .regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime

def register_estimator(func):
    estimator[func.__name__] = func
    return func

@register_estimator
def linear():
    return [LinearPatternRegime()]

@register_estimator
def relu():
    return [PositivePatternRegime(), NegativePatternRegime()]
