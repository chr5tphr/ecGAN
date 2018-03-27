from .regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime

estimators = {}

def register_estimator(func):
    estimators[func.__name__] = func
    return func

@register_estimator
def linear():
    return [LinearPatternRegime()]

@register_estimator
def relu():
    return [PositivePatternRegime(), NegativePatternRegime()]

@register_estimator
def positive():
    return [PositivePatternRegime()]

