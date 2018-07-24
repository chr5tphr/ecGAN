from .regimes import LinearPatternRegime, PositivePatternRegime, NegativePatternRegime, ClipPatternRegime

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

@register_estimator
def clip(ymin=-1., ymax=1.):
    return [ClipPatternRegime(*args) for args in [[None, ymin], [ymin, ymax], [ymax, None]]]

@register_estimator
def cliptop(ymin=-1., ymax=1.):
    return [ClipPatternRegime(*args) for args in [[ymin, ymax], [ymax, None]]]
