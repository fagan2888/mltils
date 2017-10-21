

def to_grid(params):
    return {
        key: _to_iterable(value)
        for key, value in params.items()
    }


def _to_iterable(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        return [x]
