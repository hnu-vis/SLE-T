




from functools import partial


def multi_apply(func, *args, **kwargs):
















    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def add_prefix(inputs, prefix):











    outputs = dict()
    for name, value in inputs.items():
        outputs[f"{prefix}.{name}"] = value

    return outputs
