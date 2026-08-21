




from mmcv.utils import Registry

TRANSFORMER = Registry("Transformer")
MASK_ASSIGNERS = Registry("mask_assigner")
MATCH_COST = Registry("match_cost")


def build_match_cost(cfg):

    return MATCH_COST.build(cfg)


def build_assigner(cfg):

    return MASK_ASSIGNERS.build(cfg)


def build_transformer(cfg):

    return TRANSFORMER.build(cfg)
