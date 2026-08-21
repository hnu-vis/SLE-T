




from mmcv.runner import BaseModule

from ..builder import BACKBONES


@BACKBONES.register_module()
class DinoVisionTransformer(BaseModule):


    def __init__(self, *args, **kwargs):
        super().__init__()
