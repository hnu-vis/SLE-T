




from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16


class BaseDepther(BaseModule, metaclass=ABCMeta):


    def __init__(self, init_cfg=None):
        super(BaseDepther, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):

        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_auxiliary_head(self):

        return hasattr(self, "auxiliary_head") and self.auxiliary_head is not None

    @property
    def with_decode_head(self):

        return hasattr(self, "decode_head") and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):

        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):


        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):

        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):

        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):

        pass

    def forward_test(self, imgs, img_metas, **kwargs):









        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got " f"{type(var)}")
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f"num of augmentations ({len(imgs)}) != " f"num of image meta ({len(img_metas)})")
        
        
        for img_meta in img_metas:
            ori_shapes = [_["ori_shape"] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_["img_shape"] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_["pad_shape"] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, img, img_metas, return_loss=True, **kwargs):









        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):

























        losses = self(**data_batch)

        
        real_losses = {}
        log_imgs = {}
        for k, v in losses.items():
            if "img" in k:
                log_imgs[k] = v
            else:
                real_losses[k] = v

        loss, log_vars = self._parse_losses(real_losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch["img_metas"]), log_imgs=log_imgs)

        return outputs

    def val_step(self, data_batch, **kwargs):






        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):











        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
