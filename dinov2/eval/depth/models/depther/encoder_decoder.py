




import torch
import torch.nn.functional as F

from ...models import builder
from ...models.builder import DEPTHER
from ...ops import resize
from .base import BaseDepther


def add_prefix(inputs, prefix):











    outputs = dict()
    for name, value in inputs.items():
        outputs[f"{prefix}.{name}"] = value

    return outputs


@DEPTHER.register_module()
class DepthEncoderDecoder(BaseDepther):





    def __init__(self, backbone, decode_head, neck=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(DepthEncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get("pretrained") is None, "both backbone and depther set pretrained weight"
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self._init_decode_head(decode_head)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners

    def extract_feat(self, img):

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas, rescale=True, size=None):


        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        
        out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        if rescale:
            if size is None:
                if img_metas is not None:
                    size = img_metas[0]["ori_shape"][:2]
                else:
                    size = img.shape[2:]
            out = resize(input=out, size=size, mode="bilinear", align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, img, x, img_metas, depth_gt, **kwargs):


        losses = dict()
        loss_decode = self.decode_head.forward_train(img, x, img_metas, depth_gt, self.train_cfg, **kwargs)
        losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def _decode_head_forward_test(self, x, img_metas):


        depth_pred = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return depth_pred

    def forward_dummy(self, img):

        depth = self.encode_decode(img, None)

        return depth

    def forward_train(self, img, img_metas, depth_gt, **kwargs):
















        x = self.extract_feat(img)

        losses = dict()

        
        loss_decode = self._decode_head_forward_train(img, x, img_metas, depth_gt, **kwargs)

        losses.update(loss_decode)

        return losses

    def whole_inference(self, img, img_meta, rescale, size=None):

        depth_pred = self.encode_decode(img, img_meta, rescale, size=size)

        return depth_pred

    def slide_inference(self, img, img_meta, rescale):






        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                depth_pred = self.encode_decode(crop_img, img_meta, rescale)
                preds += F.pad(depth_pred, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        return preds

    def inference(self, img, img_meta, rescale, size=None):















        assert self.test_cfg.mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == "slide":
            depth_pred = self.slide_inference(img, img_meta, rescale)
        else:
            depth_pred = self.whole_inference(img, img_meta, rescale, size=size)
        output = depth_pred
        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):

        depth_pred = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            
            depth_pred = depth_pred.unsqueeze(0)
            return depth_pred
        depth_pred = depth_pred.cpu().numpy()
        
        depth_pred = list(depth_pred)
        return depth_pred

    def aug_test(self, imgs, img_metas, rescale=True):




        
        assert rescale
        
        depth_pred = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_depth_pred = self.inference(imgs[i], img_metas[i], rescale, size=depth_pred.shape[-2:])
            depth_pred += cur_depth_pred
        depth_pred /= len(imgs)
        depth_pred = depth_pred.cpu().numpy()
        
        depth_pred = list(depth_pred)
        return depth_pred
