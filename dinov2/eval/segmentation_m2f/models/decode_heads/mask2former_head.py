




import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmseg.models.builder import HEADS, build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ...core import build_sampler, multi_apply, reduce_mean
from ..builder import build_assigner
from ..utils import get_uncertain_point_coords_with_randomness


@HEADS.register_module()
class Mask2FormerHead(BaseDecodeHead):




































    def __init__(
        self,
        in_channels,
        feat_channels,
        out_channels,
        num_things_classes=80,
        num_stuff_classes=53,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=None,
        enforce_decoder_input_project=False,
        transformer_decoder=None,
        positional_encoding=None,
        loss_cls=None,
        loss_mask=None,
        loss_dice=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        **kwargs,
    ):
        super(Mask2FormerHead, self).__init__(
            in_channels=in_channels,
            channels=feat_channels,
            num_classes=(num_things_classes + num_stuff_classes),
            init_cfg=init_cfg,
            input_transform="multiple_select",
            **kwargs,
        )
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(in_channels=in_channels, feat_channels=feat_channels, out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        
        for _ in range(num_transformer_feat_level):
            if self.decoder_embed_dims != feat_channels or enforce_decoder_input_project:
                self.decoder_input_projs.append(Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )
        self.conv_seg = None  

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get("num_points", 12544)
            self.oversample_ratio = self.train_cfg.get("oversample_ratio", 3.0)
            self.importance_sample_ratio = self.train_cfg.get("importance_sample_ratio", 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas):
































        (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):





























        
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        
        mask_points_pred = point_sample(mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        
        gt_points_masks = point_sample(gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)).squeeze(1)

        
        assign_result = self.assigner.assign(cls_score, mask_points_pred, gt_labels, gt_points_masks, img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        
        labels = gt_labels.new_full((self.num_queries,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries,))

        
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)

    def loss_single(self, cls_scores, mask_preds, gt_labels_list, gt_masks_list, img_metas):



















        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas)
        
        labels = torch.stack(labels_list, dim=0)
        
        label_weights = torch.stack(label_weights_list, dim=0)
        
        mask_targets = torch.cat(mask_targets_list, dim=0)
        
        mask_weights = torch.stack(mask_weights_list, dim=0)

        
        
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        
        
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points, self.oversample_ratio, self.importance_sample_ratio
            )
            
            mask_point_targets = point_sample(mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        
        mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(1)

        
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        
        
        mask_point_preds = mask_point_preds.reshape(-1, 1)
        
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets, avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    @force_fp32(apply_to=("all_cls_scores", "all_mask_preds"))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list, gt_masks_list, img_metas):


















        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds, all_gt_labels_list, all_gt_masks_list, img_metas_list
        )

        loss_dict = dict()
        
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_mask"] = losses_mask[-1]
        loss_dict["loss_dice"] = losses_dice[-1]
        
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_mask"] = loss_mask_i
            loss_dict[f"d{num_dec_layer}.loss_dice"] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):



















        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        
        cls_pred = self.cls_embed(decoder_out)
        
        mask_embed = self.mask_embed(decoder_out)
        
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False)
        
        
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, feats, img_metas):


















        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            
            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:]
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def forward_train(self, x, img_metas, gt_semantic_seg, gt_labels, gt_masks):



















        
        all_cls_scores, all_mask_preds = self(x, img_metas)

        
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):













        all_cls_scores, all_mask_preds = self(inputs, img_metas)
        cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
        ori_h, ori_w, _ = img_metas[0]["ori_shape"]

        
        cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        seg_mask = torch.einsum("bqc,bqhw->bchw", cls_score, mask_pred)
        return seg_mask
