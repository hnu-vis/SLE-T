
import copy
import logging
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from adapteacher.data.detection_utils import build_strong_augmentation

from fvcore.transforms.transform import (
    Transform,
)
import torch.nn.functional as F
from math import floor
from fvcore.transforms.transform import CropTransform, PadTransform, TransformList
from detectron2.data.transforms.augmentation import Augmentation
from cityscapesscripts.helpers.labels import id2label



















































































































































class DatasetMapperTwoCropSeparateKeepTf(DatasetMapper):





    def __init__(self, cfg, is_train=True, keep_tf_data=False):
        

        if cfg.SEMISUPNET.USE_FEATURE_ALIGN or 'vit' in cfg.MODEL.BACKBONE.NAME:
            crop_to_patch_size = True
            self.augmentation = augs_with_transformer_patch(cfg, is_train)
        else:
            crop_to_patch_size = False
            self.augmentation = utils.build_augmentation(cfg, is_train)
        
        if cfg.INPUT.CROP.ENABLED and is_train:
            if crop_to_patch_size:
                self.augmentation.insert(
                    0, RandomCropAndPad(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
                )
            else:
                self.augmentation.insert(
                    0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
                )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        
        self.img_format = cfg.INPUT.FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        self.keep_tf_data = keep_tf_data

        self.mask_format = 'polygon'
        self.mask_on = cfg.MODEL.MASK_ON

    def __call__(self, dataset_dict):







        dataset_dict = copy.deepcopy(dataset_dict)  
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  

        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.copy()
            labelIds = np.unique(sem_seg_gt)
            
            
            
            
            
            
            
            
            for labelId in labelIds:
                trainId = id2label[labelId].trainId
                mask = sem_seg_gt == labelId
                sem_seg_gt[mask] = trainId
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            for i in range(len(annos)):
                if 'segmentation' in annos[i].keys() and self.mask_format == "polygon":
                    if type(annos[i]['segmentation']) != list:
                        annos[i]['segmentation'] = [annos[i]['segmentation']]
                    
                    
                    
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        if self.keep_tf_data:
            dataset_dict['tf_data'] = transforms

        
        
        
        
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        img_strong = self.strong_augmentation(image_pil)
        image_strong_aug = np.array(img_strong)
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)





























































































































        






































































































































    











































































































































    
def augs_with_transformer_patch(cfg, is_train, use_w=False):







    dino_patch = cfg.SEMISUPNET.DINO_PATCH_SIZE
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        sample_style = "choice"
    augmentation = [ResizeTransformDinoScale(min_size, dino_patch, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation

class ResizeTransformDinoScale(Transform):




    def __init__(self, new_h, dino_patch, inv=False, interp=None):






        
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        if type(new_h) == tuple:
            new_h = new_h[0]
        self.set_h = new_h
        self.dino_patch = dino_patch
        self.inv = inv
        self.interp = interp

    def apply_image(self, img, interp=None):
        
        assert img.shape[0] < img.shape[1]
        assert len(img.shape) <= 4
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.new_h = floor(self.set_h / self.dino_patch) * self.dino_patch
        scale = self.new_h / self.h
        temp_w = scale * self.w
        self.new_w = round(temp_w / self.dino_patch) * self.dino_patch

        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransformDinoScale(self.new_h, self.new_w, self.h, self.w, self.interp)

















































































class RandomCropAndPad(Augmentation):




    def __init__(self, crop_type: str, crop_size):

















        
        
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        crop_tf = CropTransform(w0, h0, cropw, croph)
        dh = h - croph
        dw = w - cropw

        
        pad_tf = PadTransform(0,0,dw,dh)
        return TransformList([crop_tf, pad_tf])

    def get_crop_size(self, image_size):







        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))
