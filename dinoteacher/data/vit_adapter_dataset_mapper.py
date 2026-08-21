
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.config import get_cfg

import copy
import random

import random as py_random

__all__ = ["ViT_Adapter_DatasetMapper"]

class RandomChoice(T.Augmentation):




    def __init__(self, transforms):




        super().__init__()
        self.transforms = transforms
    
    def get_transform(self, image):
        
        selected = py_random.choice(self.transforms)
        
        return selected.get_transform(image)
    
    def __call__(self, aug_input):
        
        selected = py_random.choice(self.transforms)
        return selected(aug_input)


class ViT_Adapter_DatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        
        self.img_format = cfg.INPUT.FORMAT
        
        if is_train:
            
            
            
            policy1 = T.AugmentationList([
                T.ResizeShortestEdge(
                    
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomFlip(prob=cfg.INPUT.RANDOM_FLIP_PROB, horizontal=True),
            ])
            
            
            policy2 = T.AugmentationList([
                
                T.ResizeShortestEdge(
                    short_edge_length=[400, 500, 600],
                    
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                
                T.RandomCrop(
                    crop_type="absolute_range",
                    crop_size=(384, 600)
                ),
                
                T.ResizeShortestEdge(
                    
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomFlip(prob=cfg.INPUT.RANDOM_FLIP_PROB, horizontal=True),
            ])
            
            
            auto_augment = RandomChoice([policy1, policy2])
            
            
            
            second_crop = T.RandomCrop(
                crop_type="absolute_range",
                crop_size=(1024, 1024)
            )
            
            
            self.augmentations = T.AugmentationList([
                auto_augment,
                second_crop,
            ])
            
        else:
            
            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                    max_size=cfg.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                ),
                
            ])
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        
        original_h, original_w = image.shape[:2]
        
        
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]
        
         
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        
        h, w = image.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),  
                mode='constant',
                constant_values=0.0
            )
        
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        return dataset_dict
        