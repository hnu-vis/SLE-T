




import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
from detectron2.data import MetadataCatalog

from adapteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from adapteacher.engine.hooks import LossEvalHook
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.solver.build import build_lr_scheduler as build_adapteacher_lr_scheduler

from dinoteacher.data.dataset_mapper import DatasetMapperTwoCropSeparateKeepTf, DatasetMapperTwoCropSeparateKeepTf_lt
from dinoteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from dinoteacher.engine.align_head import TeacherStudentAlignHead, TeacherStudentAlignHead_GC, TeacherStudentAlignHead_lt
from dinoteacher.engine.build_dino import DinoVitFeatureExtractor, DinoVitFeatureExtractor_GC, DinoVitFeatureExtractor_14_16

from adapteacher.engine.probe import OpenMatchTrainerProbe
import copy
import pickle

from IPython import embed
from torch import nn
from geomloss import SamplesLoss
import torch.nn.functional as F
import torch.nn.init as init

from dinoteacher.engine.build_dinov3 import DinoVitFeatureExtractor_v3
from dinoteacher.data.vit_adapter_dataset_mapper import ViT_Adapter_DatasetMapper

from dinoteacher.solver.build_wd_norm_bias import build_lr_scheduler_v2, build_optimizer_with_layer_decay
from detectron2.solver import build_optimizer
from dinoteacher.modeling.meta_arch.adv_grl import grad_reverse
import random

from .utils import PascalVOCDetectionPerClassEvaluator

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class CosineSinkhornKDLoss(nn.Module):
    def __init__(self, epsilon=0.1, sinkhorn_iters=100):
        super().__init__()
        self.epsilon = epsilon
        self.iters = sinkhorn_iters

    def forward(self, T, S):




        B, N, C = T.shape
        _, M, _ = S.shape

        
        T = F.normalize(T, p=2, dim=-1)
        S = F.normalize(S, p=2, dim=-1)

        losses = []
        for b in range(B):
            cost = 1 - torch.matmul(T[b], S[b].T)   
            mu = torch.full((N,), 1/N, device=T.device)
            nu = torch.full((M,), 1/M, device=S.device)

            K = torch.exp(-cost / self.epsilon)
            u = torch.ones_like(mu)
            v = torch.ones_like(nu)

            for _ in range(self.iters):
                u = mu / (K @ v)
                v = nu / (K.T @ u)

            P = torch.diag(u) @ K @ torch.diag(v)
            loss_b = torch.sum(P * cost)
            losses.append(loss_b)

        return torch.mean(torch.stack(losses))


class DINOTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg, wandb_run=None):





        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        
        self.use_adversarial_invariance = cfg.SEMISUPNET.DIS_LOSS_WEIGHT > 0
        self.branch = 'supervised'
        if cfg.SEMISUPNET.USE_FEATURE_ALIGN:
            
            self.align_layer = cfg.SEMISUPNET.FEATURE_ALIGN_LAYER
            self.align_teacher_pool = cfg.SEMISUPNET.POOL
            if self.align_teacher_pool == "None":
                self.align_teacher_pool = None
            if self.align_teacher_pool not in (None, "max_pool", "mean", "mean_pool"):
                raise NotImplementedError("{} pooling not supported.".format(self.align_teacher_pool))
            self.use_feature_align = True
            self.student_align_feat = {}
            student_align_dim = model.backbone._out_feature_channels[cfg.SEMISUPNET.FEATURE_ALIGN_LAYER]
            model.align_teacher = DinoVitFeatureExtractor(cfg, model_name=cfg.SEMISUPNET.ALIGN_MODEL, normalize_feature=cfg.SEMISUPNET.ALIGN_HEAD_NORMALIZE).eval()

            
            
            

            teacher_align_dim = [*model.align_teacher.modules()][-2].normalized_shape[0]
            model.align_student_head = TeacherStudentAlignHead(cfg, student_align_dim, teacher_align_dim, normalize_feature=model.align_teacher.normalize_feature)
            self._register_input_hook_feat_align(model, 'proposal_generator')

            model.align_teacher = model.align_teacher.to((torch.device(model.device)))
            model.align_student_head = model.align_student_head.to((torch.device(model.device)))   
            
            
        else:
            self.use_feature_align = False

        if type(cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT) == str:
            file_in = cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT
            self.use_dino_PL = True
            with open(file_in, 'rb') as f_in:
                temp_dict = pickle.load(f_in)
            self.dino_pseudogt = {}
            for img in temp_dict:
                self.dino_pseudogt[img['image_id']] = img
        else:
            self.use_dino_PL = False

        if type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP) == str:
            assert type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER) == int
            self.PL_swap = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP
            self.PL_swap_iter = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER
        else:
            self.PL_swap = None

        
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            if self.use_feature_align:
                model.align_teacher = model.module.align_teacher
                model.align_student_head = model.module.align_student_head
                model.forward_backbone = model.module.forward_backbone


        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        

        
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

        
        
        
        
        

    def _pool_align_teacher_feat(self, feat):
        if self.align_teacher_pool is None:
            return feat
        if self.align_teacher_pool == "max_pool":
            return F.max_pool2d(feat, kernel_size=2, stride=2)
        if self.align_teacher_pool in ("mean", "mean_pool"):
            return F.avg_pool2d(feat, kernel_size=2, stride=2)
        raise NotImplementedError("{} pooling not supported.".format(self.align_teacher_pool))

    def resume_or_load(self, resume=True):











        print('\n\n\n\n',self.cfg.MODEL.WEIGHTS,'\n\n\n\n')
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            
            
        if isinstance(self.model, DistributedDataParallel):
            
            
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        
        
        
        
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionPerClassEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparateKeepTf(cfg, is_train=True, keep_tf_data=True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_adapteacher_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        elif proposal_type == "dino":
            valid_map = proposal_bbox_inst.gt_scores > thres
            new_proposal_inst = proposal_bbox_inst[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, pseudo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "Model in eval mode while training, set it to train mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        
        
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        
        if self.iter % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            self._update_teacher_model(
                keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        
        
        
        

        record_dict = {}
        

        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q + unlabel_data_k
            self.branch = "supervised"
            
            
            record_dict, _, _, _, _, _ = self.model(
                all_label_data, branch="supervised")

            has_target_backbone_feats = 0                

        else:
            
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            
            use_DT_labels = False
            if self.use_dino_PL:
                use_DT_labels = True
                if self.PL_swap == 'full':
                    if self.iter > self.PL_swap_iter:
                        use_DT_labels = False
                elif self.PL_swap == 'half':
                    if self.iter > self.PL_swap_iter and self.iter % 2:
                        use_DT_labels = False

            if use_DT_labels:
                
                instances = [self.dino_pseudogt[x['image_id']]['instances_dino'] for x in unlabel_data_q]
                boxes = [(x['tf_data'].apply_box(y.pred_boxes),y.scores,y.pred_classes) for x,y in zip(unlabel_data_q,instances)]
                dino_pseudo_labels = []
                for i in range(len(instances)):
                    new_instances = Instances(unlabel_data_k[i]['image'].shape[-2:])
                    new_instances.gt_boxes = Boxes(boxes[i][0])
                    new_instances.gt_scores = boxes[i][1]
                    new_instances.gt_classes = boxes[i][2]
                    dino_pseudo_labels.append(new_instances)
                
                joint_proposal_dict = {}
                pseudo_proposals_dino, num_pseudo_bbox_roih = self.process_pseudo_label(
                    dino_pseudo_labels, cur_threshold, "dino", "thresholding")
                joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_dino

            else:
                
                self.branch = "unsup_data_weak"
                with torch.no_grad():
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

                
                joint_proposal_dict = {}

                
                
                
                
                
                

                
                pseudo_proposals_roih_unsup_k, num_pseudo_bbox_roih = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")
                joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k

            
            all_label_data = label_data_q + label_data_k
            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
            if use_DT_labels:
                unlabel_data_k = self.add_label(
                    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])
                all_unlabel_data = unlabel_data_q + unlabel_data_k
            else:
                all_unlabel_data = unlabel_data_q

            
            self.branch = "supervised"
            record_all_label_data, _, _, _, _, _ = self.model(
                all_label_data, branch="supervised")
            record_dict.update(record_all_label_data)

            
            self.branch = "supervised_target"
            record_all_unlabel_data, _, _, _, _, _  = self.model(
                all_unlabel_data, branch="supervised_target")
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = \
                    record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)

            
            if use_DT_labels:
                has_target_backbone_feats = 2
            else:
                has_target_backbone_feats = 1

            if self.use_adversarial_invariance:
                
                
                for i_index in range(len(unlabel_data_k)):
                    
                    for k, v in unlabel_data_k[i_index].items():
                        
                        label_data_k[i_index][k + "_unlabeled"] = v
                    

                all_domain_data = label_data_k
                
                self.branch = "domain"
                record_all_domain_data, _, _, _ = self.model(
                    all_domain_data, branch="domain")
                record_dict.update(record_all_domain_data)
        
        
        if self.use_feature_align:
            
            if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                
                
                
                easy_feat = self._pool_align_teacher_feat(self.model.align_teacher(label_data_k))
                teacher_feat = easy_feat.repeat(2,1,1,1)
            else:
                teacher_feat = self._pool_align_teacher_feat(self.model.align_teacher(all_label_data))
            
            
            
            
            student_feat = self.model.align_student_head(
                self.student_align_feat['supervised'], teacher_feat.shape[2:])
            align_loss = self.model.align_student_head.align_loss(student_feat, teacher_feat)
            record_dict['loss_align'] = align_loss

            if self.iter >= self.cfg.SEMISUPNET.FEATURE_ALIGN_TARGET_START:
                
                
                

                if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                    
                    easy_feat_target = self._pool_align_teacher_feat(self.model.align_teacher(unlabel_data_k))
                    if has_target_backbone_feats == 0 or has_target_backbone_feats == 2:
                        teacher_feat_target = easy_feat_target.repeat(2,1,1,1)
                    else:
                        teacher_feat_target = easy_feat_target
                else:
                    teacher_feat_target = self._pool_align_teacher_feat(self.model.align_teacher(all_unlabel_data))

                if has_target_backbone_feats == 0:
                    all_unlabel_data = unlabel_data_q + unlabel_data_k
                    backbone_feat_target = self.model.forward_backbone(all_unlabel_data)[self.cfg.SEMISUPNET.FEATURE_ALIGN_LAYER] 
                    student_feat_target = self.model.align_student_head(
                        backbone_feat_target, teacher_feat_target.shape[2:])
                else:
                    student_feat_target = self.model.align_student_head(
                        self.student_align_feat['supervised_target'], teacher_feat_target.shape[2:])
                align_loss_target = self.model.align_student_head.align_loss(
                    student_feat_target, teacher_feat_target)
                record_dict['loss_align_target'] = align_loss_target


        
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                    
                    loss_dict[key] = record_dict[key] * 0
                elif key[-6:] == "pseudo":  
                    loss_dict[key] = (
                        record_dict[key] *
                        self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    )
                elif (
                    key == "loss_D_img_s" or key == "loss_D_img_t"
                ):  
                    
                    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif key == "loss_align":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT
                elif key == "loss_align_target":    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET
                else:  
                    loss_dict[key] = record_dict[key] * 1

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()


    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        
        
        
        all_metrics_dict = comm.gather(metrics_dict)
        

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                
                
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        
        
        
        if not getattr(self, "_teacher_ema_initialized", False):
            self._copy_main_model()
            self._teacher_ema_initialized = True
            return

        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            
            self.model_teacher.load_state_dict(self.model.state_dict(), strict=False)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                
                cfg.TEST.EVAL_PERIOD,
                self.model,
                
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        
        
        
        
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        
        
        
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _get_detector_input_hook(self, module, input, output):
        self.student_align_feat[self.branch] = input[1][self.cfg.MODEL.RPN.IN_FEATURES[0]]
        
        
        

    def _register_input_hook_feat_align(self, model, target_layer):
        for (name, module) in model.named_modules():
            if name == target_layer:
                module.register_forward_hook(self._get_detector_input_hook)
        return True



class DINOTeacherTrainer_GC(DefaultTrainer):
    def __init__(self, cfg, wandb_run=None):





        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        
        self.use_adversarial_invariance = cfg.SEMISUPNET.DIS_LOSS_WEIGHT > 0
        self.branch = 'supervised'
        if cfg.SEMISUPNET.USE_FEATURE_ALIGN:
            
            self.align_layer = cfg.SEMISUPNET.FEATURE_ALIGN_LAYER
            self.use_feature_align = True
            self.student_align_feat = {}
            student_align_dim = model.backbone._out_feature_channels[cfg.SEMISUPNET.FEATURE_ALIGN_LAYER]
            model.align_teacher = DinoVitFeatureExtractor_GC(cfg, model_name=cfg.SEMISUPNET.ALIGN_MODEL, normalize_feature=cfg.SEMISUPNET.ALIGN_HEAD_NORMALIZE).eval()
            
            
            teacher_align_dim = [*model.align_teacher.gcBlock.modules()][3][-1].weight.shape[0]

            if(cfg.SEMISUPNET.T_ADAPT == True):
                model.t_adapt_layer = nn.Sequential(nn.Conv2d(teacher_align_dim, cfg.SEMISUPNET.T_ADAPT_PORJ_DIM, 1, 1),
                                                
                                                   nn.ReLU(),
                                                   nn.Conv2d(cfg.SEMISUPNET.T_ADAPT_PORJ_DIM, teacher_align_dim, 1, 1))
                model.t_adapt_layer.apply(init_weights)

            
            model.align_student_head = TeacherStudentAlignHead_GC(cfg, student_align_dim, teacher_align_dim, normalize_feature=cfg.SEMISUPNET.ALIGN_HEAD_NORMALIZE)
            self._register_input_hook_feat_align(model, 'proposal_generator')

            model.align_teacher = model.align_teacher.to((torch.device(model.device)))
            model.align_student_head = model.align_student_head.to((torch.device(model.device)))
            if(cfg.SEMISUPNET.T_ADAPT == True):
                model.t_adapt_layer = model.t_adapt_layer.to((torch.device(model.device)))
            
            
        else:
            self.use_feature_align = False


        if type(cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT) == str:
            file_in = cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT
            self.use_dino_PL = True
            with open(file_in, 'rb') as f_in:
                temp_dict = pickle.load(f_in)
            self.dino_pseudogt = {}
            for img in temp_dict:
                self.dino_pseudogt[img['image_id']] = img
        else:
            self.use_dino_PL = False

        if type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP) == str:
            assert type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER) == int
            self.PL_swap = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP
            self.PL_swap_iter = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER
        else:
            self.PL_swap = None

        
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            if self.use_feature_align:
                model.align_teacher = model.module.align_teacher
                model.align_student_head = model.module.align_student_head
                model.forward_backbone = model.module.forward_backbone


        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        

        
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())
        
        
        
        
        

    def resume_or_load(self, resume=True):











        print('\n\n\n\n',self.cfg.MODEL.WEIGHTS,'\n\n\n\n')
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            
            
        if isinstance(self.model, DistributedDataParallel):
            
            
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparateKeepTf(cfg, is_train=True, keep_tf_data=True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_adapteacher_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                
                self.freeze_model()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
    
    def freeze_model(self):
        
        if self.use_feature_align:
            self.model.align_teacher.eval()
            self.model.align_teacher.requires_grad_(False)

    
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        elif proposal_type == "dino":
            valid_map = proposal_bbox_inst.gt_scores > thres
            new_proposal_inst = proposal_bbox_inst[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, pseudo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    
    def compare_state_dict(self, sd1, sd2, atol=1e-6, rtol=1e-5, ignore_bn_stats=True):
        if sd1.keys() != sd2.keys():
            return False, "Keys mismatch"

        for k in sd1:
            if ignore_bn_stats and (
                "running_mean" in k or
                "running_var" in k or
                "num_batches_tracked" in k
            ):
                continue

            if not torch.allclose(sd1[k], sd2[k], atol=atol, rtol=rtol):
                return False, k

        return True, None
    
    def get_state_dict_gc(self):
        gc_ckpt_path = self.cfg.SEMISUPNET.GC_CKPT_PATH
        full_state_dict = torch.load(gc_ckpt_path, map_location='cuda')
        
        prefix = "modelStudent.backbone.gcBlock."
        pretrained_gc_state_dict = {
            k[len(prefix):]: v
            for k, v in full_state_dict['model'].items()
            if k.startswith(prefix)
        }
        return pretrained_gc_state_dict

    def DT_pseudo_label_process(self, label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict):

        
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

        
        use_DT_labels = False
        if self.use_dino_PL:
            use_DT_labels = True
            if self.PL_swap == 'full':
                if self.iter > self.PL_swap_iter:
                    use_DT_labels = False
            elif self.PL_swap == 'half':
                if self.iter > self.PL_swap_iter and self.iter % 2:
                    use_DT_labels = False

        if use_DT_labels:
            
            instances = [self.dino_pseudogt[x['image_id']]['instances_dino'] for x in unlabel_data_q]
            boxes = [(x['tf_data'].apply_box(y.pred_boxes),y.scores,y.pred_classes) for x,y in zip(unlabel_data_q,instances)]
            dino_pseudo_labels = []
            for i in range(len(instances)):
                new_instances = Instances(unlabel_data_k[i]['image'].shape[-2:])
                new_instances.gt_boxes = Boxes(boxes[i][0])
                new_instances.gt_scores = boxes[i][1]
                new_instances.gt_classes = boxes[i][2]
                dino_pseudo_labels.append(new_instances)
            
            joint_proposal_dict = {}
            pseudo_proposals_dino, num_pseudo_bbox_roih = self.process_pseudo_label(
                dino_pseudo_labels, cur_threshold, "dino", "thresholding")
            joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_dino

        else:
            
            self.branch = "unsup_data_weak"
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            
            joint_proposal_dict = {}

            
            
            
            
            
            

            
            pseudo_proposals_roih_unsup_k, num_pseudo_bbox_roih = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")
            joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k

        
        all_label_data = label_data_q + label_data_k
        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
        if use_DT_labels:
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])
            all_unlabel_data = unlabel_data_q + unlabel_data_k
        else:
            all_unlabel_data = unlabel_data_q

        
        self.branch = "supervised"
        record_all_label_data, _, _, _, images_all_label, gt_instances_all_label = self.model(
            all_label_data, branch="supervised")
        record_dict.update(record_all_label_data)

        
        self.branch = "supervised_target"
        record_all_unlabel_data, _, _, _, images_all_unlabel, gt_instances_all_unlabel = self.model(
            all_unlabel_data, branch="supervised_target")

        new_record_all_unlabel_data = {}
        if(self.cfg.SEMISUPNET.PSEUDO_KD == True):
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = \
                    record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)
        else:
            pass

        
        if use_DT_labels:
            has_target_backbone_feats = 2
        else:
            has_target_backbone_feats = 1

        if self.use_adversarial_invariance:
            
            
            for i_index in range(len(unlabel_data_k)):
                
                for k, v in unlabel_data_k[i_index].items():
                    
                    label_data_k[i_index][k + "_unlabeled"] = v
                

            all_domain_data = label_data_k
            
            self.branch = "domain"
            record_all_domain_data, _, _, _ = self.model(
                all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)
        
        return has_target_backbone_feats

    
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "Model in eval mode while training, set it to train mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        
        
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        
        
        if self.iter % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            self._update_teacher_model(
                keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)


        
        
        
        
        

        
        record_dict = {}
        
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q + unlabel_data_k
            self.branch = "supervised"
            if(self.cfg.MODEL.META_ARCHITECTURE == 'DAobjTwoStagePseudoLabGeneralizedRCNN_GC_shortcut'):
                record_dict, _, _, _ = self.model(
                    all_label_data, branch="supervised")
            elif(self.cfg.MODEL.META_ARCHITECTURE == 'DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut'):
                record_dict, _, _, _, _, _ = self.model(
                    all_label_data, branch="supervised")
            else:
                raise NotImplementedError()

            has_target_backbone_feats = 0                

        else:
            has_target_backbone_feats = self.DT_pseudo_label_process(label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict)
        
        
        if self.use_feature_align:
            
            if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                
                with torch.no_grad():
                    
                    easy_feat = self.model.align_teacher(label_data_k)
                    teacher_feat = easy_feat.repeat(2,1,1,1)
            else:
                with torch.no_grad():
                    teacher_feat = self.model.align_teacher(all_label_data)

            if(self.cfg.SEMISUPNET.T_ADAPT == True):
                teacher_feat = self.model.t_adapt_layer(teacher_feat)

            if(self.cfg.SEMISUPNET.ALIGN_METHOD == "PIXEL"):
                
                
                student_feat = self.model.align_student_head(
                    self.student_align_feat['supervised'], teacher_feat.shape[2:])
                align_loss = self.model.align_student_head.align_loss(student_feat, teacher_feat)

            elif(self.cfg.SEMISUPNET.ALIGN_METHOD == "Sinkhorn"):
                align_loss = self.wasserstein_sinkhorn_loss(self.student_align_feat['supervised'], teacher_feat)

            else:
                raise NotImplementedError()
            record_dict['loss_align'] = align_loss
            
            
            
            
            
            if(self.cfg.SEMISUPNET.CROSS_ALIGN == True):
                
                if(self.cfg.MODEL.BACKBONE.NAME == "build_vgg_backbone"):
                    self.branch = "supervised"
                    
                    max_pool = nn.AdaptiveAvgPool2d((self.student_align_feat['supervised'].shape[-2], self.student_align_feat['supervised'].shape[-1]))
                    t_f_ups_source_k = max_pool(teacher_feat[:teacher_feat.shape[0]//2])
                    
                    rpn_k_loss_dict, roi_head_k_loss_dict = self.get_cross_align_loss(label_data_k, {'vgg4':t_f_ups_source_k}, branch="supervised")
                    for key_ in rpn_k_loss_dict.keys():
                        record_dict[key_ + "_CA"] = rpn_k_loss_dict[key_]
                    for key_ in roi_head_k_loss_dict.keys():
                        record_dict[key_ + "_CA"] = roi_head_k_loss_dict[key_]
                else:
                    raise NotImplementedError()
            

            if self.iter >= self.cfg.SEMISUPNET.FEATURE_ALIGN_TARGET_START:
                
                
                

                with torch.no_grad():
                    
                    if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                        
                        easy_feat_target = self.model.align_teacher(unlabel_data_k)
                        if has_target_backbone_feats == 0 or has_target_backbone_feats == 2:
                            teacher_feat_target = easy_feat_target.repeat(2,1,1,1)
                        else:
                            teacher_feat_target = easy_feat_target
                    else:
                        teacher_feat_target = self.model.align_teacher(all_unlabel_data)

                if(self.cfg.SEMISUPNET.T_ADAPT == True):
                    teacher_feat = self.model.t_adapt_layer(teacher_feat)

                if(self.cfg.SEMISUPNET.ALIGN_METHOD == "PIXEL"):
                    if has_target_backbone_feats == 0:
                        all_unlabel_data = unlabel_data_q + unlabel_data_k
                        backbone_feat_target = self.model.forward_backbone(all_unlabel_data)[self.cfg.SEMISUPNET.FEATURE_ALIGN_LAYER] 
                        student_feat_target = self.model.align_student_head(
                            backbone_feat_target, teacher_feat_target.shape[2:])
                    else:
                        
                        student_feat_target = self.model.align_student_head(
                            self.student_align_feat['supervised_target'], teacher_feat_target.shape[2:])
                    align_loss_target = self.model.align_student_head.align_loss(
                        student_feat_target, teacher_feat_target)

                elif(self.cfg.SEMISUPNET.ALIGN_METHOD == "Sinkhorn"):
                    if has_target_backbone_feats == 0:
                        all_unlabel_data = unlabel_data_q + unlabel_data_k
                        student_feat_target = self.model.forward_backbone(all_unlabel_data)[self.cfg.SEMISUPNET.FEATURE_ALIGN_LAYER]
                    else:
                        student_feat_target = self.student_align_feat['supervised_target']
                    align_loss_target = self.wasserstein_sinkhorn_loss(student_feat_target, teacher_feat_target)
                else:
                    raise NotImplementedError()
                
                
                
                
                
                
                
                
                
                record_dict['loss_align_target'] = align_loss_target
                
                
                if(self.cfg.SEMISUPNET.CROSS_ALIGN == True):
                    
                    if(self.cfg.MODEL.BACKBONE.NAME == "build_vgg_backbone"):
                        self.branch = "supervised_target"
                        
                        max_pool = nn.AdaptiveAvgPool2d((self.student_align_feat['supervised_target'].shape[-2], self.student_align_feat['supervised_target'].shape[-1]))
                        t_f_ups_target_k = max_pool(teacher_feat_target[:teacher_feat_target.shape[0]//2])
                        
                        rpn_unlabel_k_loss_dict, roi_head_unlabel_k_loss_dict = self.get_cross_align_loss(unlabel_data_k, {'vgg4':t_f_ups_target_k}, branch="supervised_target")
                        for key_ in rpn_unlabel_k_loss_dict.keys():
                            record_dict[key_ + "_CA_pseudo"] = rpn_unlabel_k_loss_dict[key_]
                        for key_ in roi_head_unlabel_k_loss_dict.keys():
                            record_dict[key_ + "_CA_pseudo"] = roi_head_unlabel_k_loss_dict[key_]
                    else:
                        raise NotImplementedError()  
                
                
        
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo" \
                    or key == "loss_rpn_loc_CA_pseudo" or key == "loss_box_reg_CA_pseudo":
                    
                    loss_dict[key] = record_dict[key] * 0
                elif key[-6:] == "pseudo":  
                    loss_dict[key] = (
                        record_dict[key] *
                        self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    )
                elif (
                    key == "loss_D_img_s" or key == "loss_D_img_t"
                ):  
                    
                    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif key == "loss_align":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT
                elif key == "loss_align_target":    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET
                
                elif key[-2:] == "CA":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CROSS_ALING_WEIGHT
                elif key[-10:] == "_CA_target":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CROSS_ALING_WEIGHT_TARGET
                    
                else:  
                    loss_dict[key] = record_dict[key] * 1

        losses = sum(loss_dict.values())

        
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def cosine_cost(x, y, **kwargs):
        cosine_sim = torch.bmm(x, y.transpose(1, 2))
        return 1 - cosine_sim  

    def wasserstein_sinkhorn_loss(self, student_feat, teacher_feat, normalize=True):
        
        

        batch_size, C, H_s, W_s = student_feat.shape
        
        

        
        teacher_points = teacher_feat.view(batch_size, C, -1).permute(0, 2, 1) 
        student_points = student_feat.view(batch_size, C, -1).permute(0, 2, 1) 

        if(normalize == True):
            criterion = CosineSinkhornKDLoss(epsilon=0.05, sinkhorn_iters=50)
            loss = criterion(teacher_points, student_points)

        else:
            sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

            
            total_loss = 0
            for i in range(batch_size):
                
                total_loss += sinkhorn_loss(teacher_points[i], student_points[i])

            loss = total_loss / batch_size
        
        return loss

    def get_cross_align_loss(self, data, features, branch):
        imgs_label_k = self.model.preprocess_image(data)
        if "instances" in data[0]:
            gt_instances_label = [x["instances"].to(self.model.device) for x in data]
        else:
            gt_instances_label = None
            
        proposals_rpn_label, proposal_losses_label = self.model.proposal_generator(imgs_label_k, features, gt_instances_label)
        _, detector_losses = self.model.roi_heads(
            imgs_label_k,
            features,
            proposals_rpn_label,
            compute_loss=True,
            targets=gt_instances_label,
            branch=branch,
            )
        return proposal_losses_label, detector_losses
    
    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        
        
        
        all_metrics_dict = comm.gather(metrics_dict)
        

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                
                
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                
                cfg.TEST.EVAL_PERIOD,
                self.model,
                
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        
        
        
        
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        
        
        
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _get_detector_input_hook(self, module, input, output):
        self.student_align_feat[self.branch] = input[1][self.cfg.MODEL.RPN.IN_FEATURES[0]]


    def _register_input_hook_feat_align(self, model, target_layer):
        for (name, module) in model.named_modules():
            if name == target_layer:
                module.register_forward_hook(self._get_detector_input_hook)
        return True


class DINOTeacherTrainer_va(DefaultTrainer):
    def __init__(self, cfg, wandb_run=None):





        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        
        self.use_adversarial_invariance = cfg.SEMISUPNET.DIS_LOSS_WEIGHT > 0
        self.branch = 'supervised'
        if cfg.SEMISUPNET.USE_FEATURE_ALIGN:
            
            self.align_layer = cfg.SEMISUPNET.FEATURE_ALIGN_LAYER
            self.use_feature_align = True
            self.student_align_feat = {}
            student_align_dim = model.backbone._out_feature_channels[cfg.SEMISUPNET.FEATURE_ALIGN_LAYER]
            model.align_teacher = DinoVitFeatureExtractor_GC(cfg, model_name=cfg.SEMISUPNET.ALIGN_MODEL, normalize_feature=cfg.SEMISUPNET.ALIGN_HEAD_NORMALIZE).eval()
            
            
            teacher_align_dim = [*model.align_teacher.gcBlock.modules()][3][-1].weight.shape[0]

            if(cfg.SEMISUPNET.T_ADAPT == True):
                model.t_adapt_layer = nn.Sequential(nn.Conv2d(teacher_align_dim, cfg.SEMISUPNET.T_ADAPT_PORJ_DIM, 1, 1),
                                                
                                                   nn.ReLU(),
                                                   nn.Conv2d(cfg.SEMISUPNET.T_ADAPT_PORJ_DIM, teacher_align_dim, 1, 1))
                model.t_adapt_layer.apply(init_weights)

            
            model.align_student_head = TeacherStudentAlignHead_GC(cfg, student_align_dim, teacher_align_dim, normalize_feature=cfg.SEMISUPNET.ALIGN_HEAD_NORMALIZE)
            self._register_input_hook_feat_align(model, 'proposal_generator')

            model.align_teacher = model.align_teacher.to((torch.device(model.device)))
            model.align_student_head = model.align_student_head.to((torch.device(model.device)))
            if(cfg.SEMISUPNET.T_ADAPT == True):
                model.t_adapt_layer = model.t_adapt_layer.to((torch.device(model.device)))
            
            
        else:
            self.use_feature_align = False


        if type(cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT) == str:
            file_in = cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT
            self.use_dino_PL = True
            with open(file_in, 'rb') as f_in:
                temp_dict = pickle.load(f_in)
            self.dino_pseudogt = {}
            for img in temp_dict:
                self.dino_pseudogt[img['image_id']] = img
        else:
            self.use_dino_PL = False

        if type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP) == str:
            assert type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER) == int
            self.PL_swap = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP
            self.PL_swap_iter = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER
        else:
            self.PL_swap = None

        
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            if self.use_feature_align:
                model.align_teacher = model.module.align_teacher
                model.align_student_head = model.module.align_student_head
                model.forward_backbone = model.module.forward_backbone


        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        

        
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())
        
        
        
        
        

    def resume_or_load(self, resume=True):











        print('\n\n\n\n',self.cfg.MODEL.WEIGHTS,'\n\n\n\n')
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            
            
        if isinstance(self.model, DistributedDataParallel):
            
            
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = ViT_Adapter_DatasetMapper(cfg, is_train=True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_adapteacher_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                
                self.freeze_model()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
    
    def freeze_model(self):
        
        if self.use_feature_align:
            self.model.align_teacher.eval()
            self.model.align_teacher.requires_grad_(False)

    
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        elif proposal_type == "dino":
            valid_map = proposal_bbox_inst.gt_scores > thres
            new_proposal_inst = proposal_bbox_inst[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, pseudo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    
    def compare_state_dict(self, sd1, sd2, atol=1e-6, rtol=1e-5, ignore_bn_stats=True):
        if sd1.keys() != sd2.keys():
            return False, "Keys mismatch"

        for k in sd1:
            if ignore_bn_stats and (
                "running_mean" in k or
                "running_var" in k or
                "num_batches_tracked" in k
            ):
                continue

            if not torch.allclose(sd1[k], sd2[k], atol=atol, rtol=rtol):
                return False, k

        return True, None
    
    def get_state_dict_gc(self):
        gc_ckpt_path = self.cfg.SEMISUPNET.GC_CKPT_PATH
        full_state_dict = torch.load(gc_ckpt_path, map_location='cuda')
        
        prefix = "modelStudent.backbone.gcBlock."
        pretrained_gc_state_dict = {
            k[len(prefix):]: v
            for k, v in full_state_dict['model'].items()
            if k.startswith(prefix)
        }
        return pretrained_gc_state_dict

    def DT_pseudo_label_process(self, label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict):

        
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

        
        use_DT_labels = False
        if self.use_dino_PL:
            use_DT_labels = True
            if self.PL_swap == 'full':
                if self.iter > self.PL_swap_iter:
                    use_DT_labels = False
            elif self.PL_swap == 'half':
                if self.iter > self.PL_swap_iter and self.iter % 2:
                    use_DT_labels = False

        if use_DT_labels:
            
            instances = [self.dino_pseudogt[x['image_id']]['instances_dino'] for x in unlabel_data_q]
            boxes = [(x['tf_data'].apply_box(y.pred_boxes),y.scores,y.pred_classes) for x,y in zip(unlabel_data_q,instances)]
            dino_pseudo_labels = []
            for i in range(len(instances)):
                new_instances = Instances(unlabel_data_k[i]['image'].shape[-2:])
                new_instances.gt_boxes = Boxes(boxes[i][0])
                new_instances.gt_scores = boxes[i][1]
                new_instances.gt_classes = boxes[i][2]
                dino_pseudo_labels.append(new_instances)
            
            joint_proposal_dict = {}
            pseudo_proposals_dino, num_pseudo_bbox_roih = self.process_pseudo_label(
                dino_pseudo_labels, cur_threshold, "dino", "thresholding")
            joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_dino

        else:
            
            self.branch = "unsup_data_weak"
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            
            joint_proposal_dict = {}

            
            
            
            
            
            

            
            pseudo_proposals_roih_unsup_k, num_pseudo_bbox_roih = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")
            joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k

        
        all_label_data = label_data_q + label_data_k
        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
        if use_DT_labels:
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])
            all_unlabel_data = unlabel_data_q + unlabel_data_k
        else:
            all_unlabel_data = unlabel_data_q

        
        self.branch = "supervised"
        record_all_label_data, _, _, _, images_all_label, gt_instances_all_label = self.model(
            all_label_data, branch="supervised")
        record_dict.update(record_all_label_data)

        
        self.branch = "supervised_target"
        record_all_unlabel_data, _, _, _, images_all_unlabel, gt_instances_all_unlabel = self.model(
            all_unlabel_data, branch="supervised_target")

        new_record_all_unlabel_data = {}
        if(self.cfg.SEMISUPNET.PSEUDO_KD == True):
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = \
                    record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)
        else:
            pass

        
        if use_DT_labels:
            has_target_backbone_feats = 2
        else:
            has_target_backbone_feats = 1

        if self.use_adversarial_invariance:
            
            
            for i_index in range(len(unlabel_data_k)):
                
                for k, v in unlabel_data_k[i_index].items():
                    
                    label_data_k[i_index][k + "_unlabeled"] = v
                

            all_domain_data = label_data_k
            
            self.branch = "domain"
            record_all_domain_data, _, _, _ = self.model(
                all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)
        
        return has_target_backbone_feats

    
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "Model in eval mode while training, set it to train mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        
        
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        
        
        if self.iter % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            self._update_teacher_model(
                keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)


        
        
        
        
        

        
        record_dict = {}
        
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q + unlabel_data_k
            self.branch = "supervised"
            if(self.cfg.MODEL.META_ARCHITECTURE == 'DAobjTwoStagePseudoLabGeneralizedRCNN_GC_shortcut'):
                record_dict, _, _, _ = self.model(
                    all_label_data, branch="supervised")
            elif(self.cfg.MODEL.META_ARCHITECTURE == 'DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut'):
                record_dict, _, _, _, _, _ = self.model(
                    all_label_data, branch="supervised")
            else:
                raise NotImplementedError()

            has_target_backbone_feats = 0                

        else:
            has_target_backbone_feats = self.DT_pseudo_label_process(label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict)
        
        
        if self.use_feature_align:
            
            if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                
                with torch.no_grad():
                    
                    easy_feat = self.model.align_teacher(label_data_k)
                    teacher_feat = easy_feat.repeat(2,1,1,1)
            else:
                with torch.no_grad():
                    teacher_feat = self.model.align_teacher(all_label_data)

            if(self.cfg.SEMISUPNET.T_ADAPT == True):
                teacher_feat = self.model.t_adapt_layer(teacher_feat)

            if(self.cfg.SEMISUPNET.ALIGN_METHOD == "PIXEL"):
                
                
                student_feat = self.model.align_student_head(
                    self.student_align_feat['supervised'], teacher_feat.shape[2:])
                align_loss = self.model.align_student_head.align_loss(student_feat, teacher_feat)

            elif(self.cfg.SEMISUPNET.ALIGN_METHOD == "Sinkhorn"):
                align_loss = self.wasserstein_sinkhorn_loss(self.student_align_feat['supervised'], teacher_feat)

            else:
                raise NotImplementedError()
            record_dict['loss_align'] = align_loss
            
            
            
            
            
            if(self.cfg.SEMISUPNET.CROSS_ALIGN == True):
                
                if(self.cfg.MODEL.BACKBONE.NAME == "build_vgg_backbone"):
                    self.branch = "supervised"
                    
                    max_pool = nn.AdaptiveAvgPool2d((self.student_align_feat['supervised'].shape[-2], self.student_align_feat['supervised'].shape[-1]))
                    t_f_ups_source_k = max_pool(teacher_feat[:teacher_feat.shape[0]//2])
                    
                    rpn_k_loss_dict, roi_head_k_loss_dict = self.get_cross_align_loss(label_data_k, {'vgg4':t_f_ups_source_k}, branch="supervised")
                    for key_ in rpn_k_loss_dict.keys():
                        record_dict[key_ + "_CA"] = rpn_k_loss_dict[key_]
                    for key_ in roi_head_k_loss_dict.keys():
                        record_dict[key_ + "_CA"] = roi_head_k_loss_dict[key_]
                else:
                    raise NotImplementedError()
            

            if self.iter >= self.cfg.SEMISUPNET.FEATURE_ALIGN_TARGET_START:
                
                
                

                with torch.no_grad():
                    
                    if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                        
                        easy_feat_target = self.model.align_teacher(unlabel_data_k)
                        if has_target_backbone_feats == 0 or has_target_backbone_feats == 2:
                            teacher_feat_target = easy_feat_target.repeat(2,1,1,1)
                        else:
                            teacher_feat_target = easy_feat_target
                    else:
                        teacher_feat_target = self.model.align_teacher(all_unlabel_data)

                if(self.cfg.SEMISUPNET.T_ADAPT == True):
                    teacher_feat = self.model.t_adapt_layer(teacher_feat)

                if(self.cfg.SEMISUPNET.ALIGN_METHOD == "PIXEL"):
                    if has_target_backbone_feats == 0:
                        all_unlabel_data = unlabel_data_q + unlabel_data_k
                        backbone_feat_target = self.model.forward_backbone(all_unlabel_data)[self.cfg.SEMISUPNET.FEATURE_ALIGN_LAYER] 
                        student_feat_target = self.model.align_student_head(
                            backbone_feat_target, teacher_feat_target.shape[2:])
                    else:
                        
                        student_feat_target = self.model.align_student_head(
                            self.student_align_feat['supervised_target'], teacher_feat_target.shape[2:])
                    align_loss_target = self.model.align_student_head.align_loss(
                        student_feat_target, teacher_feat_target)

                elif(self.cfg.SEMISUPNET.ALIGN_METHOD == "Sinkhorn"):
                    if has_target_backbone_feats == 0:
                        all_unlabel_data = unlabel_data_q + unlabel_data_k
                        student_feat_target = self.model.forward_backbone(all_unlabel_data)[self.cfg.SEMISUPNET.FEATURE_ALIGN_LAYER]
                    else:
                        student_feat_target = self.student_align_feat['supervised_target']
                    align_loss_target = self.wasserstein_sinkhorn_loss(student_feat_target, teacher_feat_target)
                else:
                    raise NotImplementedError()
                
                
                
                
                
                
                
                
                
                record_dict['loss_align_target'] = align_loss_target
                
                
                if(self.cfg.SEMISUPNET.CROSS_ALIGN == True):
                    
                    if(self.cfg.MODEL.BACKBONE.NAME == "build_vgg_backbone"):
                        self.branch = "supervised_target"
                        
                        max_pool = nn.AdaptiveAvgPool2d((self.student_align_feat['supervised_target'].shape[-2], self.student_align_feat['supervised_target'].shape[-1]))
                        t_f_ups_target_k = max_pool(teacher_feat_target[:teacher_feat_target.shape[0]//2])
                        
                        rpn_unlabel_k_loss_dict, roi_head_unlabel_k_loss_dict = self.get_cross_align_loss(unlabel_data_k, {'vgg4':t_f_ups_target_k}, branch="supervised_target")
                        for key_ in rpn_unlabel_k_loss_dict.keys():
                            record_dict[key_ + "_CA_pseudo"] = rpn_unlabel_k_loss_dict[key_]
                        for key_ in roi_head_unlabel_k_loss_dict.keys():
                            record_dict[key_ + "_CA_pseudo"] = roi_head_unlabel_k_loss_dict[key_]
                    else:
                        raise NotImplementedError()  
                
                
        
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo" \
                    or key == "loss_rpn_loc_CA_pseudo" or key == "loss_box_reg_CA_pseudo":
                    
                    loss_dict[key] = record_dict[key] * 0
                elif key[-6:] == "pseudo":  
                    loss_dict[key] = (
                        record_dict[key] *
                        self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    )
                elif (
                    key == "loss_D_img_s" or key == "loss_D_img_t"
                ):  
                    
                    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif key == "loss_align":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT
                elif key == "loss_align_target":    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET
                
                elif key[-2:] == "CA":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CROSS_ALING_WEIGHT
                elif key[-10:] == "_CA_target":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CROSS_ALING_WEIGHT_TARGET
                    
                else:  
                    loss_dict[key] = record_dict[key] * 1

        losses = sum(loss_dict.values())

        
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def cosine_cost(x, y, **kwargs):
        cosine_sim = torch.bmm(x, y.transpose(1, 2))
        return 1 - cosine_sim  

    def wasserstein_sinkhorn_loss(self, student_feat, teacher_feat, normalize=True):
        
        

        batch_size, C, H_s, W_s = student_feat.shape
        
        

        
        teacher_points = teacher_feat.view(batch_size, C, -1).permute(0, 2, 1) 
        student_points = student_feat.view(batch_size, C, -1).permute(0, 2, 1) 

        if(normalize == True):
            criterion = CosineSinkhornKDLoss(epsilon=0.05, sinkhorn_iters=50)
            loss = criterion(teacher_points, student_points)

        else:
            sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

            
            total_loss = 0
            for i in range(batch_size):
                
                total_loss += sinkhorn_loss(teacher_points[i], student_points[i])

            loss = total_loss / batch_size
        
        return loss

    def get_cross_align_loss(self, data, features, branch):
        imgs_label_k = self.model.preprocess_image(data)
        if "instances" in data[0]:
            gt_instances_label = [x["instances"].to(self.model.device) for x in data]
        else:
            gt_instances_label = None
            
        proposals_rpn_label, proposal_losses_label = self.model.proposal_generator(imgs_label_k, features, gt_instances_label)
        _, detector_losses = self.model.roi_heads(
            imgs_label_k,
            features,
            proposals_rpn_label,
            compute_loss=True,
            targets=gt_instances_label,
            branch=branch,
            )
        return proposal_losses_label, detector_losses
    
    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        
        
        
        all_metrics_dict = comm.gather(metrics_dict)
        

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                
                
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                
                cfg.TEST.EVAL_PERIOD,
                self.model,
                
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        
        
        
        
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        
        
        
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _get_detector_input_hook(self, module, input, output):
        self.student_align_feat[self.branch] = input[1][self.cfg.MODEL.RPN.IN_FEATURES[0]]


    def _register_input_hook_feat_align(self, model, target_layer):
        for (name, module) in model.named_modules():
            if name == target_layer:
                module.register_forward_hook(self._get_detector_input_hook)
        return True



class DINOTeacherTrainer_lt(DefaultTrainer):
    def __init__(self, cfg, wandb_run=None):





        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        
        self.use_adversarial_invariance = cfg.SEMISUPNET.DIS_LOSS_WEIGHT > 0
        self.branch = 'supervised'
        if cfg.SEMISUPNET.USE_FEATURE_ALIGN:
            
            self.align_layer = cfg.SEMISUPNET.FEATURE_ALIGN_LAYER
            self.align_layer_weak = cfg.SEMISUPNET.FEATURE_ALIGN_LAYER_WEAK
            self.align_teacher_pool = cfg.SEMISUPNET.POOL
            if self.align_teacher_pool == "None":
                self.align_teacher_pool = None
            if self.align_teacher_pool not in (None, "max_pool", "mean", "mean_pool"):
                raise NotImplementedError("{} pooling not supported.".format(self.align_teacher_pool))
            self.use_feature_align = True
            self.student_align_feat = {}
            student_align_dim = model.backbone._out_feature_channels[cfg.SEMISUPNET.FEATURE_ALIGN_LAYER]
            student_align_weak_dim = model.backbone._out_feature_channels[cfg.SEMISUPNET.FEATURE_ALIGN_LAYER_WEAK]
            model.align_teacher = DinoVitFeatureExtractor_14_16(cfg, 
                                model_name=cfg.SEMISUPNET.ALIGN_MODEL, 
                                normalize_feature=cfg.SEMISUPNET.ALIGN_HEAD_NORMALIZE).eval()

            
            
            

            
            teacher_align_dim = model.align_teacher.embed_dim
            model.align_student_head = TeacherStudentAlignHead_lt(cfg, 
                                    student_align_dim, teacher_align_dim,
                                    align_loss_type=cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_TYPE,
                                    normalize_feature=model.align_teacher.normalize_feature)
            model.align_student_head_weak = TeacherStudentAlignHead_lt(cfg, 
                                    student_align_weak_dim, teacher_align_dim,
                                    align_loss_type=cfg.SEMISUPNET.FEATURE_ALIGN_WEAK_LOSS_TYPE,
                                    normalize_feature=model.align_teacher.normalize_feature)
            
            
            
            self._register_input_hook_feat_align(model, 'proposal_generator')

            model.align_teacher = model.align_teacher.to((torch.device(model.device)))
            model.align_student_head = model.align_student_head.to((torch.device(model.device)))   
            model.align_student_head_weak = model.align_student_head_weak.to((torch.device(model.device)))   
            
            
        else:
            self.use_feature_align = False


        if type(cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT) == str:
            file_in = cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT
            self.use_dino_PL = True
            with open(file_in, 'rb') as f_in:
                temp_dict = pickle.load(f_in)
            self.dino_pseudogt = {}
            for img in temp_dict:
                self.dino_pseudogt[img['image_id']] = img
        else:
            self.use_dino_PL = False

        if type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP) == str:
            assert type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER) == int
            self.PL_swap = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP
            self.PL_swap_iter = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER
        else:
            self.PL_swap = None

        
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            if self.use_feature_align:
                model.align_teacher = model.module.align_teacher
                model.align_student_head = model.module.align_student_head
                model.align_student_head_weak = model.module.align_student_head_weak
                model.forward_backbone = model.module.forward_backbone


        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        

        
        ensem_ts_model = EnsembleTSModel(model_teacher, model)


        
        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

        self.logger = logging.getLogger(__name__)

        
        
        
        
        
        
        
        
        

    def _pool_align_teacher_feat(self, feat):
        if self.align_teacher_pool is None:
            return feat
        if self.align_teacher_pool == "max_pool":
            return F.max_pool2d(feat, kernel_size=2, stride=2)
        if self.align_teacher_pool in ("mean", "mean_pool"):
            return F.avg_pool2d(feat, kernel_size=2, stride=2)
        raise NotImplementedError("{} pooling not supported.".format(self.align_teacher_pool))
    
    @classmethod
    def build_optimizer(cls, cfg, model):







        if(cfg.MODEL.BACKBONE.NAME == 'build_dino_vit_backbone_14_16'):
            return build_optimizer_with_layer_decay(model, cfg)
        elif( (cfg.MODEL.BACKBONE.NAME == 'build_vgg_backbone') or (cfg.MODEL.BACKBONE.NAME == 'build_vgg_lc_backbone') \
            or (cfg.MODEL.BACKBONE.NAME == 'build_resnet50_backbone')):
            return build_optimizer(cfg, model)
        else:
            raise NotImplementedError()

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):




        if(cfg.MODEL.BACKBONE.NAME == 'build_dino_vit_backbone_14_16'):
            return build_lr_scheduler_v2(cfg, optimizer)
        elif( (cfg.MODEL.BACKBONE.NAME == 'build_vgg_backbone') or (cfg.MODEL.BACKBONE.NAME == 'build_vgg_lc_backbone') or \
            (cfg.MODEL.BACKBONE.NAME == 'build_resnet50_backbone')):
            return build_adapteacher_lr_scheduler(cfg, optimizer)
        else:
            raise NotImplementedError()


    def resume_or_load(self, resume=True):











        print('\n\n\n\n',self.cfg.MODEL.WEIGHTS,'\n\n\n\n')
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            
            
        if isinstance(self.model, DistributedDataParallel):
            
            
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionPerClassEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparateKeepTf_lt(cfg, is_train=True, keep_tf_data=True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                
                self.freeze_model()
                
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
    
    def freeze_model(self):
        if(self.cfg.MODEL.BACKBONE.NAME not in ['build_vgg_backbone', 'build_vgg_lc_backbone', 'build_resnet50_backbone']):
            freeze = self.cfg.SEMISUPNET.FREEZE
            
            if self.cfg.SEMISUPNET.DINO_TUNE_LAYER:
                dino_tune_layers = [f"blocks.{i}." for i in self.cfg.SEMISUPNET.DINO_TUNE_LAYER]
            else:
                dino_tune_layers = None
            
            if freeze == 'freeze_DINO':
                for key, value in self.model.backbone.encoder.named_parameters():
                    if (key.split('.')[0] in ['blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed', 'pos_embed']):
                        value.requires_grad = False
                        
                        
                        
                    if ((dino_tune_layers is not None) and (any(layer in key for layer in dino_tune_layers))):
                        value.requires_grad = True
                        
                        
                        
                
                for key, value in self.model.backbone.encoder.named_modules():
                    if (key.split('.')[0] in ['blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed', 'pos_embed']):
                        value.eval()
                    
                    if ((dino_tune_layers is not None) and (any(layer in key for layer in dino_tune_layers))):
                        
                        value.train()        
                        
                        
            elif freeze == 'freeze_all':
                for param in self.model.backbone.encoder.parameters():
                    param.requires_grad = False
                self.model.backbone.encoder.eval()
                
            elif freeze == 'No':
                pass
            
            else:
                raise NotImplementedError()
            

        
        
        if self.use_feature_align:
            self.model.align_teacher.eval()
            self.model.align_teacher.requires_grad_(False)
    
    
    def run_step_full_semisup(self):
        
        
        self._trainer.iter = self.iter
        assert self.model.training, "Model in eval mode while training, set it to train mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        
        
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        use_fg_align = self.cfg.SEMISUPNET.FG_ALIGN
        if use_fg_align:
            label_fg_specs_q = self._collect_foreground_specs(label_data_q)
            label_fg_specs_k = self._collect_foreground_specs(label_data_k)
            unlabel_fg_specs_q = self._collect_foreground_specs(unlabel_data_q)
            unlabel_fg_specs_k = self._collect_foreground_specs(unlabel_data_k)
        
        

        
        if self.cfg.SEMISUPNET.GRAD_ACCUM == False:
            if self.iter % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
        else:
            if self.iter % (self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER * self.cfg.SEMISUPNET.ACCUM_NUM) == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        
        
        
        

        record_dict = {}
        

        
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            
            
            
            
            
            if self.cfg.SEMISUPNET.WS_RANDOM:
                p = self.cfg.SEMISUPNET.WS_PROB  
                if random.random() < p:
                    all_label_data = label_data_q
                    all_unlabel_data = unlabel_data_q
                else:
                    all_label_data = label_data_k
                    all_unlabel_data = unlabel_data_k
                    
                    
            else:
                all_label_data = label_data_q + label_data_k
                all_unlabel_data = unlabel_data_q + unlabel_data_k
            
            self.branch = "supervised"
                
            
            
            record_dict, _, _, _, _, _ = self.model(
                all_label_data, branch="supervised")

            has_target_backbone_feats = 0
            if use_fg_align:
                all_label_fg_specs = self._collect_foreground_specs(all_label_data)
                all_unlabel_fg_specs = self._collect_foreground_specs(all_unlabel_data)

        else:
            has_target_backbone_feats = self.DT_pseudo_label_process(
                label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict
            )
            all_label_data = label_data_q + label_data_k
            if has_target_backbone_feats == 2:
                all_unlabel_data = unlabel_data_q + unlabel_data_k
            else:
                all_unlabel_data = unlabel_data_q
            if use_fg_align:
                all_label_fg_specs = label_fg_specs_q + label_fg_specs_k
                if has_target_backbone_feats == 2:
                    all_unlabel_fg_specs = unlabel_fg_specs_q + unlabel_fg_specs_k
                else:
                    all_unlabel_fg_specs = unlabel_fg_specs_q
        
        if(self.cfg.SEMISUPNET.SPM_GRL == True):
                f_s = self.model.forward_spm(label_data_k)
                f_t = self.model.forward_spm(unlabel_data_k)
                
                source_label = 0
                target_label = 1
                
                adv_f_s = grad_reverse(f_s)
                D_img_out_s = self.model.backbone.D_img(adv_f_s)
                loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, 
                                torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.model.device))
                
                adv_f_t = grad_reverse(f_t)
                D_img_out_t = self.model.backbone.D_img(adv_f_t)
                loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, 
                                torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.model.device))
                
                
                
                
                record_dict['loss_D_spm_s'] = loss_D_img_s
                record_dict['loss_D_spm_t'] = loss_D_img_t
                
        
        if self.use_feature_align:
            
            if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                
                
                
                easy_feat = self._pool_align_teacher_feat(self.model.align_teacher(label_data_k))
                teacher_feat = easy_feat.repeat(2,1,1,1)
            else:
                teacher_feat = self._pool_align_teacher_feat(self.model.align_teacher(all_label_data))
            
            student_feat = self.model.align_student_head(
                self.student_align_feat['supervised'][self.align_layer], teacher_feat.shape[2:])
            if use_fg_align:
                source_fg_mask = self._build_foreground_mask(
                    all_label_fg_specs, teacher_feat.shape[2:], teacher_feat.device, teacher_feat.dtype
                )
                
                
                align_loss = self._foreground_align_loss(
                    self.model.align_student_head, student_feat, teacher_feat, source_fg_mask
                )
            else:
                align_loss = self.model.align_student_head.align_loss(student_feat, teacher_feat)
            
            if(self.cfg.SEMISUPNET.WEAK_LAYER_ALIGN == True):
                teacher_feat_down = self.model.align_student_head_weak.avg_pool(teacher_feat)
                
                student_feat_deep = self.model.align_student_head_weak(
                    self.student_align_feat['supervised'][self.align_layer_weak], teacher_feat_down.shape[2:])
                if use_fg_align:
                    source_fg_mask_down = self._build_foreground_mask(
                        all_label_fg_specs, teacher_feat_down.shape[2:], teacher_feat_down.device, teacher_feat_down.dtype
                    )
                    weak_align_loss = self._foreground_align_loss(
                        self.model.align_student_head_weak, student_feat_deep, teacher_feat_down, source_fg_mask_down
                    )
                else:
                    weak_align_loss = self.model.align_student_head_weak.align_loss(
                        student_feat_deep, teacher_feat_down
                    )
                
                record_dict['loss_align_weak'] = weak_align_loss
                
                
            record_dict['loss_align'] = align_loss

            if self.iter >= self.cfg.SEMISUPNET.FEATURE_ALIGN_TARGET_START:
                
                
                

                if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                    
                    easy_feat_target = self._pool_align_teacher_feat(self.model.align_teacher(unlabel_data_k))
                    if has_target_backbone_feats == 0 or has_target_backbone_feats == 2:
                        teacher_feat_target = easy_feat_target.repeat(2,1,1,1)
                    else:
                        teacher_feat_target = easy_feat_target
                else:
                    teacher_feat_target = self._pool_align_teacher_feat(self.model.align_teacher(all_unlabel_data))

                
                
                if use_fg_align:
                    target_fg_specs = all_unlabel_fg_specs
                    if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                        target_fg_specs = unlabel_fg_specs_k
                        if has_target_backbone_feats == 0 or has_target_backbone_feats == 2:
                            target_fg_specs = target_fg_specs * 2
                    target_fg_mask = self._build_foreground_mask(
                        target_fg_specs, teacher_feat_target.shape[2:], teacher_feat_target.device, teacher_feat_target.dtype
                    )
                
                if(self.cfg.SEMISUPNET.WEAK_LAYER_ALIGN == True):
                    teacher_feat_target_d = self.model.align_student_head_weak.avg_pool(teacher_feat_target)
                    if use_fg_align:
                        target_fg_mask_down = self._build_foreground_mask(
                            target_fg_specs,
                            teacher_feat_target_d.shape[2:],
                            teacher_feat_target_d.device,
                            teacher_feat_target_d.dtype,
                        )
                
                if has_target_backbone_feats == 0:
                    all_unlabel_data = unlabel_data_q + unlabel_data_k
                    
                    backbone_feats = self.model.forward_backbone(all_unlabel_data)
                    backbone_feat_target = backbone_feats[self.align_layer] 
                    
                    student_feat_target = self.model.align_student_head(
                        backbone_feat_target, teacher_feat_target.shape[2:])
                    
                    
                    if(self.cfg.SEMISUPNET.WEAK_LAYER_ALIGN == True):
                        backbone_feat_weak_target = backbone_feats[self.align_layer_weak] 
                        
                        student_feat_weak_target = self.model.align_student_head_weak(
                        backbone_feat_weak_target, teacher_feat_target_d.shape[2:])
                        
                else:
                    student_feat_target = self.model.align_student_head(
                        self.student_align_feat['supervised_target'][self.align_layer], teacher_feat_target.shape[2:])
                    
                    if(self.cfg.SEMISUPNET.WEAK_LAYER_ALIGN == True):
                        student_feat_weak_target = self.model.align_student_head_weak(
                        self.student_align_feat['supervised_target'][self.align_layer_weak], teacher_feat_target_d.shape[2:])
                        
    
                if use_fg_align:
                    align_loss_target = self._foreground_align_loss(
                        self.model.align_student_head, student_feat_target, teacher_feat_target, target_fg_mask
                    )
                else:
                    align_loss_target = self.model.align_student_head.align_loss(
                        student_feat_target, teacher_feat_target
                    )
                
                if(self.cfg.SEMISUPNET.WEAK_LAYER_ALIGN == True):
                    if use_fg_align:
                        align_loss_weak_target = self._foreground_align_loss(
                            self.model.align_student_head_weak,
                            student_feat_weak_target,
                            teacher_feat_target_d,
                            target_fg_mask_down,
                        )
                    else:
                        align_loss_weak_target = self.model.align_student_head_weak.align_loss(
                            student_feat_weak_target, teacher_feat_target_d
                        )
                    
                    record_dict['loss_align_weak_target'] = align_loss_weak_target
                
                record_dict['loss_align_target'] = align_loss_target

            if(self.cfg.SEMISUPNET.TWO_STAGE_ALIGN_WD is not None) \
                and (self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP):
                
                
                record_dict["loss_align"] = record_dict["loss_align"] * self.cfg.SEMISUPNET.TWO_STAGE_ALIGN_WD
                if "loss_align_target" in record_dict.keys():
                    record_dict["loss_align_target"] = record_dict["loss_align_target"] * self.cfg.SEMISUPNET.TWO_STAGE_ALIGN_WD

        
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                    
                    loss_dict[key] = record_dict[key] * 0
                elif key[-6:] == "pseudo":  
                    loss_dict[key] = (
                        record_dict[key] *
                        self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    )
                elif (
                    key == "loss_D_img_s" or key == "loss_D_img_t"
                ):  
                    
                    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif (key == "loss_align"):
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT
                elif (key == "loss_align_weak"):
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_WEAK_LOSS_WEIGHT
                elif (key == "loss_align_target"):    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET
                elif (key == "loss_align_weak_target"):
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_WEAK_LOSS_WEIGHT_TARGET
                elif ((key == "loss_D_spm_s") or (key == "loss_D_spm_t")):
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.ADV_SPM_WEIGHT
                else:  
                    loss_dict[key] = record_dict[key] * 1

        if (self.cfg.SEMISUPNET.GRAD_ACCUM == False):
            losses = sum(loss_dict.values())
        else:
            losses = sum(loss_dict.values()) / self.cfg.SEMISUPNET.ACCUM_NUM
            
            

        
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        if ( (self.cfg.SEMISUPNET.GRAD_ACCUM == True) \
            and ((self.iter + 1) % self.cfg.SEMISUPNET.ACCUM_NUM == 0) ):
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        elif proposal_type == "dino":
            valid_map = proposal_bbox_inst.gt_scores > thres
            new_proposal_inst = proposal_bbox_inst[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, pseudo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    def DT_pseudo_label_process(self, label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict):

        
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

        
        use_DT_labels = False
        if self.use_dino_PL:
            use_DT_labels = True
            if self.PL_swap == 'full':
                if self.iter > self.PL_swap_iter:
                    use_DT_labels = False
            elif self.PL_swap == 'half':
                if self.iter > self.PL_swap_iter and self.iter % 2:
                    use_DT_labels = False

        if use_DT_labels:
            
            instances = [self.dino_pseudogt[x['image_id']]['instances_dino'] for x in unlabel_data_q]
            boxes = [(x['tf_data'].apply_box(y.pred_boxes),y.scores,y.pred_classes) for x,y in zip(unlabel_data_q,instances)]
            dino_pseudo_labels = []
            for i in range(len(instances)):
                new_instances = Instances(unlabel_data_k[i]['image'].shape[-2:])
                new_instances.gt_boxes = Boxes(boxes[i][0])
                new_instances.gt_scores = boxes[i][1]
                new_instances.gt_classes = boxes[i][2]
                dino_pseudo_labels.append(new_instances)
            
            joint_proposal_dict = {}
            pseudo_proposals_dino, num_pseudo_bbox_roih = self.process_pseudo_label(
                dino_pseudo_labels, cur_threshold, "dino", "thresholding")
            joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_dino

        else:
            
            self.branch = "unsup_data_weak"
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            
            joint_proposal_dict = {}

            
            
            
            
            
            

            
            pseudo_proposals_roih_unsup_k, num_pseudo_bbox_roih = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")
            joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k

        
        all_label_data = label_data_q + label_data_k
        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
        if use_DT_labels:
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])
            all_unlabel_data = unlabel_data_q + unlabel_data_k
        else:
            all_unlabel_data = unlabel_data_q

        
        self.branch = "supervised"
        record_all_label_data, _, _, _, images_all_label, gt_instances_all_label = self.model(
            all_label_data, branch="supervised")
        record_dict.update(record_all_label_data)

        
        self.branch = "supervised_target"
        record_all_unlabel_data, _, _, _, images_all_unlabel, gt_instances_all_unlabel = self.model(
            all_unlabel_data, branch="supervised_target")

        new_record_all_unlabel_data = {}
        if(self.cfg.SEMISUPNET.PSEUDO_KD == True):
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = \
                    record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)
        else:
            pass

        
        if use_DT_labels:
            has_target_backbone_feats = 2
        else:
            has_target_backbone_feats = 1

        if self.use_adversarial_invariance:
            
            
            for i_index in range(len(unlabel_data_k)):
                
                for k, v in unlabel_data_k[i_index].items():
                    
                    label_data_k[i_index][k + "_unlabeled"] = v
                

            all_domain_data = label_data_k
            
            self.branch = "domain"
            record_all_domain_data, _, _, _ = self.model(
                all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)
        
        return has_target_backbone_feats

    def _collect_foreground_specs(self, batched_inputs):
        fg_specs = []
        for data in batched_inputs:
            image_size = tuple(data["image"].shape[-2:])
            boxes = None
            if "instances" in data and data["instances"].has("gt_boxes"):
                boxes = data["instances"].gt_boxes.tensor.detach().cpu()
                if hasattr(data["instances"], "image_size"):
                    image_size = tuple(data["instances"].image_size)
            elif "annotations" in data:
                ann_boxes = []
                for anno in data["annotations"]:
                    if anno.get("iscrowd", 0) != 0 or "bbox" not in anno:
                        continue
                    bbox_mode = anno.get("bbox_mode", BoxMode.XYXY_ABS)
                    ann_boxes.append(BoxMode.convert(anno["bbox"], bbox_mode, BoxMode.XYXY_ABS))
                if len(ann_boxes) > 0:
                    boxes = torch.as_tensor(ann_boxes, dtype=torch.float32)
            if boxes is None:
                boxes = torch.empty((0, 4), dtype=torch.float32)
            fg_specs.append({"boxes": boxes, "image_size": image_size})
        return fg_specs

    def _match_foreground_specs_to_batch(self, fg_specs, batch_size):
        if len(fg_specs) == batch_size:
            return fg_specs
        if len(fg_specs) == 0:
            return [{"boxes": torch.empty((0, 4), dtype=torch.float32), "image_size": (1, 1)}
                    for _ in range(batch_size)]
        repeats = (batch_size + len(fg_specs) - 1) // len(fg_specs)
        return (fg_specs * repeats)[:batch_size]

    def _build_foreground_mask(self, fg_specs, feature_size, device, dtype):
        feat_h, feat_w = feature_size
        fg_specs = self._match_foreground_specs_to_batch(fg_specs, len(fg_specs))
        mask = torch.zeros((len(fg_specs), 1, feat_h, feat_w), device=device, dtype=dtype)
        for idx, spec in enumerate(fg_specs):
            boxes = spec["boxes"].to(device=device, dtype=torch.float32)
            if boxes.numel() == 0:
                continue
            img_h, img_w = spec["image_size"]
            scale_x = float(feat_w) / max(float(img_w), 1.0)
            scale_y = float(feat_h) / max(float(img_h), 1.0)
            boxes = boxes.clone()
            boxes[:, 0::2] = boxes[:, 0::2] * scale_x
            boxes[:, 1::2] = boxes[:, 1::2] * scale_y
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, feat_w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, feat_h)
            for box in boxes:
                x1 = int(torch.floor(box[0]).item())
                y1 = int(torch.floor(box[1]).item())
                x2 = int(torch.ceil(box[2]).item())
                y2 = int(torch.ceil(box[3]).item())
                if x2 > x1 and y2 > y1:
                    mask[idx, :, y1:y2, x1:x2] = 1
        return mask

    def _foreground_align_loss(self, align_head, feat_student, feat_teacher, fg_mask):
        fg_mask = self._match_foreground_mask_to_batch(fg_mask, feat_student.shape[0])
        fg_pixels = fg_mask.sum()
        if fg_pixels <= 0:
            return (feat_student - feat_teacher).sum() * 0

        if align_head.align_loss_type == "L2":
            loss = ((feat_student - feat_teacher) ** 2 * fg_mask).sum()
            return loss / (fg_pixels * feat_student.shape[1])
        elif align_head.align_loss_type == "L1":
            loss = (torch.abs(feat_student - feat_teacher) * fg_mask).sum()
            return loss / (fg_pixels * feat_student.shape[1])
        elif align_head.align_loss_type == "Cosine":
            loss = 1 - F.cosine_similarity(feat_student, feat_teacher, dim=1, eps=1e-8)
            return (loss * fg_mask.squeeze(1)).sum() / fg_pixels
        else:
            raise NotImplementedError()

    def _match_foreground_mask_to_batch(self, fg_mask, batch_size):
        if fg_mask.shape[0] == batch_size:
            return fg_mask
        if fg_mask.shape[0] == 0:
            return torch.zeros(
                (batch_size, 1, fg_mask.shape[-2], fg_mask.shape[-1]),
                device=fg_mask.device,
                dtype=fg_mask.dtype,
            )
        repeats = (batch_size + fg_mask.shape[0] - 1) // fg_mask.shape[0]
        return fg_mask.repeat(repeats, 1, 1, 1)[:batch_size]

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        
        
        
        all_metrics_dict = comm.gather(metrics_dict)
        

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                
                
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        
        
        
        if not getattr(self, "_teacher_ema_initialized", False):
            self._copy_main_model()
            self._teacher_ema_initialized = True
            return

        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict(), strict=False)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                
                cfg.TEST.EVAL_PERIOD,
                self.model,
                
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        
        
        
        
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        
        
        
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _get_detector_input_hook(self, module, input, output):
        
        self.student_align_feat[self.branch] = input[1]
        
        
        

    def _register_input_hook_feat_align(self, model, target_layer):
        for (name, module) in model.named_modules():
            if name == target_layer:
                module.register_forward_hook(self._get_detector_input_hook)
        return True
