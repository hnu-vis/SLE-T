
import argparse
import datetime
import logging
import os
import pickle
import time
from collections import OrderedDict, abc
from contextlib import ExitStack

import torch
from torch import nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, inference_context, print_csv_format
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes
from detectron2.utils.logger import log_every_n_seconds

from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer
from adapteacher.modeling.meta_arch.rcnn import (  
    DAobjTwoStagePseudoLabGeneralizedRCNN,
    TwoStagePseudoLabGeneralizedRCNN,
)
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN  
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab  
from dinoteacher import add_dinoteacher_config
from dinoteacher.engine.trainer import DINOTeacherTrainer, DINOTeacherTrainer_lt
from dinoteacher.modeling.meta_arch.dino_vit import (  
    build_dino_vit_adapter_backbone,
    build_dino_vit_backbone,
)
from dinoteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut  
from dinoteacher.modeling.roi_heads.roi_heads import SingleScaleROIHeadsPseudoLab  
from dinoteacher.solver.build_wd_norm_bias import add_custom_config

import dinoteacher.data.datasets.builtin  


logger = logging.getLogger("eval_ap_sml")


def setup(args):
    cfg = get_cfg()
    add_dinoteacher_config(cfg)
    if args.stage == "train_dino":
        add_custom_config(cfg)
    elif args.stage != "align_or_pseudo":
        raise NotImplementedError("Unknown stage: {}".format(args.stage))

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.dataset_name:
        cfg.DATASETS.TEST = tuple(args.dataset_name)
    cfg.freeze()

    default_setup(cfg, args)
    return cfg


def get_trainer_class(cfg):
    if cfg.SEMISUPNET.Trainer == "dinoteacher":
        return DINOTeacherTrainer_lt
    if cfg.SEMISUPNET.Trainer == "adapteacher":
        return ATeacherTrainer
    if cfg.SEMISUPNET.Trainer == "baseline":
        return BaselineTrainer
    raise ValueError("Trainer Name is not found: {}".format(cfg.SEMISUPNET.Trainer))


def build_eval_model(cfg, trainer_cls, model_role, direct_model=False):
    if direct_model:
        model = trainer_cls.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        return model

    if cfg.SEMISUPNET.Trainer in ("dinoteacher", "adapteacher"):
        model = trainer_cls.build_model(cfg)
        model_teacher = trainer_cls.build_model(cfg)
        ensemble = EnsembleTSModel(model_teacher, model)
        DetectionCheckpointer(ensemble, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        if model_role == "teacher":
            return ensemble.modelTeacher
        if model_role == "student":
            return ensemble.modelStudent
        if model_role == "raw":
            return model
        raise ValueError("Unknown model role: {}".format(model_role))

    model = trainer_cls.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    return model


def build_coco_evaluator(dataset_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    
    return COCOEvaluator(dataset_name, tasks=("bbox",), output_dir=output_folder)


def prediction_file_for(cfg, dataset_name, explicit_path=None):
    if explicit_path:
        return explicit_path
    return os.path.join(cfg.OUTPUT_DIR, "predictions", dataset_name + "_predictions.pkl")


def simplify_instances(instances):
    instances = instances.to(torch.device("cpu"))
    return instances


def run_inference_with_optional_dump(model, data_loader, evaluator, save_predictions=False):
    num_devices = comm.get_world_size()
    total = len(data_loader)
    logger.info("Start inference on {} batches".format(total))

    if evaluator is None:
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    predictions_to_save = [] if save_predictions else None
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            if save_predictions:
                for input_per_image, output_per_image in zip(inputs, outputs):
                    if "instances" not in output_per_image:
                        continue
                    predictions_to_save.append(
                        {
                            "file_name": input_per_image.get("file_name"),
                            "image_id": input_per_image["image_id"],
                            "height": input_per_image.get("height"),
                            "width": input_per_image.get("width"),
                            "instances": simplify_instances(output_per_image["instances"]),
                        }
                    )

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        "Inference done {}/{}. Dataloading: {:.4f} s/iter. "
                        "Inference: {:.4f} s/iter. Eval: {:.4f} s/iter. "
                        "Total: {:.4f} s/iter. ETA={}"
                    ).format(
                        idx + 1,
                        total,
                        data_seconds_per_iter,
                        compute_seconds_per_iter,
                        eval_seconds_per_iter,
                        total_seconds_per_iter,
                        eta,
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    total_time = time.perf_counter() - start_time
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            str(datetime.timedelta(seconds=total_time)), total_time / max(total - num_warmup, 1), num_devices
        )
    )

    if save_predictions:
        all_predictions = comm.gather(predictions_to_save, dst=0)
        if comm.is_main_process():
            predictions_to_save = []
            for predictions_per_rank in all_predictions:
                predictions_to_save.extend(predictions_per_rank)
        else:
            predictions_to_save = None

    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results, predictions_to_save


def load_prediction_records(prediction_path):
    with open(prediction_path, "rb") as f:
        predictions = pickle.load(f)

    
    
    if isinstance(predictions, dict):
        raise ValueError("Unsupported prediction dict format in {}".format(prediction_path))
    return predictions


def get_instances_from_record(record):
    if "instances" in record:
        instances = record["instances"]
    elif "instances_dino" in record:
        instances = record["instances_dino"]
    else:
        raise KeyError("Prediction record has neither 'instances' nor 'instances_dino'.")

    if instances.has("pred_boxes") and torch.is_tensor(instances.pred_boxes):
        instances.pred_boxes = Boxes(instances.pred_boxes)
    return instances


def evaluate_saved_predictions(dataset_name, prediction_path, output_folder):
    evaluator = build_coco_evaluator(dataset_name, output_folder)
    evaluator.reset()
    predictions = load_prediction_records(prediction_path)

    for record in predictions:
        image_id = record["image_id"]
        input_per_image = {
            "image_id": image_id,
            "file_name": record.get("file_name", ""),
            "height": record.get("height"),
            "width": record.get("width"),
        }
        instances = get_instances_from_record(record)
        output_per_image = {"instances": instances}
        evaluator.process([input_per_image], [output_per_image])

    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results


def summarize_ap_sml(results):
    bbox = results.get("bbox", results)
    keys = ("APs", "APm", "APl")
    summary = OrderedDict()
    for key in keys:
        if key in bbox:
            summary[key] = bbox[key]
    return summary


def save_predictions(predictions, path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(predictions, f)
    logger.info("Predictions saved to {}".format(path))


def evaluate_dataset_from_model(cfg, trainer_cls, model, dataset_name, args):
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_ap_sml", dataset_name)
    data_loader = trainer_cls.build_test_loader(cfg, dataset_name)
    evaluator = build_coco_evaluator(dataset_name, output_folder)
    results, predictions = run_inference_with_optional_dump(
        model, data_loader, evaluator, save_predictions=args.save_predictions
    )
    if args.save_predictions and comm.is_main_process():
        out_path = prediction_file_for(cfg, dataset_name, args.prediction_output)
        save_predictions(predictions, out_path)
    return results


def main(args):
    cfg = setup(args)
    if args.prediction_output and len(cfg.DATASETS.TEST) != 1:
        raise ValueError("--prediction-output is only supported when evaluating one dataset.")

    if args.predictions:
        if len(cfg.DATASETS.TEST) != 1 and len(args.predictions) == 1:
            raise ValueError("Please use --dataset-name with one dataset when passing one prediction file.")
        if len(args.predictions) not in (1, len(cfg.DATASETS.TEST)):
            raise ValueError("--predictions should contain either 1 file or one file per test dataset.")

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            prediction_path = args.predictions[idx if len(args.predictions) > 1 else 0]
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_ap_sml", dataset_name)
            results_i = evaluate_saved_predictions(dataset_name, prediction_path, output_folder)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
                logger.info("AP_s/m/L summary for {}: {}".format(dataset_name, summarize_ap_sml(results_i)))
        return list(results.values())[0] if len(results) == 1 else results

    trainer_cls = DefaultTrainer if args.direct_model else get_trainer_class(cfg)
    model = build_eval_model(cfg, trainer_cls, args.model_role, direct_model=args.direct_model)
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        results_i = evaluate_dataset_from_model(cfg, trainer_cls, model, dataset_name, args)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            logger.info("AP_s/m/L summary for {}: {}".format(dataset_name, summarize_ap_sml(results_i)))
    return list(results.values())[0] if len(results) == 1 else results


def add_eval_args(parser):
    parser.add_argument("--stage", type=str, default="align_or_pseudo", choices=["align_or_pseudo", "train_dino"])
    parser.add_argument("--dataset-name", nargs="+", default=None, help="Override cfg.DATASETS.TEST.")
    parser.add_argument("--weights", default=None, help="Override cfg.MODEL.WEIGHTS.")
    parser.add_argument("--output-dir", default=None, help="Override cfg.OUTPUT_DIR.")
    parser.add_argument(
        "--model-role",
        default="teacher",
        choices=["teacher", "student", "raw"],
        help="For teacher/student trainers, choose which branch to evaluate.",
    )
    parser.add_argument(
        "--direct-model",
        action="store_true",
        help="Load cfg.MODEL.WEIGHTS directly into one model instead of a teacher/student ensemble.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-image Instances predictions as a pickle for later AP_s/m/L evaluation.",
    )
    parser.add_argument(
        "--prediction-output",
        default=None,
        help="Prediction pickle path used with --save-predictions. Only valid for one dataset.",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=None,
        help="Evaluate from saved prediction pickle(s) instead of running the model.",
    )
    return parser


if __name__ == "__main__":
    parser = add_eval_args(default_argument_parser())
    args = parser.parse_args()
    args.resume = False
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
