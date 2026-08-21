




import logging
import os
from typing import Any

import torch
from omegaconf import OmegaConf

import dinov3.distributed
from dinov3.eval import results

logger = logging.getLogger("dinov3")

CONFIG_FILE_KEY = "config_file"
EVAL_CONFIG_FNAME = "eval_config.yaml"


def write_results(results_dict, output_dir, results_filename) -> None:

    if torch.cuda.is_available() and not dinov3.distributed.is_main_process():
        return
    results_path = os.path.join(output_dir, results_filename)
    logger.info(f"Saving results to {results_path}")
    results.save_from_dict(results_dict=results_dict, results_path=results_path)


def args_dict_to_dataclass(eval_args: dict[str, object], config_dataclass, save_config: bool = True) -> tuple[Any, str]:






    if CONFIG_FILE_KEY in eval_args:
        config_file = eval_args.pop(CONFIG_FILE_KEY)
        eval_args_config = OmegaConf.merge(OmegaConf.load(config_file), OmegaConf.create(eval_args))
    else:
        eval_args_config = OmegaConf.create(eval_args)

    structured_config = OmegaConf.merge(OmegaConf.structured(config_dataclass), eval_args_config)
    logger.info(f"Evaluation Configuration:\n{OmegaConf.to_yaml(structured_config)}")
    output_dir = structured_config.output_dir

    if save_config and dinov3.distributed.is_main_process():
        OmegaConf.save(config=structured_config, f=os.path.join(output_dir, EVAL_CONFIG_FNAME))

    return OmegaConf.to_object(structured_config), output_dir


def cli_parser(argv: list[str]) -> tuple[dict[str, Any]]:






    cli_eval_args_dict = OmegaConf.to_container(OmegaConf.from_cli(argv))
    if "output_dir" not in cli_eval_args_dict:
        cli_eval_args_dict["output_dir"] = cli_eval_args_dict.pop("--output-dir", ".")
    return cli_eval_args_dict
