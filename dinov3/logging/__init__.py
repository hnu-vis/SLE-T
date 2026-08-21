




import functools
import logging
import os
import sys
from typing import Optional

from termcolor import colored

from dinov3.distributed import TorchDistributedEnvironment

from dinov3.logging.helpers import MetricLogger, SmoothedValue

_LEVEL_COLORED_KWARGS = {
    logging.DEBUG: {"color": "green", "attrs": ["bold"]},
    logging.INFO: {"color": "green"},
    logging.WARNING: {"color": "yellow"},
    logging.ERROR: {"color": "red"},
    logging.CRITICAL: {"color": "red", "attrs": ["bold"]},
}


class _LevelColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super().formatMessage(record)

        colored_kwargs = _LEVEL_COLORED_KWARGS.get(record.levelno)
        if colored_kwargs is None:
            return log

        msg = record.msg % record.args if record.msg == "%s" else record.msg
        index = log.rfind(msg, len(log) - len(msg))
        
        
        if index == -1:
            return log
        prefix = log[:index]
        prefix = colored(prefix, **colored_kwargs)
        return prefix + msg



@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
    color: bool = True,
    log_to_stdout_only_in_main_process: bool = True,
):



















    
    color = color and os.isatty(sys.stdout.fileno())

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    
    
    
    
    fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    plain_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    torch_env = TorchDistributedEnvironment()

    
    should_log_to_stdout = torch_env.is_main_process or not log_to_stdout_only_in_main_process
    if should_log_to_stdout:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter: logging.Formatter
        if color:
            formatter = _LevelColoredFormatter(
                fmt=fmt,
                datefmt=datefmt,
            )
        else:
            formatter = plain_formatter

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not torch_env.is_main_process:
            filename = filename + f".rank{torch_env.rank}"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(plain_formatter)
        logger.addHandler(handler)

    logger.debug(f"PyTorch distributed environment: {torch_env}")
    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    color: bool = True,
    capture_warnings: bool = True,
    log_to_stdout_only_in_main_process: bool = True,
) -> None:















    logging.captureWarnings(capture_warnings)
    
    output = output if output is None else os.path.realpath(output)
    _configure_logger(
        name,
        level=level,
        output=output,
        color=color,
        log_to_stdout_only_in_main_process=log_to_stdout_only_in_main_process,
    )


def cleanup_logging(*, name: Optional[str] = None) -> None:
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    
    _configure_logger.cache_clear()
