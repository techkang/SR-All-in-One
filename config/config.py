import argparse
import logging
from pathlib import Path

import torch as t

from config import CfgNode as CN

global_cfg = CN()


def default_argument_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--visible-gpus", type=str, default='')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg, args):
    Path(cfg.output_dir, 'logfile', cfg.model.name).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, 'checkpoint', cfg.model.name).mkdir(parents=True, exist_ok=True)
    filename = Path(cfg.output_dir, 'logfile', cfg.model.name, f'{cfg.model.name}_SR{cfg.upscale_factor}.log')
    logging.basicConfig(filename=str(filename), filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Command line arguments: ' + str(args))
    logging.info(f'Runing with full config: \n{cfg}')

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        t.backends.cudnn.benchmark = cfg.cudnn_benchmark


def get_cfg():
    from config.defaults import _C
    return _C.clone()


def set_global_cfg(cfg: CN):
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)
