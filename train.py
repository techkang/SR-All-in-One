import os

import tools.trainer
from config import get_cfg, default_argument_parser, default_setup


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    if args.visible_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
    cfg = setup(args)

    if not args.eval_only:
        getattr(tools.trainer, cfg.trainer)(cfg, resume=args.resume).train()
    else:
        trainer = getattr(tools.trainer, cfg.trainer)(cfg, resume=args.resume)
        trainer.get_baseline()
        trainer.test()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
