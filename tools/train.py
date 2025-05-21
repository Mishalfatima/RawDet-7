# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" 

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default= './configs/faster_rcnn/faster-rcnn_r50_fpn_1x_RawDet.py', help='train config file path')
    parser.add_argument('--work-dir', default='test', help='the dir to save logs and models')
    parser.add_argument('--data-root', default='./datasets/RAW-RGB-Dataset/', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type = bool,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--gpu',
        default=4,
        type=int,
        help='gpu_ids')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    ## Quantization parameters
    parser.add_argument(
        "--quant", default=4, type=int, help="Quantization Level, 0 means sRGB and no quantization, 12 means raw 12-bit input and no quantization"
    )
    parser.add_argument(
        "--log" ,action='store_true', help="Logarithmic or Linear Quantization"
    )
    parser.add_argument(
        "--log_learnable" , action='store_true', help="Logarthmic Quantization with learnable epsilon"
    )
    parser.add_argument(
        "--epsilon", default=1, type=float, help="epislon for log"
    )
    parser.add_argument(
        "--is_raw", action='store_true', help="whether to use raw data or not"
    )
    parser.add_argument(
        "--gamma_",action='store_true', help="whether to use gamma_correction or not"
    )
    parser.add_argument(
        "--data_type", default='NEW', help="currently supports pascalraw and NOD"
    )
    parser.add_argument(
        "--n_gamma", default=1, type=int, help="whether to use 1 gamma for the whole dataset or 2 for different time of the day"
    )
    parser.add_argument(
        "--use_WB", default=False, action='store_true', help="whether to use white balance correction or not"
    )
    parser.add_argument(
        "--mode_train", default=True, help="whether to train or not"
    )
    parser.add_argument(
        "--data",default='NEW',type=str,  help="PRAW, NIKON, SONY, RAOD, ZURICH, NEW"
    )

    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():

    args = parse_args()
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    if args.resume:
        args.resume = os.path.join(args.work_dir, 'latest.pth')

    if args.data == 'PRAW':
            ann_file = os.path.join(args.data_root,'coco/val_praw.json')
            ann_file_train = os.path.join(args.data_root,'coco/train_praw.json')
    elif args.data == 'NIKON':
            ann_file = os.path.join(args.data_root,'coco/val_nikon.json')
            ann_file_train = os.path.join(args.data_root,'coco/train_nikon.json')
    elif args.data == 'SONY':
            ann_file = os.path.join(args.data_root,'coco/val_sony.json')
            ann_file_train = os.path.join(args.data_root,'coco/train_sony.json')
    elif args.data == 'ZURICH':
            ann_file = os.path.join(args.data_root,'coco/val_zurich.json')
            ann_file_train = os.path.join(args.data_root,'coco/train_zurich.json')
    elif args.data == 'RAOD':
            ann_file = os.path.join(args.data_root,'coco/val_raod.json')
            ann_file_train = os.path.join(args.data_root,'coco/train_raod.json')
    elif args.data == 'NEW':
            ann_file = os.path.join(args.data_root,'coco/combined_val.json')
            ann_file_train = os.path.join(args.data_root,'coco/combined_train.json')

    args.cfg_options = {'mode_train': args.mode_train,
                        'train_dataloader.dataset.data_root':args.data_root,
                        'train_dataloader.dataset.pipeline.0.is_raw':args.is_raw,\
                        'train_dataloader.dataset.pipeline.2.is_raw':args.is_raw,\
                        'train_dataloader.dataset.pipeline.0.data_root':args.data_root,\
                        'test_dataloader.dataset.ann_file': ann_file_train,
                        'test_dataloader.dataset.pipeline.0.is_raw':args.is_raw,\
                        'test_dataloader.dataset.pipeline.2.is_raw':args.is_raw,
                        'val_dataloader.dataset.pipeline.0.is_raw':args.is_raw,\
                        'val_dataloader.dataset.pipeline.1.is_raw':args.is_raw,
                        'val_dataloader.dataset.ann_file': ann_file,
                        'test_dataloader.dataset.data_root':args.data_root,\
                        'test_dataloader.dataset.pipeline.0.data_root':args.data_root,
                        'val_dataloader.dataset.data_root':args.data_root,\
                        'val_dataloader.dataset.pipeline.0.data_root':args.data_root,\
                        'val_dataloader.dataset.data_prefix.img':args.data_root,\
                        'test_dataloader.dataset.data_prefix.img':args.data_root,\
                        'train_dataloader.dataset.data_prefix.img':args.data_root,\
                        'train_cfg.is_raw': args.is_raw,'train_cfg.gamma_': args.gamma_,\
                        'train_cfg.log': args.log,'train_cfg.epsilon': args.epsilon,'train_cfg.log_lr': args.log_learnable,\
                        'train_cfg.quant': args.quant, 'train_cfg.n_gamma': args.n_gamma, 'train_cfg.use_WB': args.use_WB,                             
                        'test_cfg.is_raw': args.is_raw,'test_cfg.gamma_': args.gamma_,\
                        'test_cfg.log': args.log,'test_cfg.epsilon': args.epsilon,\
                        'test_cfg.quant': args.quant, 'test_cfg.n_gamma': args.n_gamma, 'test_cfg.use_WB': args.use_WB,\
                        'val_cfg.is_raw': args.is_raw,'val_cfg.gamma_': args.gamma_,\
                        'val_cfg.log': args.log,'val_cfg.epsilon': args.epsilon,\
                        'val_cfg.quant': args.quant,'val_cfg.log_lr': args.log_learnable,\
                        'train_cfg.data_type': args.data_type,
                        'val_dataloader.dataset.ann_file': ann_file,
                        'val_evaluator.ann_file': ann_file,
                        'test_evaluator.ann_file': ann_file,
                        'test_cfg.data_type': args.data_type,'test_cfg.log_lr': args.log_learnable,\
                        'val_cfg.data_type': args.data_type, 'val_cfg.n_gamma': args.n_gamma,'val_cfg.use_WB': args.use_WB,
                        'custom_hooks.0.is_raw': args.is_raw, 'custom_hooks.0.gamma_': args.gamma_,\
                        'custom_hooks.0.log': args.log, 'custom_hooks.0.epsilon': args.epsilon,\
                        'custom_hooks.0.quant': args.quant, 'custom_hooks.0.n_gamma': args.n_gamma, 'custom_hooks.0.use_WB': args.use_WB,\
                        'custom_hooks.0.data_type': args.data_type}

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
