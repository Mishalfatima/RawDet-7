# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" 

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config',default= './configs/faster_rcnn/faster-rcnn_r50_fpn_1x_NEW_IN_Raw_COCO.py',help='test config file path')
    parser.add_argument('--checkpoint', default= '',help='checkpoint file')
    parser.add_argument('--data-root', default='./datasets/RAW-RGB-Dataset', help='the dir to save logs and models'),
    parser.add_argument(
        '--work-dir', default='checkpoints/',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show',  action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    parser.add_argument(
        "--quant", default=4, type=int, help="Quantization Level, 0 means sRGB and no quantization, 12 means raw 12-bit input and no quantization"
    )
    parser.add_argument(
        "--log" , action='store_true', help="Logarithmic or Linear Quantization"
    )
    parser.add_argument(
        "--epsilon", default=1, type=float, help="epislon for log"
    )
    parser.add_argument(
        "--is_raw",  action='store_true', help="whether to use raw data or not"
    )
    parser.add_argument(
        "--gamma_",  action='store_true', help="whether to use gamma_correction or not"
    )
    parser.add_argument(
        "--data_type", default='NEW', help="currently supports pascalraw and NOD"
    )
    parser.add_argument(
        "--n_gamma", default=1, type=int, help="whether to use 1 gamma for the whole dataset or 2 for different time of the day"
    )
    parser.add_argument(
        "--use_WB",default=False,  action='store_true', help="whether to use white balance correction or not"
    )
    parser.add_argument(
        "--test_data",default='NEW',type=str,  help="PRAW, NIKON, SONY, RAOD, ZURICH, NEW"
    )
    parser.add_argument(
        "--vis",default=False,type=bool,  help="whether to visualize the results or not"
    )


    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    if args.checkpoint == '' :
        args.checkpoint = os.path.join(args.work_dir, 'latest.pth')

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    if args.test_data == 'PRAW':
        ann_file = os.path.join(args.data_root,'coco/val_praw.json')
    elif args.test_data == 'NIKON':
        ann_file = os.path.join(args.data_root,'coco/val_nikon.json')
    elif args.test_data == 'SONY':
        ann_file = os.path.join(args.data_root,'coco/val_sony.json')
    elif args.test_data == 'ZURICH':
        ann_file = os.path.join(args.data_root,'coco/val_zurich.json')
    elif args.test_data == 'RAOD':
        ann_file = os.path.join(args.data_root,'coco/val_raod.json')
    elif args.test_data == 'NEW':
        ann_file = os.path.join(args.data_root,'coco/combined_val.json')
        
    args.cfg_options = {'train_dataloader.dataset.data_root':args.data_root,
                        'train_dataloader.dataset.pipeline.0.is_raw':args.is_raw,\
                        'train_dataloader.dataset.pipeline.2.is_raw':args.is_raw,\
                        'train_dataloader.dataset.pipeline.0.data_root':args.data_root,\
                        'test_dataloader.dataset.ann_file': ann_file,\
                        'test_dataloader.dataset.pipeline.0.is_raw':args.is_raw,\
                        'test_dataloader.dataset.pipeline.1.is_raw':args.is_raw,
                        'test_dataloader.dataset.data_root':args.data_root,\
                        'test_dataloader.dataset.pipeline.0.data_root':args.data_root,
                        'val_dataloader.dataset.data_root':args.data_root,\
                        'val_dataloader.dataset.ann_file': ann_file,\
                        'val_dataloader.dataset.pipeline.0.data_root':args.data_root,\
                        'val_dataloader.dataset.data_prefix.img':args.data_root,\
                        'test_dataloader.dataset.data_prefix.img':args.data_root,\
                        'train_dataloader.dataset.data_prefix.img':args.data_root,\
                        'train_cfg.is_raw': args.is_raw,'train_cfg.gamma_': args.gamma_,\
                        'train_cfg.log': args.log,'train_cfg.epsilon': args.epsilon,\
                        'train_cfg.quant': args.quant, 'train_cfg.n_gamma': args.n_gamma, 'train_cfg.use_WB': args.use_WB,                             
                        'test_cfg.is_raw': args.is_raw,'test_cfg.gamma_': args.gamma_,\
                        'test_cfg.log': args.log,'test_cfg.epsilon': args.epsilon,\
                        'test_cfg.quant': args.quant, 'test_cfg.n_gamma': args.n_gamma, 'test_cfg.use_WB': args.use_WB,\
                        'test_cfg.vis': args.vis, \
                        'test_cfg.work_dir': args.work_dir,
                        'val_cfg.is_raw': args.is_raw,'val_cfg.gamma_': args.gamma_,\
                        'val_cfg.log': args.log,'val_cfg.epsilon': args.epsilon,\
                        'val_dataloader.dataset.ann_file': ann_file,
                        'val_evaluator.ann_file': ann_file,
                        'test_evaluator.ann_file': ann_file,
                        'val_cfg.quant': args.quant,\
                        'train_cfg.data_type': args.data_type,
                        'test_cfg.data_type': args.data_type,\
                        'val_cfg.data_type': args.data_type, 'val_cfg.n_gamma': args.n_gamma,'val_cfg.use_WB': args.use_WB
                            }

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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
