_base_ = [
    '../_base_/datasets/Raw_NOD_detection.py', '../_base_/default_runtime.py'
]

randomness = dict(
    seed = 42,
    diff_rank_seed=True,
    deterministic=True
)

model = dict(
    type='DETR',
    num_queries=100,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        in_channels = 3,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(  # DetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)))),
    decoder=dict(  # DetrTransformerDecoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type='DETRHead',
        num_classes=3,
        embed_dims=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))


# Modify dataset related settings
data_root = ''

metainfo = {
    'classes': ('car','bicycle','person'),
    'palette': [(220, 20, 60),(220, 20, 60),(220, 20, 60)]}
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}, is_raw=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomChoiceResize', scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)], keep_ratio=True, is_raw=True),
    dict(type='PackDetInputs')
]

backend_args = None

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        metainfo=metainfo,
        ann_file='/ceph/mfatima/RAOD/datasets/files_list.txt',
        data_prefix=dict(img='/ceph/mfatima/RAOD/datasets/raw')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/ceph/mfatima/RAOD/datasets/anno/',
        data_prefix=dict(img='/ceph/mfatima/RAOD/datasets/raw')))

test_dataloader = val_dataloader

# Modify metric related settings
#val_evaluator = dict(ann_file='/ceph/mfatima/mmdetection/datasets/pascal_RAW/PASCALRAW/trainval/val.txt')
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='area')
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=250, type='EpochBasedTrainLoop', val_interval=5, is_raw=True, quant=4, gamma_=True,
                 log=False,
                 epsilon = 1.0,)

test_cfg = dict( type='TestLoop',  is_raw=True, quant = 4, gamma_= True,
                 log = False,
                 epsilon = 1.0)

val_cfg = dict(type='ValLoop',  is_raw=True, quant = 4, gamma_= True,
                 log = False,
                 epsilon = 1.0)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

max_epochs = 250
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
