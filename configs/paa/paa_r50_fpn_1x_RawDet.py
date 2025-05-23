_base_ = [
    '../_base_/datasets/rawdet_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Modify dataset related settings
data_root = ''

metainfo = {
    'classes':
        ('car', 'truck', 'tram', 'person', 'bicycle', 'motorcycle', 'bus'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 60, 100)]}

custom_hooks = [
    dict(type='MyHook', is_raw=True, quant=4, gamma_=True, log=True,log_lr=False, \
         epsilon=0.01, data_type='NEW', n_gamma=1, use_WB='True')
         ]

# model settings
model = dict(
    type='PAA',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='PAAHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=7,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.1,
            neg_iou_thr=0.1,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.001))

import os

randomness = dict(
    seed = 42,
    diff_rank_seed=True,
    deterministic=True
)


backend_args = None

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_coco/combined_train.json'
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_coco/combined_val.json'))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_coco/combined_val.json'))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations_coco/combined_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

train_cfg = dict(max_epochs=140, type='EpochBasedTrainLoop', val_interval=10, is_raw=True, quant = 4, gamma_= True,
                 log = False,
                 epsilon = 1.0,data_type = 'NEW', n_gamma=1, use_WB=False)

test_cfg = dict(type='TestLoop',  is_raw=True, quant=4, gamma_= True,
                 log = False,
                 epsilon = 1.0, data_type = 'NEW', n_gamma=1, use_WB=False)

val_cfg = dict(type='ValLoop',  is_raw=True, quant=4, gamma_= True,
                 log = False,
                 epsilon = 1.0, data_type = 'NEW', n_gamma=1, use_WB=False)
