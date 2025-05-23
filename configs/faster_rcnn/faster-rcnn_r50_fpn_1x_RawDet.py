_base_ = ['../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/rawdet_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

# We also need to change the num_classes in head to match the dataset's annotation
#model = dict(roi_head=dict(bbox_head=dict(num_classes=3), mask_head=dict( type='FCNMaskHead', num_classes=3, loss_mask=dict( type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
mode_train = ''

import os
model = dict(roi_head=dict(bbox_head=dict(num_classes=7)))

randomness = dict(
    seed = 42,
    diff_rank_seed=True,
    deterministic=True
)

custom_hooks = [
    dict(type='MyHook', is_raw=True, quant=4, gamma_=True, log=True,log_lr=False, \
         epsilon=0.01, data_type='NEW', n_gamma=1, use_WB='False')
]

# Modify dataset related settings
data_root = ''

metainfo = {
    'classes':
        ('car', 'truck', 'tram', 'person', 'bicycle', 'motorcycle', 'bus'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 60, 100)]}

backend_args = None

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_coco/combined_train.json'))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_coco/combined_val.json'))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
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
                 epsilon = 1.0, log_lr = False, data_type = 'NEW', n_gamma=1, use_WB=False)

test_cfg = dict(type='TestLoop',  is_raw=True, quant=4, gamma_= True,
                 log = False,
                 epsilon = 1.0,log_lr = False,  data_type = 'NEW', n_gamma=1, use_WB=False)

val_cfg = dict(type='ValLoop',  is_raw=True, quant=4, gamma_= True,
                 log = False,
                 epsilon = 1.0,log_lr = False,  data_type = 'NEW', n_gamma=1, use_WB=False)