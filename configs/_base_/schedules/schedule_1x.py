# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=140, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
'''param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=10,  #This was different in the original file along with start_factor
        end=122,
        by_epoch=True,
        milestones=[88, 120],
        gamma=0.1)
]'''

param_scheduler = [dict(
        type='LinearLR', start_factor=0.00001, by_epoch=False, begin=0, end=4000),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=140, convert_to_iter_based=True)]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
