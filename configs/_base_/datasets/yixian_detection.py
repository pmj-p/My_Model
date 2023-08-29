# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/pmj/dataset/yixian_coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_3D', to_float32=False, lesion_input=True, num_slice=7, window=[-1024, 1050] ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg, is_3d_input=True, num_slice=7),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile_3D', to_float32=False, lesion_input=True, num_slice=7, window=[-1024, 1050]),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg, is_3d_input=True, num_slice=9),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
