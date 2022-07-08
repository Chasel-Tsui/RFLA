
'''
k = 3
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.255 | bridge        | 0.151 | storage-tank | 0.349 |
| ship     | 0.386 | swimming-pool | 0.102 | vehicle      | 0.245 |
| person   | 0.099 | wind-mill     | 0.061 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2022-07-06 03:52:06,189 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.773 | bridge        | 0.870 | storage-tank | 0.697 |
| ship     | 0.651 | swimming-pool | 0.897 | vehicle      | 0.786 |
| person   | 0.914 | wind-mill     | 0.944 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2022-07-06 03:52:06,990 - mmdet - INFO - Exp name: aitod_faster_r50_rfla_kld_1x.py
2022-07-06 03:52:06,991 - mmdet - INFO - Epoch(val) [12][14018]	bbox_mAP: 0.2060, bbox_mAP_50: 0.5130, bbox_mAP_75: 0.1270, bbox_mAP_vt: 0.0720, bbox_mAP_t: 0.2070, bbox_mAP_s: 0.2650, bbox_mAP_m: 0.3190, bbox_oLRP: 0.8170, bbox_oLRP_Localisation: 0.2910, bbox_oLRP_false_positive: 0.3650, bbox_oLRP_false_negative: 0.4860, bbox_mAP_copypaste: 0.206 -1.000 0.513 0.127 0.072 0.207
Loading and preparing results...
DONE (t=10.10s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2769.24s).
Accumulating evaluation results...
DONE (t=24.98s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.206
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.513
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.127
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.072
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.207
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.265
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.317
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.339
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.343
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.132
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.394
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.439
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.817
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.291
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.365
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.486
# Class-specific LRP-Optimal Thresholds # 
 [0.732 0.791 0.736 0.811 0.706 0.694 0.7   0.763]

k = 2
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.247 | bridge        | 0.155 | storage-tank | 0.350 |
| ship     | 0.376 | swimming-pool | 0.104 | vehicle      | 0.245 |
| person   | 0.103 | wind-mill     | 0.054 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2022-07-06 16:54:10,793 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.778 | bridge        | 0.868 | storage-tank | 0.697 |
| ship     | 0.663 | swimming-pool | 0.904 | vehicle      | 0.786 |
| person   | 0.913 | wind-mill     | 0.946 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2022-07-06 16:54:11,648 - mmdet - INFO - Exp name: aitod_faster_r50_rfla_kld_1x.py
2022-07-06 16:54:11,649 - mmdet - INFO - Epoch(val) [12][14018]	bbox_mAP: 0.2040, bbox_mAP_50: 0.5010, bbox_mAP_75: 0.1240, bbox_mAP_vt: 0.0740, bbox_mAP_t: 0.2070, bbox_mAP_s: 0.2570, bbox_mAP_m: 0.3310, bbox_oLRP: 0.8200, bbox_oLRP_Localisation: 0.2910, bbox_oLRP_false_positive: 0.3700, bbox_oLRP_false_negative: 0.4970, bbox_mAP_copypaste: 0.204 -1.000 0.501 0.124 0.074 0.207
Loading and preparing results...
DONE (t=8.57s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2828.11s).
Accumulating evaluation results...
DONE (t=24.99s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.204
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.501
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.124
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.074
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.207
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.257
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.331
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.320
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.343
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.347
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.135
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.355
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.405
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.454
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.820
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.291
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.370
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.497
# Class-specific LRP-Optimal Thresholds # 
 [0.724 0.789 0.742 0.826 0.751 0.662 0.695 0.82 ]

k = 3 lr= 0.01

'''
_base_ = [
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  #Down sampled by 4, 8, 16, 32 times respectively
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='RFGenerator', # Effective Receptive Field as prior
            fpn_layer='p2', # start FPN level P2
            fraction=0.5, # the fraction of ERF to TRF
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='HieAssigner', # Hierarchical Label Assigner (HLA)
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kld', # KLD as RFD for label assignment
                topk=[3,1],
                ratio=0.9), # decay factor
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=512),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

#fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.02/4, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12, metric='bbox')