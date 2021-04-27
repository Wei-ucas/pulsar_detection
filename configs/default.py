
train_configs = {
        "epoch":10,
        "learning_rate":0.001,
        'lr_step':4,
        "save_step":5,
        "verbose":10
    }


data_set = {
    'type': 'train',
    'file_type': 'img',
    'file_path': '/ssd/wangw/Ai_pulsar_search/detection/fake_image_sets/fake_pulsar_v8_2000_png',
    'ann_path' : '/ssd/wangw/Ai_pulsar_search/detection/annotations/fake_pulsar_v8_2000_ann.json',
    'batch_size': 6,
    # 'norm_param': (250,137),
    'num_works':4,
    'pin_memory':True,

    'fch1':1499.93896484375,
    'fend':1000,
    'scale_tsamp':0.006291456,
}


feature_map_channel = 128
fpn_strides = [4, 8, 16,32]

backbone = {
    'type':'ResNet',
    'depth':50
}

neck = {
    'type':'FPN',
    'out_channels': feature_map_channel,
    'in_channels':[64, 128,256,512],
    'num_in':4,
    'num_out':4,
    'out_levels':[1,2,3,4]
}

det_head = {
    'num_convs': 2,
    'in_channels': feature_map_channel,
    'cls_channels': feature_map_channel,
    'reg_channels': feature_map_channel,
    'loss_alpha': 0.25,
    'loss_gamma': 2,
    'fpn_stride': [4, 8 ,16,32],
    'norm_reg_target': False,
    'center_sampling_radius':0
}
postprocess = {
    'score': 0.7,
    'iou': 0.1,
    'max_num_per_img': 100
}