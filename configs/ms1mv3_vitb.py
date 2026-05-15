from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "vit_b_dp005_mask_005"
config.resume = False
config.output = ""  # TODO: output directory for checkpoints / tensorboard / logs
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1
config.optimizer = "adamw"
config.batch_size = 128
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.rec = ""  # TODO: path to dataset .rec/.idx directory (e.g., MS1MV3 or CASIA-WebFace)
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
