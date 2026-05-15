from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.output = ""  # TODO: output directory for checkpoints / tensorboard / logs
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = ""  # TODO: path to dataset .rec/.idx directory (e.g., MS1MV3 or CASIA-WebFace)
config.num_classes = 10572
config.num_image = 490623
config.num_epoch = 34
config.warmup_epoch = 1
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]