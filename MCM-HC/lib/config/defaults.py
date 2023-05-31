from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.SEED = 0
_C.EXP.DEBUG = False
_C.EXP.CP_PROJECT = True
_C.EXP.DETERMINISTIC = True
_C.EXP.CUDNN_BENCHMARK = True
_C.PRETRAIN = False



# -----------------------------------------------------------------------------
# Mode
# -----------------------------------------------------------------------------
_C.MODE = CN()
_C.MODE.CMPCM        = 1
_C.MODE.MIM          = False       # 是否使用MIM
_C.MODE.MLM          = False       # 是否使用MLM
_C.MODE.MASK_SEP     = False      # MIM和MLM是否使用在整个阶段，还是只有warmup阶段使用
_C.MODE.VAL_USEMASK  = False      # 在validation的时候，是否使用MIM（测试的时候，是否也将图片d）
_C.MODE.CONCATIMAGE  = False      # 将gaussian后的feature和未gaussion的feature concat，否则只有gaussian的

_C.MODE.CODEBOOK     = False       # 是否使用codebook
_C.MODE.CODEBOOK_NUM = 800        # codebook 大小
_C.MODE.CODEBOOK_TYPE= "dvae"     # "vqvae" and "dvae" are available now
_C.MODE.CODEBOOK_DIM = 768        # codebook中的向量的维度大小
_C.MODE.VISUAL_VQ    = False       # 经过visual  backbone得到对应的visual feature后， 是否要经过codebook，替换特征
_C.MODE.TEXTUAL_VQ   = False       # 经过textual backbone得到对应的textual feature后，是否要经过codebook，替换特征
_C.MODE.VISUAL_VQ_W  = 0.2        # 要是visual feature 经过codebook则计算visual_vq_loss，这个参数是这个loss的权重
_C.MODE.TEXTUAL_VQ_W = 0.2        # 要是textual feature 经过codebook则计算textual_vq_loss，这个参数是这个loss的权重
_C.MODE.MLM_W        = 0.2        # MLM重建的权重
_C.MODE.MIM_W        = 0.2        # MIM重建的权重
_C.MODE.MIM_RATIO    = 0.15       # MIM mask掉的比率

_C.MODE.MM           = False       # 经过visual backbone和textual backbone后，的两个特征，是否连接起来，经过multi modal encoder
_C.MODE.DGA          = False       # 上次ACM MM的文章，中的share dictionary
_C.MODE.DGA_SIZE     = 400        # 上面share dictionary的大小



_C.MODE.TRAIN_NAME   = ""         # 本次训练的名称，和output dir要调成相同的
_C.MODE.VQ_BETA      = 0.15       # vqloss 的权重，现在意义不大
_C.MODE.SAVE_IMAGE   = False      # MIM 重建图片之后，是否保存图片，到output/recon/$C.MODE.TRAIN_NAME 目录下
_C.MODE.DECODER_SIZE = 1.0        # 按常理，visual feature 经过decoder之后，重建的大小为（3， 284，128），若设置为0.25 则重建的大小为（3， 96， 32），并且将图片下采样0.25，然后计算重建loss


_C.MODE.HASH         = True


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.VAL = ()
_C.DATASETS.TEST = ()
_C.DATASETS.USE_ONEHOT = False # onehot sentence
_C.DATASETS.USE_SEG = False # segmentation map
_C.DATASETS.USE_ATT = False # attribute
_C.DATASETS.BIN_SEG = False
_C.DATASETS.MAX_LENGTH = 100
_C.DATASETS.MAX_ATTR_LENGTH = 25
_C.DATASETS.VOCAB_PATH = ''
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.IMS_PER_ID = 4
_C.DATALOADER.EN_SAMPLER = "TrainingSampler"

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 224
_C.INPUT.WIDTH = 224
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# `True` if cropping is used for data augmentation during training
_C.INPUT.CROP = CN({"ENABLED": False})
# Size of the image cropped
_C.INPUT.CROP.SIZE = [224, 224]
# Size of the origin size cropped
_C.INPUT.CROP.SCALE = [0.16, 1]
# Aspect ratio of the origin aspect ratio cropped
_C.INPUT.CROP.RATIO = [3./4., 4./3.]

# Random probability for image horizontal flip
_C.INPUT.FLIP = CN({"ENABLED": False})
_C.INPUT.FLIP.PROB = 0.5

# Value of padding size
_C.INPUT.PADDING = CN({"ENABLED": False})
_C.INPUT.PADDING.MODE = 'constant'
_C.INPUT.PADDING.SIZE = 10

# Random color jitter
_C.INPUT.CJ = CN({"ENABLED": False})
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Random Affine
_C.INPUT.AFFINE = CN({"ENABLED": False})

# Auto augmentation
_C.INPUT.AUTOAUG = CN({"ENABLED": False})
_C.INPUT.AUTOAUG.PROB = 0.0

# Augmix augmentation
_C.INPUT.AUGMIX = CN({"ENABLED": False})
_C.INPUT.AUGMIX.PROB = 0.0

# Random Erasing
_C.INPUT.REA = CN({"ENABLED": False})
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.VALUE = [0.485, 0.456, 0.406]
# Random Patch
_C.INPUT.RPT = CN({"ENABLED": False})
_C.INPUT.RPT.PROB = 0.5


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.VISUAL_MODEL = "resnet50"
_C.MODEL.TEXTUAL_MODEL = "bilstm"
_C.MODEL.NUM_CLASSES = 11003
_C.MODEL.NUM_PARTS = 5
_C.MODEL.FREEZE_TEXT = False
_C.MODEL.FREEZE_VISUAL = False
_C.MODEL.WEIGHT = "imagenet"
_C.MODEL.WHOLE = False
_C.MODEL.INFERENCE_MODE = "common"
_C.MODEL.SYNC_BN = True
_C.MODEL.IMS_PER_BN = 16  # set as samples_per_gpu is better

# -----------------------------------------------------------------------------
# MoCo
# -----------------------------------------------------------------------------
_C.MODEL.MOCO = CN()
_C.MODEL.MOCO.K = 1024
_C.MODEL.MOCO.M = 0.999
_C.MODEL.MOCO.FC = True


# -----------------------------------------------------------------------------
# LSTM
# -----------------------------------------------------------------------------
_C.MODEL.LSTM = CN()
_C.MODEL.LSTM.ONEHOT = True
_C.MODEL.LSTM.EMBEDDING_SIZE = 512
_C.MODEL.LSTM.NUM_UNITS = 512
_C.MODEL.LSTM.VOCABULARY_SIZE = 12000
_C.MODEL.LSTM.DROPOUT_KEEP_PROB = 0.7
_C.MODEL.LSTM.MAX_LENGTH = 100


# -----------------------------------------------------------------------------
# TextCNN
# -----------------------------------------------------------------------------
_C.MODEL.TEXT_CNN = CN()
_C.MODEL.TEXT_CNN.EMBEDDING_SIZE = 512
_C.MODEL.TEXT_CNN.FILTER_SIZE = [3, 5, 7, 9]
_C.MODEL.TEXT_CNN.NUM_FILTERS = 256
_C.MODEL.TEXT_CNN.VOCABULARY_SIZE = 12000
_C.MODEL.TEXT_CNN.DROPOUT = 0.5


# -----------------------------------------------------------------------------
# GRU
# -----------------------------------------------------------------------------
_C.MODEL.GRU = CN()
_C.MODEL.GRU.ONEHOT = "yes"
_C.MODEL.GRU.EMBEDDING_SIZE = 512
_C.MODEL.GRU.NUM_UNITS = 512
_C.MODEL.GRU.VOCABULARY_SIZE = 12000
_C.MODEL.GRU.DROPOUT_KEEP_PROB = 0.7
_C.MODEL.GRU.MAX_LENGTH = 100
_C.MODEL.GRU.NUM_LAYER = 1
_C.MODEL.GRU.GET_MASK_LABEL = False
_C.MODEL.GRU.CUT_MIX = False
_C.MODEL.GRU.RANDOM_DELETE = False
_C.MODEL.GRU.CUT_NEG = False

# -----------------------------------------------------------------------------
# BERT
# -----------------------------------------------------------------------------
_C.MODEL.BERT = CN()
_C.MODEL.BERT.POOL = True
_C.MODEL.BERT.MAX_LENGTH = 100
_C.MODEL.BERT.OUTPUT_ALL = False

# -----------------------------------------------------------------------------
# CLIP
# -----------------------------------------------------------------------------
_C.MODEL.CLIP = CN()
_C.MODEL.CLIP.MAX_LENGTH = 100
_C.MODEL.CLIP.OUTPUT_ALL = False

# -----------------------------------------------------------------------------
# Resnet
# -----------------------------------------------------------------------------
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.RES5_STRIDE = 2
_C.MODEL.RESNET.RES5_DILATION = 1
_C.MODEL.RESNET.ATTN_POOL = False
_C.MODEL.RESNET.IBNA = False
_C.MODEL.RESNET.PATCH_MIX = False
_C.MODEL.RESNET.PRETRAINED = None
_C.MODEL.RESNET.OUTPUT_ALL = False


# -----------------------------------------------------------------------------
# Vision Transformer
# -----------------------------------------------------------------------------
_C.MODEL.VIT = CN()
_C.MODEL.VIT.MODE = "first"  # first, average, all
_C.MODEL.VIT.CKPT = None


# -----------------------------------------------------------------------------
# Hybrid Vision Transformer
# -----------------------------------------------------------------------------
_C.MODEL.HBVIT = CN()
_C.MODEL.HBVIT.BACKBONE = "resnet50"
_C.MODEL.HBVIT.EMBED_DIM = 1024
_C.MODEL.HBVIT.DEPTH = 4
_C.MODEL.HBVIT.NUM_HEADS = 8
_C.MODEL.HBVIT.PATCH_SIZE = 1
_C.MODEL.HBVIT.OUTPUT_ALL = False


# -----------------------------------------------------------------------------
# Hybrid GRU
# -----------------------------------------------------------------------------
_C.MODEL.HBGRU = CN()
_C.MODEL.HBGRU.EMBED_DIM = 1024
_C.MODEL.HBGRU.DEPTH = 4
_C.MODEL.HBGRU.NUM_HEADS = 8
_C.MODEL.HBGRU.FF_DIM = 4096
_C.MODEL.HBGRU.OUTPUT_ALL = False


# -----------------------------------------------------------------------------
# Textual Transformer Encoder
# -----------------------------------------------------------------------------
_C.MODEL.TRANS_ENCODER = CN()
_C.MODEL.TRANS_ENCODER.EMBED_DIM = 512
_C.MODEL.TRANS_ENCODER.DEPTH = 4
_C.MODEL.TRANS_ENCODER.NUM_HEADS = 8
_C.MODEL.TRANS_ENCODER.FF_DIM = 1024
_C.MODEL.TRANS_ENCODER.VOCABULARY_SIZE = 12000
_C.MODEL.TRANS_ENCODER.ONEHOT = "yes"
_C.MODEL.TRANS_ENCODER.LEARN_PS = False
_C.MODEL.TRANS_ENCODER.DROPOUT = 0.1

# HEAD related configuration
# -----------------------------------------------------------------------------
# SAFA
# -----------------------------------------------------------------------------
_C.MODEL.SAFA = CN()
_C.MODEL.SAFA.NUM_HEADS = 10
_C.MODEL.SAFA.V_DIM = 768
_C.MODEL.SAFA.DROPOUT = 0.

# -----------------------------------------------------------------------------
# K Attention
# -----------------------------------------------------------------------------
_C.MODEL.KATT = CN()
_C.MODEL.KATT.K = 3
_C.MODEL.KATT.NUM_EMBED = 128
_C.MODEL.KATT.SPA_HEIGHT = 12
_C.MODEL.KATT.SPA_WIDTH = 4
_C.MODEL.KATT.SEQLEN = 64

# -----------------------------------------------------------------------------
# ITI HEAD
# -----------------------------------------------------------------------------
_C.MODEL.ITI = CN()
_C.MODEL.ITI.ANNEAL = True

# -----------------------------------------------------------------------------
# Invertible HEAD
# -----------------------------------------------------------------------------
_C.MODEL.INVERT = CN()
# embedding input to the invertible neural network
_C.MODEL.INVERT.EMBEDDING = CN()
_C.MODEL.INVERT.EMBEDDING.STYLE = 'single' # multi, mha
_C.MODEL.INVERT.EMBEDDING.NUM_BRANCHES = 6
_C.MODEL.INVERT.EMBEDDING.TEXT_BRANCHES_TYPE = 'conv' # 'transformer'
_C.MODEL.INVERT.EMBEDDING.DROPOUT = 0.0
_C.MODEL.INVERT.EMBEDDING.BOTTLENECK_GLOBAL = False
# invertible neural network
_C.MODEL.INVERT.INVERT_NET = CN()
_C.MODEL.INVERT.INVERT_NET.STYLE = 'coupling' # 'transformer'
_C.MODEL.INVERT.INVERT_NET.HIDDEN_DIM = 512
_C.MODEL.INVERT.INVERT_NET.HIDDEN_DEPTH = 1
_C.MODEL.INVERT.INVERT_NET.NUM_FLOWS = 2
_C.MODEL.INVERT.INVERT_NET.NORM_TYPE = 'batchnorm'  # 'actnorm'
_C.MODEL.INVERT.INVERT_NET.FEAT_SHUFFLE = True
_C.MODEL.INVERT.INVERT_NET.INIT_ACTNORM = 'none'  # none, random, gaussian
# invertible transformer
_C.MODEL.INVERT.INVERT_NET.TRANS = CN()
_C.MODEL.INVERT.INVERT_NET.TRANS.SELF_ATTNETION_TYPE = 'normal'  # normal, relative
_C.MODEL.INVERT.INVERT_NET.TRANS.NUM_HEADS = 4
_C.MODEL.INVERT.INVERT_NET.TRANS.ATT_DROPOUT = 0.1
_C.MODEL.INVERT.INVERT_NET.TRANS.MAX_POSITION = 100
_C.MODEL.INVERT.INVERT_NET.TRANS.LAYER_NORM_TYPE = 'sandwich' # prenorm, sandwich
_C.MODEL.INVERT.INVERT_NET.TRANS.ENCODER_FFN_DIM = 256
_C.MODEL.INVERT.INVERT_NET.TRANS.DROPOUT = 0.1
# test augmentation
_C.MODEL.INVERT.TEST = CN()
_C.MODEL.INVERT.TEST.OUTPUT_ALL_STATES = False
_C.MODEL.INVERT.TEST.OUTPUT_ALL_SCALES = False

# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
_C.MODEL.EMBEDDING = CN()
_C.MODEL.EMBEDDING.EMBED_HEAD = "simple"  # segpool, seg, multiscale
_C.MODEL.EMBEDDING.FEATURE_SIZE = 512
_C.MODEL.EMBEDDING.DROPOUT_PROB = 0.3
_C.MODEL.EMBEDDING.BNNECK = False
_C.MODEL.EMBEDDING.K_RECIPROCAL = True
_C.MODEL.EMBEDDING.SHARED_LAYER = False
_C.MODEL.EMBEDDING.TASK = ["CMR"]
# set loss type and corresponding loss weight
_C.MODEL.EMBEDDING.LOSS_TYPE = ["instance_loss", ]
_C.MODEL.EMBEDDING.LOSS_WEIGHT = [1.0, ]
# supported pooling layer
# fastavgpool, avgpool, maxpool, gempoolP, gempool,
# avgmaxpool, clipavgpool, identity, flatten
_C.MODEL.EMBEDDING.VISUAL_POOL = 'avgpool'
_C.MODEL.EMBEDDING.TEXTUAL_POOL = 'identity'
# use text mask in head
_C.MODEL.EMBEDDING.TEXT_MASK = True

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.MODEL.LOSSES = CN()
# cross entropy loss in instance_loss.py
_C.MODEL.LOSSES.CE = CN()
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.SHARED_PROJECTION = True

# global align loss
_C.MODEL.LOSSES.GA = CN()
_C.MODEL.LOSSES.GA.LEARN_SCALE = True
_C.MODEL.LOSSES.GA.MIXTURE = False

# triplet loss
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = False
_C.MODEL.LOSSES.TRI.WEIGHT_MINING = False

# Softmax Triplet Loss options
_C.MODEL.LOSSES.STRI = CN()
_C.MODEL.LOSSES.STRI.NORM_FEAT = True
_C.MODEL.LOSSES.STRI.MARGIN = 0.0
_C.MODEL.LOSSES.STRI.TAU = 1.0

# Soft Softmax Triplet Loss options
_C.MODEL.LOSSES.SSTRI = CN()
_C.MODEL.LOSSES.SSTRI.NORM_FEAT = True
_C.MODEL.LOSSES.SSTRI.MARGIN = 0.0
_C.MODEL.LOSSES.SSTRI.TAU = 1.0

# Contrastive Loss
_C.MODEL.LOSSES.CONTRAST = CN()
_C.MODEL.LOSSES.CONTRAST.LEARNABLE_TEMP = False

# Dual Softmax Loss
_C.MODEL.LOSSES.DSL = CN()
_C.MODEL.LOSSES.DSL.LEARNABLE_TEMP = False

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.SAVE_LAST_CHECKPOINT = True
_C.SOLVER.PERIODIC_CHECKPOINT = False
_C.SOLVER.CHECKPOINT_TABLES = ['',]
_C.SOLVER.EVALUATE_PERIOD = 1
_C.SOLVER.PRINT_ITER = 100
_C.SOLVER.AMP = False

_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.BIAS_LR_FACTOR = 2

# This LR is applied to the embedding head if
# you want to 10x higher than BASE_LR.
_C.SOLVER.HEADS_LR_FACTOR = 1.

_C.SOLVER.WEIGHT_DECAY = 0.00004
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.00004
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999
_C.SOLVER.SGD_MOMENTUM = 0.9

_C.SOLVER.LRSCHEDULER = "step" # exp, linear, poly, cosine

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.PART_STRATEGY = CN({"ENABLED": False})
_C.SOLVER.PART_STRATEGY.TEXTUAL_MODEL_LR_FACTOR = 1.
_C.SOLVER.PART_STRATEGY.TEXTUAL_MODEL_WD_FACTOR = 1.
_C.SOLVER.PART_STRATEGY.VISUAL_MODEL_LR_FACTOR = 1.
_C.SOLVER.PART_STRATEGY.VISUAL_MODEL_WD_FACTOR = 1.
_C.SOLVER.PART_STRATEGY.EMBED_HEAD_LR_FACTOR = 1.
_C.SOLVER.PART_STRATEGY.EMBED_HEAD_WD_FACTOR = 1.

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500,)

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.0001

# Backbone freeze iters
_C.SOLVER.FREEZE_EPOCHS = 0
_C.SOLVER.FREEZE_LAYERS = []

# Contiguous parameters
_C.SOLVER.CONTIGUOUS_PARAMS = False

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 25.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
# Gradient accumulation
_C.SOLVER.STEPS_TO_ACCUMULATE = 1

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 16
_C.TEST.SUM_SIM = True
_C.TEST.SAVE_DATA = True


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #
# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"
# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
