from yacs.config import CfgNode as CN


# Create root config node
_C = CN()
# Config name
_C.NAME = ""
# Config version to manage version of configuration names and default
_C.VERSION = "0.1"
# Device
_C.DEVICE = 'cpu'
# Mode: ['train', 'eval']
_C.MODE = 'train'


# ----------------------------------------
# System config
# ----------------------------------------
_C.SYSTEM = CN()

# Number of workers for dataloader
_C.SYSTEM.NUM_WORKERS = 4
# Random seed for seeding everything (NumPy, Torch,...)
_C.SYSTEM.SEED = 0
# Use half floating point precision
_C.SYSTEM.FP16 = True
# FP16 Optimization level. See more at: https://nvidia.github.io/apex/amp.html#opt-levels
_C.SYSTEM.OPT_L = "O2"


# ----------------------------------------
# Directory name config
# ----------------------------------------
_C.DIRS = CN()

# Train, Validation and Testing image folders
_C.DIRS.TRAIN_IMAGES = ""
_C.DIRS.VALIDATION_IMAGES = ""
_C.DIRS.TEST_IMAGES = ""
# Trained weights folder
_C.DIRS.WEIGHTS = "./weights/"
# Inference output folder
_C.DIRS.OUTPUTS = "./outputs/"
# Training log folder
_C.DIRS.LOGS = "./logs/"


# ----------------------------------------
# Datasets config
# ----------------------------------------
_C.DATA = CN()

# Create small subset to debug
_C.DATA.DEBUG = False
# Datasets problem (multiclass / multilabel)
_C.DATA.TYPE = ""
# For CSV loading dataset style
# If dataset is contructed as folders with one class for each folder, see ImageFolder dataset style
# Train, Validation and Test CSV files
_C.DATA.CSV = CN()
_C.DATA.CSV.TRAIN = ""
_C.DATA.CSV.VALIDATION = ""
_C.DATA.CSV.TEST = ""


# ----------------------------------------
# Training config
# ----------------------------------------
_C.TRAIN = CN()

# Number of epoches for each cycle. Length of epoches list must equals number of cycle
_C.TRAIN.EPOCHES = 50
# Training batchsize
_C.TRAIN.BATCH_SIZE = 32
# Learning rate
_C.TRAIN.LEARNING_RATE = 0.01


# ----------------------------------------
# Model config
# ----------------------------------------
_C.MODEL = CN()

_C.MODEL.HIDDEN_SIZE = 256
# Load ImageNet pretrained weights
_C.MODEL.PRETRAINED = True

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`