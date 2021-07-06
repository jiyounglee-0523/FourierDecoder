import subprocess
import os
from datetime import datetime

# Configuration before run
device = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = device
PATH = '/home/jylee/generativeODE/disentangled_ODE/'
SRC_PATH = PATH+'base_trans.py'


TRAINING_CONFIG = {
    'dataset_path': '/home/jylee/data/generativeODE/input/',
    'device_num': '0',
    'path': '/home/jylee/data/generativeODE/output/sin/',
    'filename': 'base_trans'
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)