import subprocess
import os
from datetime import datetime

# Configuration before run
device = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = device
PATH = '/home/jylee/generativeODE/disentangled_ODE/'
SRC_PATH = PATH+'cond_main.py'


TRAINING_CONFIG = {
    "test_model": 'NP',
    "encoder": 'Transformer',
    'model_type': 'FNP',
    "in_features":1,
    "out_features":1,
    "encoder_hidden_dim": 256,
    "encoder_embedding_dim": 128,
    "latent_dimension": 128,
    "expfunc":'fourier',
    "n_harmonics": 8,
    "n_eig":2,
    #"path":'/home/jylee/data/generativeODE/output/sin/',    #  change this!
    #"dataset_path": '/home/data_storage/jylee_26/NSynth/',
    'dataset_path': '/home/jylee/data/generativeODE/input/',
    "path": './',
    "lower_bound": 1.0,
    "upper_bound": 8.0,
    "filename": f'{datetime.now().date()}_Transformer_z_cls_conf',                      #  change this!
    "dataset_type": 'sin',
    "notes":'Transformer+FNP+sin, input z cls, confusing dataset',             # change this!
    "n_epochs":1000000,
    "batch_size":256,
    "device_num" : device,
    "encoder_blocks": 2,
    "encoder_attnheads": 2,
    # "debug": True,
    "skip_step": 1
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)