import subprocess
import os
from datetime import datetime

# Configuration before run
device = '0'
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
    "n_harmonics": 1000,
    "n_eig":2,
    "path":'/home/edlab/jylee/generativeODE/output/NSynth/',    #  change this!
    #"dataset_path": '/home/edlab/jylee/generativeODE/input/',
    'dataset_path': '/home/data_storage/jylee_26/NSynth/',
    #'dataset_path': '/home/edlab/jylee/generativeODE/input/not_duplicatedECG/',
    #"path": './',
    "lower_bound": 1.0,
    "upper_bound": 1000.0,
    "skip_step": 1,
    "filename": f'{datetime.now().date()}_Transformer_nonquery_1000',                      #  change this!
    "dataset_type": 'NSynth',                    # change this!
    "notes":'Transformer+FNP',             # change this!
    "n_epochs":1000000,
    "batch_size": 64,
    "device_num" : device,
    "encoder_blocks": 3,
    "encoder_attnheads": 2,
    #"debug": True,
    #"query": True,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)