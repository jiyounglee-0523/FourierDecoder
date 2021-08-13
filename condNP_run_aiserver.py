import subprocess
import os
from datetime import datetime

# Configuration before run
device = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = device
PATH = '/home/jylee/generativeODE/disentangled_ODE/'
SRC_PATH = PATH+'noncond_main.py'


TRAINING_CONFIG = {
    "encoder": 'Conv',
    'model_type': 'FNP',
    "in_features":1,
    "out_features":1,
    "encoder_hidden_dim": 256,
    "encoder_embedding_dim": 128,
    "latent_dimension": 128,
    "expfunc":'fourier',
    "n_harmonics": 8,
    "n_eig":2,
    "path":'/home/edlab/jylee/generativeODE/output/unconditional/sin/',    #  change this!
    "dataset_path": '/home/edlab/jylee/generativeODE/input/',
    #'dataset_path': '/home/data_storage/jylee_26/NSynth/',
    #'dataset_path': '/home/edlab/jylee/generativeODE/input/not_duplicatedECG/',
    #"path": './',
    "dataset_name": 'conf',     # change this!
    "lower_bound": 1.0,
    "upper_bound": 8.0,      # change this!
    "skip_step": 1,
    "filename": f'{datetime.now().date()}_conf_Conv',                      #  change this!
    "dataset_type": 'sin',                    # change this!
    "notes":'Transformer+FNP+query ',             # change this!
    "n_epochs":1000000,
    "batch_size": 512,
    "device_num" : device,
    "encoder_blocks": 3,
    "encoder_attnheads": 2,
    "decoder_layers": 2,
    # "debug": True,
    "query": True,
    #"continual": True,
    #"mask_reverse": True
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)