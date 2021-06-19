import subprocess
import os

# Configuration before run
device = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = device
PATH = '/home/swryu/jylee/generativeODE/disentangled_ODE/'
SRC_PATH = PATH+'cond_main.py'


TRAINING_CONFIG = {
    "test_model": 'NP',
    "encoder": 'Transformer',
    'model_type': 'FNP',
    "in_features":1,
    "out_features":1,
    "encoder_hidden_dim": 128,
    "encoder_embedding_dim": 128,
    "latent_dimension":64,
    "expfunc":'fourier',
    "n_harmonics": 8,
    "n_eig":2,
    "path":'/home/swryu/jylee/data/output/0619/',    #  change this!
    "dataset_path": '/home/swryu/jylee/data/input/',
    #"path": './',
    "lower_bound": 1.0,
    "upper_bound": 8.0,
    "filename": 'Transformer_FNP_sin',                      #  change this!
    "dataset_type": 'sin',
    "notes":'Transformer+FNP+sin, updated dataset',             # change this!
    "n_epochs":10000,
    "batch_size":512,
    "device_num" : device
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)