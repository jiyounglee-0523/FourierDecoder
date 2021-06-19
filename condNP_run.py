import subprocess
import os

# Configuration before run
device = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = device
PATH = '/home/swryu/jylee/generativeODE/disentangled_ODE/'
SRC_PATH = PATH+'cond_main.py'


TRAINING_CONFIG = {
    "test_model": 'NP',
    "encoder": 'RNNODE',
    'model_type': 'FNP',
    "in_features":1,
    "out_features":1,
    "encoder_hidden_dim": 64,
    "encoder_embedding_dim": 32,
    "latent_dimension":32,
    "expfunc":'fourier',
    "n_harmonics": 6,
    "n_eig":2,
    "path":'/home/swryu/jylee/data/output/0619/',    #  change this!
    "dataset_path": '/home/swryu/jylee/data/input/',
    #"path": './',
    "lower_bound": 1.0,
    "upper_bound": 6.0,
    "filename": 'RNNODE_FNP_sin',                      #  change this!
    "dataset_type": 'sin',
    "description":'RNNODE+FNP+sin',             # change this!
    "n_epochs":1000000,
    "batch_size":256,
    "device_num" : device
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)