import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = '/home/disentangled_ODE/disentangled_ODE/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "test_model": 'NODE',
    "encoder": 'RNNODE',   # Transformer
    "model_type": 'FNODEs',
    "in_features":1,
    "out_features":1,
    "encoder_embedding_dim": 32,
    "encoder_output_dim": 6,
    "latent_dimension":6,
    "expfunc":'fourier',
    "n_harmonics": 5,
    "n_eig":2,
    "path":'/data/private/generativeODE/galerkin_pretest/encoder/',    #  change this!
    #"path": './',
    "lower_bound": 1.0,
    "upper_bound": 5.0,
    "filename": 'dataset7_rnnode_emb32',                      #  change this!
    "dataset_type":'dataset7',
    "description":'dataset7 transformer embedding dimension 32',             # change this!
    "n_epochs":1000000,
    "batch_size":1024,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)