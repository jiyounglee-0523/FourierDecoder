import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
PATH = '/home/disentangled_ODE/disentangled_ODE/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "in_features":1,
    "out_features":1,
    "encoder_output_dim": 3,
    "latent_dimension":3,
    "expfunc":'fourier',
    "n_harmonics": 30,
    "n_eig":2,
    "path":'/data/private/generativeODE/galerkin_pretest/ECG/',    #  change this!
    #"path": './',
    "lower_bound": 1.0,
    "upper_bound": 30.0,
    "filename": 'ECG_upto30_cycle1_hz282',                      #  change this!
    "dataset_type":'dataset9',
    "description":'cycle 1 make sure that it duplicates after one sec',             # change this!
    "n_epochs":1000000,
    "batch_size":1,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)