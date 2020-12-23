import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = '/home/generativeode/generative_ODE_2/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "in_features":1,
    "out_features":1,
    "latent_dimension":3,
    "expfunc":'fourier',
    "n_harmonics":1,
    "n_eig":2,
    # "zero_out":True,
    "path":'/data/private/generativeODE/galerkin_pretest/pretest_5/',
    "filename": 'dataset3_one_harmonics_zero_out',
    "dataset_type":'dataset3',
    "description":'dataset3, zero_out, relu, one harmonic',
    "n_epochs":10000,
    "batch_size":1024,
}
TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)