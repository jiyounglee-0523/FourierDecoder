import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
PATH = '/home/disentangled_ODE/disentangled_ODE/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "in_features":1,
    "out_features":1,
    "latent_dimension":6,
    "expfunc":'fourier',
    "n_harmonics": 12,
    "n_eig":2,
    "path":'/data/private/generativeODE/galerkin_pretest/dilation_test/',    #  change this!
    #"path": './',
    "lower_bound": 0.9,
    "upper_bound": 2.0,
    "filename": 'finegrain_morecombination_linear_weight',                      #  change this!
    "dataset_type":'dataset7',
    "description":'dataset7 with 12 fixed dilation learn coeffs with more data did not add amps',             # change this!
    "n_epochs":100000,
    "batch_size":1024,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)