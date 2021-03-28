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
    "n_harmonics":12,
    "n_eig":2,
    "zero_out": True,
    "path":'/data/private/generativeODE/galerkin_pretest/dilation_test/',    #  change this!
    #"path": './',
    "filename": 'finegrain_morecombination_fixed_64',                      #  change this!
    "dataset_type":'dataset7',
    "description":'dataset7 with 12 fixed dilation learn coeffs with more data did not add amps',             # change this!
    "n_epochs":100000,
    "batch_size":64,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)