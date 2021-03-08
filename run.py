import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = '/home/disentangled_ODE/disentangled_ODE/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "in_features":1,
    "out_features":1,
    "latent_dimension":2,
    "expfunc":'fourier',
    "n_harmonics":12,
    "n_eig":2,
    "zero_out": True,
    "path":'/data/private/generativeODE/galerkin_pretest/dilation_test/',    #  change this!
    #"path": './',
    "filename": 'finegrain_moredata',                      #  change this!
    "dataset_type":'dataset8',
    "description":'dataset8 with 12 fixed dilation learn coeffs with more data',             # change this!
    "n_epochs":100000,
    "batch_size":1024,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)