import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
PATH = '/home/disentangled_ODE/disentangled_ODE/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "in_features":1,
    "out_features":1,
    "latent_dimension":6,
    "expfunc":'fourier',
    "n_harmonics": 5,
    "n_eig":2,
    "path":'/data/private/generativeODE/galerkin_pretest/dilation_test/',    #  change this!
    #"path": './',
    "lower_bound": 1.0,
    "upper_bound": 5.0,
    "filename": 'synthetic_encdec_upto5',                      #  change this!
    "dataset_type":'dataset7',
    "description":'encoder: RNNODE, decoder=ODE, upto 5, 3sin 3cos',             # change this!
    "n_epochs":1000000,
    "batch_size":1024,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)