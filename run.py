import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = '/home/generativeode/disentangled_ODE/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "in_features":1,
    "out_features":1,
    "latent_dimension":3,
    "expfunc":'fourier',
    "n_harmonics":1,
    "n_eig":2,
    "zero_out":False,
    "path":'/data/private/generativeODE/galerkin_pretest/pretest_5/',    #  change this!
    "filename": 'initial_data2',                      #  change this!
    "dataset_type":'dataset2',
    "description":'initial_point, zero_out, relu, one harmonic, second test',             # change this!
    "n_epochs":10000,
    "batch_size":1024,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)