import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
    "path":'/data/private/generativeODE/galerkin_pretest/pretest_5/',
    "filename": 'dataset2_one_harmonics_test',                          #  change this!
    "dataset_type":'dataset2',
    "description":'dataset2, relu, one harmonic, given dilation',            # change this!
    "n_epochs":10000,
    "batch_size":1024,
}
TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)