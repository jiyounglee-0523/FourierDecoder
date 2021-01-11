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
    "n_harmonics":10,
    "n_eig":2,
    "zero_out":True,
    "path":'/data/private/generativeODE/galerkin_pretest/pretest_6/',    #  change this!
    "filename": 'dataset2_dilationdelta',                      #  change this!
    "dataset_type":'dataset2',
    "description":'dataset2 dilation delta zero out',             # change this!
    "n_epochs":10000,
    "batch_size":32,
}
TRAINING_CONFIG_LIST = ["--{}".format(k,v) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)