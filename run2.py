import subprocess
import os

# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
PATH = '/home/disentangled_ODE/disentangled_ODE/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "in_features":1,
    "out_features":1,
    "latent_dimension":4,
    "expfunc":'fourier',
    "n_harmonics":2,
    "n_eig":2,
    "path":'/data/private/generativeODE/galerkin_pretest/pretest_6/',
    #"path": './',
    "filename": 'encoder_decoder_dataset7_diagmatrix',                          #  change this!
    "dataset_type":'dataset7',
    "description":'dataset7, encoder decoder fixed true decoder',            # change this!
    "n_epochs":10000,
    "batch_size":2048,
    "zero_out": True,
}
TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)