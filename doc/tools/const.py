import os
import numpy as np

# Data settings
ROOT_DIR = '/Users/samperochon/Duke/notes_on_stats/doc/'
DATA_DIR = '/Users/samperochon/Duke/notes_on_stats/doc/data'

# DATASET PARAMETERS
DATASET_NAME = 'circles'
NUM_SAMPLES = 10000
IMBALANCE_RATIO = .5

# Missingness default parameters
MISSINGNESS_PATTERN = 3
MAX_TRY_MISSSINGNESS = 100
RATIO_OF_MISSING_VALUES = .2
RATIO_MISSING_PER_CLASS = [.1, .3]

# pdf estimation default parameters
RESOLUTION = 20
BANDWIDTH = .2

# Classification default parameters
PROPORTION_TRAIN = .8

# Machine parameters
EPSILON = np.finfo(float).eps
RANDOM_STATE = 105




