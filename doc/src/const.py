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
MAX_TRY_MISSSINGNESS = 10
RATIO_OF_MISSING_VALUES = .2
RATIO_MISSING_PER_CLASS = [.1, .3]

# pdf estimation default parameters
RESOLUTION = 50
BANDWIDTH = .2

# Classification default parameters
PROPORTION_TRAIN = .8
DEFAULT_MISSING_VALUE = -5

# Machine parameters
EPSILON = np.finfo(float).eps
RANDOM_STATE = 105


# Colmns of interest in the dataset that recap all the experiments
COLUMNS_DF = ['dataset_name', 'experiment_number','num_samples', 'imbalance_ratio', 'missingness_pattern',
                   'missingness_mechanism', 'ratio_of_missing_values', 'missing_X1',
                   'missing_X2', 'missing_first_quarter', 'ratio_missing_per_class_0','ratio_missing_per_class_1',
                   'Accuracy', 'F1', 'Sensitivity', 'Specificity', 'Precision']




