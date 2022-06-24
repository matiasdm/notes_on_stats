import os
import numpy as np

# Data settings
ROOT_DIR = '/Users/samperochon/Duke/work/notes_on_stats/doc/'
DATA_DIR = '/Users/samperochon/Duke/work/notes_on_stats/doc/src/data'
EXPERIMENT_FOLDER_NAME = 'experiments_24_june'


# DATASET PARAMETERS
DATASET_NAME = 'circles'
NUM_SAMPLES = 5000
IMBALANCE_RATIO = .5

# Missingness default parameters
MISSINGNESS_PATTERN = 3
MAX_TRY_MISSSINGNESS = 10
RATIO_OF_MISSING_VALUES = .25
RATIO_MISSING_PER_CLASS = [.1, .3]

# Handling of issing data in the case of non-robust algorithms, or for experimental purposes.
MISSING_DATA_HANDLING = 'without'
DEFAULT_IMPUTATION_METHOD = 'multi_dimensional_weighting'
DEFAULT_MISSING_VALUE = -5

# pdf estimation default parameters
RESOLUTION = 20
BANDWIDTH = .2

# Classification default parameters
PROPORTION_TRAIN = .8
CLASSIFICATION_THRESHOLD = .5

# Machine parameters
EPSILON = np.finfo(float).eps
RANDOM_STATE = 105


# Neural Additive Model 
NAM_DEFAULT_PARAMETERS = {'num_replicates': 10,
                          'use_missing_indicator_variables': True, 
                        'model': {'num_features': None,
                                'hidden_sizes': [64, 64, 32],
                                'dropout_rate': 0.1,
                                'feature_dropout': 0.05,
                                'use_exu': False},
                        'training': {'regression': False,
                                    'batch_size': 16,
                                    'max_epochs': 20,
                                    'learning_rate': 0.0002,
                                    'weight_decay': 0.0,
                                    'output_penalty': 0.2}
                            }


# Colmns of interest in the dataset that recap all the experiments
COLUMNS_DF = ['dataset_name', 'experiment_number','num_samples', 'imbalance_ratio', 'missingness_pattern',
                   'missingness_mechanism', 'ratio_of_missing_values', 'missing_X1',
                   'missing_X2', 'missing_first_quarter', 'ratio_missing_per_class_0','ratio_missing_per_class_1',
                   'Accuracy', 'F1', 'Sensitivity', 'Specificity', 'Precision']






