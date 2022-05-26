
import os
import sys
import json
from glob import glob

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Import local packages
from const import *

def fi(x=12, y=12):
    return plt.figure(figsize=(x, y))


def compare_imputation_methods(dataset='None', kernel_bandwidth=.2, num_samples=100, ratio_of_missing_values=.7, imbalance_ratio=.5, resolution=20, methods=None):
    
    h = kernel_bandwidth
    
    # (1) Create toy and ground truth data
    X, Xgt, _, _ = create_dataset(name=dataset, 
                                      num_samples=num_samples, 
                                      ratio_of_missing_values=ratio_of_missing_values, 
                                      imbalance_ratio=imbalance_ratio,
                                      provide_labels=True, 
                                      verbose=True)
    
    print('{} samples created'.format(X.shape[0]))
    plt.figure(figsize=[10,10]); plt.subplot(3,3,1); plt.scatter(X[:,0],X[:,1]); 
    plt.title('Toy data'); plt.xlim(-2.5, 2.5); plt.ylim(2.5, -2.5); 
    plt.xticks(()); plt.yticks(()); plt.axis('equal'); plt.axis('off')

    # Ground truth
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(Xgt)
    xygrid = np.meshgrid(np.linspace(-2.5,2.5,resolution),np.linspace(-2.5,2.5,resolution))
    H,W = xygrid[0].shape
    hat_f = np.zeros_like(xygrid[0])  # init. the pdf estimation
    for i in range(H):
        for j in range(W):
            x = xygrid[0][i,j]
            y = xygrid[1][i,j]
            hat_f[i,j] = np.exp(kde.score_samples([[x,y]]))
    plt.subplot(3,3,2); plt.imshow(hat_f); plt.axis('off'); plt.title('Ground truth')
    hat_fgt = hat_f
    hat_fgt /= hat_fgt.sum()
            
    if methods is None:
        methods = ['naive', 'mean', 'median', 'knn', 'mice', 'our', 'missing']
    for i,method in enumerate(methods):
        hat_f = estimate_pdf(X=X, method=method, resolution=resolution, bandwidth=h)  
        hat_f /= hat_f.sum()
        plt.subplot(3,3,i+3); plt.imshow(hat_f); plt.axis('off');
        l2diff = np.mean( (hat_fgt-hat_f)**2 ); 
        plt.title('{} error {:2.5f}'.format(method,1e6*l2diff))
    
    return


def label_bar(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*(2/4), .5*height,
                 "{:.2f}".format(height),
                 fontsize = 11,
                ha='center', va='bottom')

def create_df():
    
    df = pd.DataFrame(columns = ['dataset_name','experiment_number', 'approach', 'missing_data_handling','imputation_method',
                                'num_samples', 'imbalance_ratio', 'missingness_pattern', 'missingness_mechanism', 
                                'ratio_of_missing_values', 'missing_X1', 'missing_X2', 'missing_first_quarter','ratio_missing_per_class_0', 'ratio_missing_per_class_1','auc',
                                'Accuracy', 'F1', 'MCC', 'Sensitivity', 'Specificity', 'Precision', 'PPV', 'NPV', 'FNR', 'FDR', 'FOR', 
                                'resolution', 'bandwidth', 'estimation_time_0', 'estimation_time_1'])


    experiments_paths = glob(os.path.join(DATA_DIR, 'experiments', "*", '*'))


    for experiment_path in experiments_paths:

        exp_path = os.path.join(experiment_path, 'experiment_log.json')
        dataset_path = os.path.join(experiment_path, 'dataset_log.json')

        dist_None_path = os.path.join(experiment_path, 'distributions_None_log.json')
        dist_1_path = os.path.join(experiment_path, 'distributions_1_log.json')
        dist_0_path = os.path.join(experiment_path, 'distributions_0_log.json')

        dist_0_data, dist_0_data, dist_None_data = None, None, None

        if os.path.isfile(exp_path):

            with open(exp_path) as experiment_json:

                # Load experiment data
                experiment_data = json.load(experiment_json)
        else:
            continue

        if os.path.isfile(dataset_path):
            with open(dataset_path) as data_json:

                # Load experiment data
                dataset_data = json.load(data_json)
        else:
            continue

        if os.path.isfile(dist_None_path):

            with open(dist_None_path) as dist_json:

                # Load experiment data
                dist_None_data = json.load(dist_json)

        if os.path.isfile(dist_1_path):

            with open(dist_1_path) as dist_json:

                # Load experiment data
                dist_1_data = json.load(dist_json)

        if os.path.isfile(dist_0_path):

            with open(dist_0_path) as dist_json:

                # Load experiment data
                dist_0_data = json.load(dist_json)
                
        # append rows to an empty DataFrame
        df = df.append({'dataset_name' : experiment_data['dataset_name'], 
                        'experiment_number' : experiment_data['experiment_number'],  
                        'approach' : experiment_data['approach'],  
                        'missing_data_handling' : experiment_data['missing_data_handling'],  
                        'imputation_method' : experiment_data['imputation_method'],  
                        'num_samples' : dataset_data['num_samples'],  
                        'imbalance_ratio' : dataset_data['imbalance_ratio'],  
                        'missingness_pattern' : int(dataset_data['missingness_pattern']),  
                        'missingness_mechanism' : dataset_data['missingness_parameters']['missingness_mechanism'],  
                        'ratio_of_missing_values' : dataset_data['missingness_parameters']['ratio_of_missing_values'],  
                        'missing_X1' : dataset_data['missingness_parameters']['missing_X1'],  
                        'missing_X2' : dataset_data['missingness_parameters']['missing_X2'],  
                        'missing_first_quarter' : dataset_data['missingness_parameters']['missing_first_quarter'],  
                        'ratio_missing_per_class_0' : dataset_data['missingness_parameters']['ratio_missing_per_class'][0] if dataset_data['missingness_parameters']['ratio_missing_per_class'] is not None else np.nan,
                        'ratio_missing_per_class_1' : dataset_data['missingness_parameters']['ratio_missing_per_class'][1] if dataset_data['missingness_parameters']['ratio_missing_per_class'] is not None else np.nan,
                        'resolution' : experiment_data['resolution'],
                        'bandwidth' : experiment_data['bandwidth'],
                        'auc' : experiment_data['performances_df']['Area Under the Curve (AUC)'][0] if 'Area Under the Curve (AUC)' in experiment_data['performances_df'].keys() else np.nan,
                        'Accuracy' : experiment_data['performances_df']['Accuracy'][0],  
                        'F1' : experiment_data['performances_df']['F1 score (2 PPVxTPR/(PPV+TPR))'][0],  
                        'MCC' : experiment_data['performances_df']['Matthews correlation coefficient (MCC)'][0],  
                        'Sensitivity' : experiment_data['performances_df']['Sensitivity, recall, hit rate, or true positive rate (TPR)'][0],  
                        'Specificity' : experiment_data['performances_df']['Specificity, selectivity or true negative rate (TNR)'][0],  
                        'Precision' : experiment_data['performances_df']['Precision or positive predictive value (PPV)'][0],  
                        'PPV' : experiment_data['performances_df']['Precision or positive predictive value (PPV)'][0],  
                        'NPV' : experiment_data['performances_df']['Negative predictive value (NPV)'][0],  
                        'FNR' : experiment_data['performances_df']['Miss rate or false negative rate (FNR)'][0],  
                        'FDR' : experiment_data['performances_df']['False discovery rate (FDR=1-PPV)'][0],  
                        'FOR' : experiment_data['performances_df']['False omission rate (FOR=1-NPV)'][0],  
                        }, 
                        ignore_index = True)

    df.drop_duplicates(inplace=True)
    df = df.astype({"missingness_pattern": int, "experiment_number": int})
    
    return df


def repr(object_, indent=0):

    import seaborn as sns 
    import numpy as np
    
    if indent==0:
        
        print("{0:10}{1:30}\t {2:40}\t {3:150}".format("","Attribute Name", "type", "Value or first element"))
        print("{0:10}-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n".format(""))
    
    for _ in range(indent):
        print("\t")
    
    if not isinstance(object_, dict):
    
        dict_ = object_.__dict__
    else:
        dict_ = object_
    
    for k, o in dict_.items():
        if type(o) == dict:
            print("{0:10}{1:30}\t {2:40}".format(indent*"\t" if indent > 0 else "", k, _print_correct_type(o)))
            repr(o, indent=indent+1)
        else:
            print("{0:10}{1:30}\t {2:40}\t {3:150}".format(indent*"\t" if indent > 0 else "", k, _print_correct_type(o), _print_correct_sample(o, indent=indent)))
    
    print("\n")
    return 

def _print_correct_sample(o, indent=0):
    """
        This helper function is associated with the show method, used to print properly classes object.
        This one output a string of the element o, taking the type into account.

    """


    
    if o is None:
        return "None"    
    
    elif isinstance(o, (int, float, np.float32, np.float64)):
        return str(o)
    
    elif isinstance(o, str):
        return o.replace('\n', '-') if len(o) < 80 else o.replace('\n', '-')[:80]+'...'
    
    elif isinstance(o, (list, tuple)) :#and not type(o) == sns.palettes._ColorPalette:
        return "{} len: {}".format(str(o[0]), len(o))
    
    elif isinstance(o, np.ndarray) :#and not type(o) == sns.palettes._ColorPalette:
        return "{} shape: {}".format(str(o[0]), str(o.shape))
    
    elif type(o) == dict : 
        return repr(o, indent=indent+1)

    elif type(o) == pd.core.frame.DataFrame:
        return 'dataframe'
    
    else:
        
        return str(o)
    
def _print_correct_type(o):
    """
        This helper function is associated with the show method, used to print properly classes object.
        This one output a string of the type of the element o.

    """
    if o is None:
        return "None"    
    
    elif isinstance(o, int):
        return "int"
    
    elif isinstance(o, float):
        return "float"
    
    elif isinstance(o, np.float32):
        return "np.float32"
    
    elif isinstance(o, np.float64):
        return "np.float64"    
    
    elif isinstance(o, list):
        return "list"
    
    elif isinstance(o, str):
        return "str"
    
    elif isinstance(o, tuple):
        return "tuple"
    
    elif isinstance(o, np.ndarray):
        return "np.ndarray"
    
    elif type(o) == dict : 
        return "dict"
    
    else:
        
        return str(type(o))
        
