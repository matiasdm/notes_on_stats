
import os
import sys
import json
from glob import glob
from copy import deepcopy

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Import local packages
from const import *
sys.path.insert(0, '../../src')


def fi(x=12, y=12):
    return plt.figure(figsize=(x, y))

def create_dataset(name, num_samples=10, ratio_of_missing_values=.5, imbalance_ratio=.5, provide_labels=True, only_one_missing=True, verbose=True):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle 
    
    n = num_samples + 2000  # we generate num_samples for testing and 2k as ground truth 

    ################################
    # Generate the positive examples
    ################################
    if name=='moons':
        data = datasets.make_moons(n_samples=int(2*imbalance_ratio*n), noise=.05)
    elif name=='circles':
        data = datasets.make_circles(n_samples=int(2*imbalance_ratio*n), factor=.5, noise=.05)
    elif name=='blobs':
        data = datasets.make_blobs(n_samples=n, random_state=8)
    else:
        raise ValueError("Please use 'moons', 'circles', or 'blobs' datasets.") 
        
    X_all, labels = data  # keep the 2D samples 

    # normalize dataset for easier parameter selection
    X_all = StandardScaler().fit_transform(X_all)

    # Select the positive examples
    X_all = X_all[np.argwhere(labels==1).squeeze()]

    # Separate ground truth and training data
    X_pos = X_all[:int(num_samples*imbalance_ratio),:] 
    Xgt_pos = X_all[int(num_samples*imbalance_ratio):,:]
    labels_pos, labelsgt_pos = 1*np.ones((X_pos.shape[0], 1)), 1*np.ones((Xgt_pos.shape[0], 1))

    ################################
    # Generate the negative examples
    ################################
    if name=='moons':
        data = datasets.make_moons(n_samples=int(2*(1-imbalance_ratio)*n), noise=.05)
    elif name=='circles':
        data = datasets.make_circles(n_samples=int(2*(1-imbalance_ratio)*n), factor=.5, noise=.05)
    else:
        raise ValueError("Please use 'moons' or 'circles' datasets.") 
    
    X_all, labels = data  # keep the 2D samples 

    # normalize dataset for easier parameter selection
    X_all = StandardScaler().fit_transform(X_all)

    # Select the negative examples
    X_all = X_all[np.argwhere(labels==0).squeeze()]

    # Separate ground truth and training data
    X_neg = X_all[:int(num_samples*(1-imbalance_ratio)),:] 
    Xgt_neg = X_all[int(num_samples*(1-imbalance_ratio)):,:]
    labels_neg, labelsgt_neg = np.zeros((X_neg.shape[0], 1)), np.zeros((Xgt_neg.shape[0], 1))

    # Combine the positive and negative samples
    X, labels = np.concatenate([X_neg, X_pos], axis=0), np.concatenate([labels_neg, labels_pos], axis=0)
    Xgt, labelsgt = np.concatenate([Xgt_neg, Xgt_pos], axis=0), np.concatenate([labelsgt_neg, labelsgt_pos], axis=0)

    # Shuffle the data 
    X, labels = shuffle(X, labels, random_state=0)
    Xgt, labelsgt = shuffle(Xgt, labelsgt, random_state=0)
    
    
    if only_one_missing:
        # Simulate missing samples
        for i in range(X.shape[0]):  # randomtly remove features
            if np.random.random() < ratio_of_missing_values:
                if np.random.random() < .5:  # remove samples from x or y with 
                    # equal probability
                    X[i,0] = np.nan
                else:
                    X[i,1] = np.nan
    else:
        # Simulate missing samples
        for i in range(X.shape[0]):  # randomtly remove features
            if np.random.random() < ratio_of_missing_values:
                X[i,0] = np.nan

            if np.random.random() < ratio_of_missing_values:
                X[i,1] = np.nan

    if verbose:
        import seaborn as sns
        color = plt.get_cmap('tab20')(np.arange(0,2)); cmap = sns.color_palette(color)
        colors, colorsgt = [cmap[0] if l==1 else cmap[1] for l in labels], [cmap[0] if l==1 else cmap[1] for l in labelsgt]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax1.scatter(Xgt[:,0], Xgt[:,1], c=colorsgt);ax1.axis('off');ax1.set_title("Ground Truth\n{}% imbalance ratio\n".format(int(imbalance_ratio*100)), weight='bold')
        ax2.scatter(X[:,0], X[:,1], c=colors);ax2.axis('off');ax2.set_title("Created samples\n{}% imbalance ratio\n{} % missing data".format( int(imbalance_ratio*100),int(ratio_of_missing_values*100)), weight='bold')

    if provide_labels: 
        return X, Xgt, labels.squeeze(), labelsgt.squeeze()
    else:
        return X, Xgt


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
    plt.title('Toy data'); plt.xlim(-3, 3); plt.ylim(2.5, -2.5); 
    plt.xticks(()); plt.yticks(()); plt.axis('equal'); plt.axis('off')

    # Ground truth
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(Xgt)
    xygrid = np.meshgrid(np.linspace(-3, 3,resolution),np.linspace(-3, 3,resolution))
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
        methods = ['naive', 'mean', 'median', 'knn', 'mice', 'our', 'multi_distributions']
    for i,method in enumerate(methods):
        hat_f = estimate_pdf_TODO(X=X, method=method, resolution=resolution, bandwidth=h)  
        hat_f /= hat_f.sum()
        plt.subplot(3,3,i+3); plt.imshow(hat_f); plt.axis('off');
        l2diff = np.mean( (hat_fgt-hat_f)**2 ); 
        plt.title('{} error {:2.5f}'.format(method,1e6*l2diff))
    
    return


def estimate_pdf_TODO(X=None, method='multi_distributions', resolution=20, bandwidth=None):
    
    xygrid = np.meshgrid(np.linspace(-3, 3,resolution),np.linspace(-3, 3,resolution))
    H,W = xygrid[0].shape
    hat_f = np.zeros_like(xygrid[0])  # init. the pdf estimation
    h = bandwidth


    if method=='our':
        # See documentation
        from model.bayesian.stats import kernel_based_pdf_estimation
    
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(X=X, x=[x,y], h=h)

    elif method=='naive':
        # Ignore missing values
        from model.bayesian.stats import kernel_based_pdf_estimation
        imp_X = X[~np.isnan(X[:,0]),:]
        imp_X = imp_X[~np.isnan(imp_X[:,1]),:]        
                
    elif method=='mean':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_X = imp.fit_transform(X)
    
    elif method=='median':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_X = imp.fit_transform(X)
    
    elif method=='knn':
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer()
        imp_X = knn_imputer.fit_transform(X)
        
    elif method=='mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
        imp_X = imp.fit_transform(X)

    elif method=='multi_distributions':


        #----------------------------------------------------------------------------------
        #  Estimation of f(X_1,X_2|Z_1=1, Z_2=1), f(X_2|Z_1=0,Z_2=1) and f(X_1|Z_1=1,Z_2=0)
        #----------------------------------------------------------------------------------

        from model.bayesian.stats import kernel_based_pdf_estimation_side_spaces

        hat_f_0 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_1 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_2 = np.zeros_like(xygrid[0])  # init. the pdf estimation

        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                # Computing contribution on coordinates i, j of hat_f, and coordinate i of hat_f_1 and coordinate j of hat_f_2
                hat_f[i,j], hat_f_0[i,j], hat_f_1[i,j], hat_f_2[i,j] =  kernel_based_pdf_estimation_side_spaces(X=X, x=[x, y], h=h)
                
        # Average the contribution of all i's and j's coordinate
        hat_f_0 = np.mean(hat_f_0)

        # Average the contribution of all j's coordinate on this horizontal line
        hat_f_1 = np.mean(hat_f_1, axis=0)
        
        # Average the contribution of all i's coordinate to form the vertical line
        hat_f_2 = np.mean(hat_f_2, axis=1) 

        # Normalization of the distributions
        hat_f /= (hat_f.sum()+EPSILON);hat_f_1 /= (hat_f_1.sum()+EPSILON);hat_f_2 /= (hat_f_2.sum()+EPSILON)

        
    
    if method in ['mice', 'knn', 'median', 'mean', 'naive']:
        from model.bayesian.stats import kernel_based_pdf_estimation
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(imp_X, x=[x,y],h=h)


    return hat_f



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

def check_experiment_already_done(df, verbose=False,return_df=False, **kwargs):
    
    narrowed_df=deepcopy(df)
    if verbose:
        print(len(narrowed_df)) 
    
    for key, value in kwargs.items():
        
        if key=='ratio_missing_per_class':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0'].isnull()]
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0']==value[0]]
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1']==value[1]]    
        elif key == 'use_missing_indicator_variables':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['use_missing_indicator_variables'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['use_missing_indicator_variables']==value]
                
                
        elif key == 'ratio_of_missing_values':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_of_missing_values'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_of_missing_values']==value]
                
                
        elif key == 'ratio_missing_per_class_0':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_0']==value]
                
        elif key == 'ratio_missing_per_class_1':
            if value is None:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1'].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df['ratio_missing_per_class_1']==value]   
                
                
        else:
        
            narrowed_df = narrowed_df[narrowed_df[key]==value]
            
        print(len(narrowed_df), key, value) if verbose else None
        
    if not return_df:
        
        return len(narrowed_df) > 0
    
    else:
        
        return narrowed_df
        


def create_df(folder_names=EXPERIMENT_FOLDER_NAME):

    if not isinstance(folder_names, list):
        folder_names = list(folder_names)

    
    df = pd.DataFrame(columns = ['dataset_name','experiment_number', 'approach', 'missing_data_handling','imputation_method', 'use_missing_indicator_variables', 
                                'num_samples', 'imbalance_ratio', 'missingness_pattern', 'missingness_mechanism', 
                                'ratio_of_missing_values', 'missing_X1', 'missing_X2', 'missing_first_quarter','ratio_missing_per_class_0', 'ratio_missing_per_class_1','auc',
                                'Accuracy', 'F1', 'MCC', 'Sensitivity', 'Specificity', 'Precision', 'PPV', 'NPV', 'FNR', 'FDR', 'FOR', 
                                'resolution', 'bandwidth', 'estimation_time_0', 'estimation_time_1'])

    
    experiments_paths = []
    for folder_name in folder_names:
        experiments_paths.extend(glob(os.path.join(DATA_DIR, folder_name, "*", '*')))


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
                        'missing_data_handling' : dataset_data['missing_data_handling'],  
                        'imputation_method' : dataset_data['imputation_method'],  
                        'use_missing_indicator_variables': experiment_data['use_missing_indicator_variables'], #if 'use_missing_indicator_variables' in experiment_data.keys() else None,   # TODO 
                        'num_samples' : dataset_data['num_samples'],  
                        'imbalance_ratio' : dataset_data['imbalance_ratio'],  
                        'missingness_pattern' : int(dataset_data['missingness_pattern']),  
                        'missingness_mechanism' : dataset_data['missingness_parameters']['missingness_mechanism'],  
                        'ratio_of_missing_values' : dataset_data['missingness_parameters']['ratio_of_missing_values'],  
                        'missing_X1' : dataset_data['missingness_parameters']['missing_X1'],  
                        'missing_X2' : dataset_data['missingness_parameters']['missing_X2'],  
                        'missing_first_quarter' : dataset_data['missingness_parameters']['missing_first_quarter'],  
                        'ratio_missing_per_class_0' : dataset_data['missingness_parameters']['ratio_missing_per_class'][0] if dataset_data['missingness_parameters']['ratio_missing_per_class'] is not None else None,
                        'ratio_missing_per_class_1' : dataset_data['missingness_parameters']['ratio_missing_per_class'][1] if dataset_data['missingness_parameters']['ratio_missing_per_class'] is not None else None,
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
        
    #df['ratio_missing_per_class_0'] = df['ratio_missing_per_class_0'].astype(float).round(2)
    #df['ratio_missing_per_class_1'] = df['ratio_missing_per_class_1'].astype(float).round(2)
    #df['ratio_of_missing_values'] = df['ratio_of_missing_values'].astype(float).round(2)
    

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
        
