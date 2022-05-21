import os
import sys 
import json
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle 

# add tools path and import our own tools
sys.path.insert(0, '../tools')

from const import *


class DatasetGenerator(object):
    """
    A class used to generate toy datasets with potential missing data. 

    It is understand in this work as the percentage of single number information that is missing in the dataset.
    E.g. if X is of shape 1000 x 3 (1000 subjects and 3 possible featurs per subjects), then a ratio of missing data of 20% mean there is .2x1000x3 = 600 numbers that are missing.
    This view of missing data is number-wise, although it could be subject-wise, group-wise, class-wise, or a mix! 
    
    Available missingness mechanisms include:
    
    1) Missing Completely at Random (MCAR). 
    
    kwargs:
    ----------
        missing_both_coordinates: bool, default=False
            Whether the missingness happen on each dimension separately, or jointly. 
            If True, it is possible that both coordinates are missing at the same time, leading to an understanding of 
            the ratio of missing data as a percentage of missing varibale from all numerical value (e.g. a ratio of 20% 
            with this argument raised will inflate the total number of samples having missing data).
        
    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    
    def __init__(self, dataset_name, purpose='train', num_samples=1000, imbalance_ratio=IMBALANCE_RATIO, class_used=None, fast=False, verbosity=1, debug=False, random_state=RANDOM_STATE):
        
        self.dataset_name = dataset_name

        if fast:
            #TODOCHECK
            return 

        self.num_samples = num_samples
        self.class_used = class_used
        self.purpose = purpose
        
        #self.ratio_of_missing_values = ratio_of_missing_values
        self.imbalance_ratio = imbalance_ratio
        
        self.missingness_pattern = None
        self.missingness_parameters = {'missingness_mechanism' : None, 
                                        'ratio_of_missing_values' : None,
                                        'missing_X1' : None,
                                        'missing_X2' : None,
                                        'missing_first_quarter' : None,
                                        'ratio_missing_per_class' : None,
                                        'allow_missing_both_coordinates' : None}
        
        
        self.dataset_description = 'Number of samples: {}'.format(self.num_samples)
        self.missingness_description = ''
        self.cmap = sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))
        self.verbosity = verbosity    
        self.debug=debug
        self.random_state = random_state
        np.random.seed(random_state)

        # Generate a dataset with samples from both classes - stored for internal use and for keeping track of changes etc...
        self._X_raw, self._y = self._init_data()

        # Masked features - stored for internal use and for keeping track of changes etc...
        self._X = deepcopy(self._X_raw)
        # Actual features used 
        self.X = None
        self.y = None

        self._mask_missing = None
        self.mask_missing = None

    def subset(self, class_used):

        if class_used is not None:

            self.class_used = class_used

            # Create X and y used for experiments
            self.X = deepcopy(self._X[(self._y==class_used).squeeze()])
            self.y = deepcopy(self._y[(self._y==class_used).squeeze()])

            # Create masks for the missingness
            self.mask_missing = np.isnan(self.X[(self.y==class_used).squeeze()])

    def reset(self):
        self.X = deepcopy(self._X)
        self.y = deepcopy(self._y)
        self.mask_missing = np.isnan(self.X)
        self.class_used = None
        
    def generate_missing_coordinates(self, missingness_mechanism='MCAR', ratio_of_missing_values=RATIO_OF_MISSING_VALUES, missing_first_quarter=False, missing_X1=False, missing_X2=False, ratio_missing_per_class=[.1, .5], missingness_pattern=None, verbosity=1):

        """
        Example of the currently implemented Missingness mechanisms and settings.

        # No missing data at all. Here for compatibility reasons. 
        self.generate_missing_coordinates(missingness_mechanism='None')
        
        # There are no mutual information between Z and X, Y
        self.generate_missing_coordinates(missingness_mechanism='MCAR')

        # There are mutual information between Z and X (if X_1=0 then Z_2=1 and vice-versa), but not between Z and Y
        self.generate_missing_coordinates(missingness_mechanism='MAR', allow_missing_both_coordinates=False)

        # There are mutual information between Z and X (Z_1 and Z_2 depend on X_1 and X_2), but not between Z and Y
        self.generate_missing_coordinates(missingness_mechanism='MAR', missing_first_quarter=True)

        # There are no mutual information between Z and X, but there are between Z and Y (one class has higher rate of missing value)
        self.generate_missing_coordinates(missingness_mechanism='MNAR', missing_first_quarter=False, ratio_missing_per_class=[.1, .3])

        # There are mutual information between Z and X (Z_1 and Z_2 depend on X_1 and X_2), and between Z and Y (one class has higher rate of missing value)
        self.generate_missing_coordinates(missingness_mechanism='MNAR', missing_first_quarter=True, ratio_missing_per_class=[.1, .3])

        """
        if missingness_pattern is not None:
            self._retrieve_missingness_parameters(missingness_pattern)

        else:
            self.missingness_parameters =   {'missingness_mechanism' : missingness_mechanism, 
                                            'ratio_of_missing_values' : ratio_of_missing_values,
                                            'missing_X1' : missing_X1,
                                            'missing_X2' : missing_X2,
                                            'missing_first_quarter' : missing_first_quarter,
                                            'ratio_missing_per_class' : ratio_missing_per_class}              
        self._X = deepcopy(self._X_raw)

        excedded_time = 0
                
        if self.missingness_parameters['missingness_mechanism'] == 'MCAR':

            # Making sure that the total amount of missing coordinate does not exceed the threshold
            while not self.met_missingness_rate():
    
                # Simulate missing samples
                for i in range(self._X.shape[0]):  # randomly remove features
                    if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:
                        self._X[i,0] = np.nan

                    if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:
                        self._X[i,1] = np.nan

                    if self.met_missingness_rate():
                        break  
                    
            self.mask_missing = np.isnan(self._X)

        elif self.missingness_parameters['missingness_mechanism'] == 'MAR':

            if  self.missingness_parameters['missing_first_quarter']: 

                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate() and excedded_time < MAX_TRY_MISSSINGNESS:

                    # Simulate missing samples
                    for i in range(self._X.shape[0]):  # randomly remove features

                        if self._X_raw[i,0] > 0 and self._X_raw[i,1] > 0:

                            if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                # equal probability
                                self._X[i,0] = np.nan

                            if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                self._X[i,1] = np.nan

                        if self.met_missingness_rate():
                            break  
                    excedded_time+=1
                                        
            else:
                
                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate() and excedded_time < MAX_TRY_MISSSINGNESS:
                
                    for i in range(self.X.shape[0]):  # randomly remove features

                        if np.random.random() < self.missingness_parameters['ratio_of_missing_values']:

                            if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                self._X[i,0] = np.nan

                            if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                self._X[i,1] = np.nan  

                        if self.met_missingness_rate():
                            break  
                    excedded_time+=1
                                                    
        elif self.missingness_parameters['missingness_mechanism'] == 'MNAR':
            
            if self.missingness_parameters['missing_first_quarter']:

                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate(label=0) and not self.met_missingness_rate(label=1) and excedded_time < MAX_TRY_MISSSINGNESS:
                    for label in [0, 1]:
                        
                        # Simulate missing samples
                        for i in range(self._X.shape[0]):  # randomly remove features
                            
                            if self._y[i]==label and self._X_raw[i,0] > 0 and self._X_raw[i,1] > 0:

                                if self.missingness_parameters['missing_X1'] and np.random.random()  < self.missingness_parameters['ratio_missing_per_class'][label]:
                                    # equal probability
                                    self._X[i,0] = np.nan

                                if self.missingness_parameters['missing_X2'] and np.random.random()  < self.missingness_parameters['ratio_missing_per_class'][label]:
                                    self._X[i,1] = np.nan     

                            if self.met_missingness_rate(label=label): 
                                break
                    excedded_time+=1
                                    
            else:
                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate(label=0) and not self.met_missingness_rate(label=1) and excedded_time < MAX_TRY_MISSSINGNESS:
                    for label in [0, 1]:
                        
                            # Simulate missing samples
                            for i in range(self._X.shape[0]):  # randomly remove features

                                if self._y[i]==label:

                                    if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_missing_per_class'][label]:
                                        self._X[i,0] = np.nan

                                    if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_missing_per_class'][label]:
                                        self._X[i,1] = np.nan 

                                if self.met_missingness_rate(label=label): 
                                    break  
                    excedded_time+=1

        if excedded_time == MAX_TRY_MISSSINGNESS:
            print("/!\. Missingness constraints were ambitious. Try lower them to reach the desired criteria.") if self.verbosity > 1 else None
            new_ratio, new_ratio_class_0, new_ratio_class_1 = self.met_missingness_rate(verbose=True if self.verbosity > 1 else False, return_values=True)
            self.missingness_parameters['ratio_of_missing_values'] = new_ratio
            self.missingness_parameters['ratio_missing_per_class'] = [new_ratio_class_0, new_ratio_class_1]

        # Create X and y used for experiments
        self.X = deepcopy(self._X)
        self.y = deepcopy(self._y)

        # Create masks for the missingness
        self._mask_missing = np.isnan(self._X)
        self.mask_missing = np.isnan(self.X)

        if verbosity:
            self.plot(verbosity=verbosity)
                    
        return None

    def met_missingness_rate(self, label=None, verbose=False, return_values=False):

        if self.missingness_parameters['missing_X1'] and self.missingness_parameters['missing_X2']:
            missing_dimension = 2
        elif self.missingness_parameters['missing_X1'] and not self.missingness_parameters['missing_X2']:
            missing_dimension = 1
        elif self.missingness_parameters['missing_X1'] and not self.missingness_parameters['missing_X2']:
            missing_dimension = 1
        else:
            print("/!\. No missing data.")
            return True
        
        if verbose or self.debug:
            if self.missingness_parameters['missingness_mechanism'] in ['MCAR', 'MAR']:
                print("Ratio of number-wise missing data {:.2f} (thres. {})".format(np.isnan(self._X).sum()/(self.num_samples*missing_dimension), self.missingness_parameters['ratio_of_missing_values']))
            else:
                for label in np.unique(self._y).astype(int):
                    print("Class {} - Ratio of number-wise missing data {:.5f} (thres. {})".format(label, np.isnan(self._X[(self._y==label).squeeze()]).sum() /((self._y==label).sum()*missing_dimension), self.missingness_parameters['ratio_missing_per_class'][label]))
        
        if label is None and not return_values:
            return np.isnan(self._X).sum()/(self.num_samples*missing_dimension) >= self.missingness_parameters['ratio_of_missing_values'] 
        elif label is None and not return_values:
            return np.isnan(self._X[(self._y==label).squeeze()]).sum() /((self._y==label).sum()*missing_dimension) >= self.missingness_parameters['ratio_missing_per_class'][label] 


        if return_values:
            if self.missingness_parameters['missingness_mechanism'] in ['MCAR', 'MAR']:
                return np.isnan(self._X).sum()/(self.num_samples*missing_dimension), None, None
            else:
                return None, np.isnan(self._X[(self._y==0).squeeze()]).sum() /((self._y==0).sum()*missing_dimension), np.isnan(self._X[(self._y==1).squeeze()]).sum() /((self._y==1).sum()*missing_dimension)

    def save(self, experiment_path):

        # Store here the objects that cannot be saved as json objects (saved and stored separately)
        mask_missing = self.mask_missing
        _mask_missing = self._mask_missing

        self.mask_missing = None 
        self._mask_missing = None 


        #-------- Save dataset ----------#
        with open(os.path.join(experiment_path, 'dataset_{}_log.json'.format(self.purpose)), 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.to_json(orient='records') if type(o) == pd.core.frame.DataFrame else o.__dict__)
            
        # Reload the object that were unsaved 
        self._mask_missing = _mask_missing
        self.mask_missing = mask_missing

    def load(self, dataset_data):

        for key, value in dataset_data.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else: 
                setattr(self, key, value)

        self._mask_missing = np.isnan(self._X)
        self.mask_missing = np.isnan(self.X)
    
    def get_data(self):
        return self.X, self.y
        
    def plot(self, verbosity=1, ax=None, title=False):
        
            colors = [self.cmap[0] if l==1 else self.cmap[1] for l in self.y]

            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            if title:
                ax.set_title("{}\n{}".format(self.dataset_description, self.missingness_description), weight='bold')
                
            ax.scatter(self.X[:,0], self.X[:,1], c=colors);ax.axis('off')

            if verbosity > 0:
                ax.scatter(self._X_raw[(self._mask_missing[:,0]) & (~self._mask_missing[:,1]) & ((self.class_used is None) | ((self.class_used is not None) & (self._y==self.class_used))).squeeze(), 0], self._X_raw[(self._mask_missing[:,0]) & (~self._mask_missing[:,1]) & ((self.class_used is None) | ((self.class_used is not None) & (self._y==self.class_used))).squeeze(), 1], c='g' if self.verbosity==4 else 'r', alpha=.7, label='Missing X1 ({})'.format(((self._mask_missing[:,0]) & (~self._mask_missing[:,1])).sum()))

                ax.scatter(self._X_raw[(~self._mask_missing[:,0]) & (self._mask_missing[:,1]) & ((self.class_used is None) | ((self.class_used is not None) & (self._y==self.class_used))).squeeze(), 0], self._X_raw[(~self._mask_missing[:,0]) & (self._mask_missing[:,1]) & ((self.class_used is None) | ((self.class_used is not None) & (self._y==self.class_used))).squeeze(), 1], c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing X2 ({})'.format(((~self._mask_missing[:,0]) & (self._mask_missing[:,1])).sum()))

                ax.scatter(self._X_raw[(self._mask_missing[:,0]) & (self._mask_missing[:,1]) & ((self.class_used is None) | ((self.class_used is not None) & (self._y==self.class_used))).squeeze(), 0], self._X_raw[(self._mask_missing[:,0]) & (self._mask_missing[:,1]) & ((self.class_used is None) | ((self.class_used is not None) & (self._y==self.class_used))).squeeze(), 1], c='r', alpha=.7, label='Missing both ({})'.format(((self._mask_missing[:,0]) & (self._mask_missing[:,1])).sum()))
                ax.legend(prop={'size':10}, loc='lower left')
            return ax

    def _retrieve_missingness_parameters(self, missingness_pattern, **kwargs):

        self.missingness_pattern = missingness_pattern

        if missingness_pattern==1:
            self.missingness_parameters = {'missingness_mechanism' : 'MCAR', 
                                            'ratio_of_missing_values' : RATIO_OF_MISSING_VALUES,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : None,
                                            'ratio_missing_per_class' : None,
                                            'allow_missing_both_coordinates' : None}
            self.missingness_description = 'Pattern 1 - MCAR {} missing, only X1'.format(RATIO_OF_MISSING_VALUES)

        elif missingness_pattern==2:
            self.missingness_parameters = {'missingness_mechanism' : 'MAR', 
                                            'ratio_of_missing_values' : RATIO_OF_MISSING_VALUES,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : None}

            self.missingness_description = 'Pattern 2 - MAR quarter missing ({}%) both X1,X2'.format(int(100*RATIO_OF_MISSING_VALUES))


        elif missingness_pattern==3:
            self.missingness_parameters = {'missingness_mechanism' : 'MAR', 
                                            'ratio_of_missing_values' : RATIO_OF_MISSING_VALUES,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : None}

            self.missingness_description = 'Pattern 3 - MAR quarter missing ({}%) only X1'.format(int(100*RATIO_OF_MISSING_VALUES))

        elif missingness_pattern==4:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : False,
                                            'ratio_missing_per_class' : RATIO_MISSING_PER_CLASS}    

            self.missingness_description = 'Pattern 4 - MNAR ({}% for pos. class {}% for neg.class)'.format(int(100*RATIO_MISSING_PER_CLASS[0]), int(100*RATIO_MISSING_PER_CLASS[1]))

        elif missingness_pattern==5:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : RATIO_MISSING_PER_CLASS}  

            self.missingness_description = 'Pattern 5 - MNAR Quarter missing\n({}% for pos. class {}% for neg.class)'.format(int(100*RATIO_MISSING_PER_CLASS[0]), int(100*RATIO_MISSING_PER_CLASS[1]))

        elif missingness_pattern==6:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : RATIO_MISSING_PER_CLASS}  

            self.missingness_description = 'Pattern 4 - MNAR Quarter missing\n({}% for pos. class {}% for neg.class)\n Only X1'.format(int(100*RATIO_MISSING_PER_CLASS[0]), int(100*RATIO_MISSING_PER_CLASS[1]))



        elif missingness_pattern=='custom':
            self.missingness_parameters = {'missingness_mechanism' : 'MAR', 
                                            'ratio_of_missing_values' : kwargs['ratio_of_missing_values'],
                                            'missing_X1' : kwargs['missing_X1'],
                                            'missing_X2' : kwargs['missing_X2'],
                                            'missing_first_quarter' : kwargs['missing_first_quarter'],
                                            'ratio_missing_per_class' : kwargs['ratio_missing_per_class'],
                                            'allow_missing_both_coordinates' : kwargs['allow_missing_both_coordinates']}   

            self.missingness_description = 'Custom pattern - {}'.format(kwargs['allow_missing_both_coordinates'])
            
    def _init_data(self):

        num_samples_gt = 2000

        ################################
        # Generate the positive examples
        ################################
        if self.dataset_name=='moons':
            X_all, labels = datasets.make_moons(n_samples=int(2*self.imbalance_ratio*(self.num_samples+num_samples_gt)), noise=.05, random_state=self.random_state)
        elif self.dataset_name=='circles':
            X_all, labels = datasets.make_circles(n_samples=int(2*self.imbalance_ratio*(self.num_samples+num_samples_gt)), factor=.5, noise=.05, random_state=self.random_state)
        else:
            raise ValueError("Please use 'moons' or 'circles' datasets.") 
        
        # normalize dataset for easier parameter selection
        X_all = StandardScaler().fit_transform(X_all)

        # Select the positive examples
        X_all = X_all[np.argwhere(labels==1).squeeze()]

        # Separate ground truth and training data
        X_pos = X_all[:int(self.num_samples*self.imbalance_ratio),:]
        #Xgt_pos = X_all[int(num_samples*imbalance_ratio):,:]
        labels_pos = 1*np.ones((X_pos.shape[0], 1))
        
        #labelsgt_pos  = 1*np.ones((Xgt_pos.shape[0], 1))

        ################################
        # Generate the negative examples
        ################################
        if self.dataset_name=='moons':
            X_all, labels = datasets.make_moons(n_samples=int(2*(1-self.imbalance_ratio)*(self.num_samples+num_samples_gt)), noise=.05, random_state=self.random_state)
        elif self.dataset_name=='circles':
            X_all, labels = datasets.make_circles(n_samples=int(2*(1-self.imbalance_ratio)*(self.num_samples+num_samples_gt)), factor=.5, noise=.05, random_state=self.random_state)
        else:
            raise ValueError("Please use 'moons' or 'circles' datasets.") 


        # normalize dataset for easier parameter selection
        X_all = StandardScaler().fit_transform(X_all)

        # Select the negative examples
        X_all = X_all[np.argwhere(labels==0).squeeze()]

        # Separate ground truth and training data
        X_neg = X_all[:int(self.num_samples*(1-self.imbalance_ratio)),:] 
        #Xgt_neg = X_all[int(num_samples*(1-imbalance_ratio)):,:]
        labels_neg = np.zeros((X_neg.shape[0], 1))
        #labelsgt_neg = np.zeros((Xgt_neg.shape[0], 1))

        # Combine the positive and negative samples
        X, y = np.concatenate([X_neg, X_pos], axis=0), np.concatenate([labels_neg, labels_pos], axis=0)
        #X_gt, y_gt = np.concatenate([Xgt_neg, Xgt_pos], axis=0), np.concatenate([labelsgt_neg, labelsgt_pos], axis=0)

        # Shuffle the data 
        X_raw, y = shuffle(X, y, random_state=self.random_state)
        #self.X_gt, self.y_gt = shuffle(X_gt, y_gt, random_state=random_state)

        return X_raw, y