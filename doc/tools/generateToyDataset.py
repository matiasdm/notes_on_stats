import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle 
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import sys 
import os
import json
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

    
    def __init__(self, dataset_name, num_samples=1000, ratio_of_missing_values=RATIO_OF_MISSING_VALUES, imbalance_ratio=IMBALANCE_RATIO, num_samples_gt=2000, fast=False, verbosity=1, debug=False, random_state=RANDOM_STATE):
        
        self.dataset_name = dataset_name

        if fast:
            return 

        self.num_samples = num_samples
        self.num_samples_gt = num_samples_gt
        
        self.ratio_of_missing_values = ratio_of_missing_values
        self.imbalance_ratio = imbalance_ratio
        
        self.missingness_pattern = None
        self.missingness_parameters = {'missingness_mechanism' : None, 
                                        'ratio_of_missing_values' : None,
                                        'missing_X1' : None,
                                        'missing_X2' : None,
                                        'missing_first_quarter' : None,
                                        'ratio_missing_per_class' : None,
                                        'allow_missing_both_coordinates' : None}


        self.mask_missing = None
        
        self.X_gt = None
        self.y_gt = None
        
        self.X_raw = None
        self.X = None
        self.y = None
        
        self.dataset_description = 'Number of samples: {}\n'.format(self.num_samples)
        self.missingness_description = ''
        self.cmap = sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))
        self.verbosity = verbosity    
        self.debug=debug
        self.random_state = random_state
        np.random.seed(random_state)
        
        
        ################################
        # Generate the positive examples
        ################################
        if dataset_name=='moons':
            X_all, labels = datasets.make_moons(n_samples=int(2*imbalance_ratio*(num_samples+num_samples_gt)), noise=.05, random_state=random_state)
        elif dataset_name=='circles':
            X_all, labels = datasets.make_circles(n_samples=int(2*imbalance_ratio*(num_samples+num_samples_gt)), factor=.5, noise=.05, random_state=random_state)
        else:
            raise ValueError("Please use 'moons' or 'circles' datasets.") 
        
        # normalize dataset for easier parameter selection
        X_all = StandardScaler().fit_transform(X_all)

        # Select the positive examples
        X_all = X_all[np.argwhere(labels==1).squeeze()]

        # Separate ground truth and training data
        X_pos, Xgt_pos = X_all[:int(num_samples*imbalance_ratio),:], X_all[int(num_samples*imbalance_ratio):,:]
        labels_pos, labelsgt_pos = 1*np.ones((X_pos.shape[0], 1)), 1*np.ones((Xgt_pos.shape[0], 1))

        ################################
        # Generate the negative examples
        ################################
        if dataset_name=='moons':
            X_all, labels = datasets.make_moons(n_samples=int(2*(1-imbalance_ratio)*(num_samples+num_samples_gt)), noise=.05, random_state=random_state)
        elif dataset_name=='circles':
            X_all, labels = datasets.make_circles(n_samples=int(2*(1-imbalance_ratio)*(num_samples+num_samples_gt)), factor=.5, noise=.05, random_state=random_state)
        else:
            raise ValueError("Please use 'moons' or 'circles' datasets.") 


        # normalize dataset for easier parameter selection
        X_all = StandardScaler().fit_transform(X_all)

        # Select the negative examples
        X_all = X_all[np.argwhere(labels==0).squeeze()]

        # Separate ground truth and training data
        X_neg = X_all[:int(num_samples*(1-imbalance_ratio)),:] 
        Xgt_neg = X_all[int(num_samples*(1-imbalance_ratio)):,:]
        labels_neg, labelsgt_neg = np.zeros((X_neg.shape[0], 1)), np.zeros((Xgt_neg.shape[0], 1))

        # Combine the positive and negative samples
        X, y = np.concatenate([X_neg, X_pos], axis=0), np.concatenate([labels_neg, labels_pos], axis=0)
        X_gt, y_gt = np.concatenate([Xgt_neg, Xgt_pos], axis=0), np.concatenate([labelsgt_neg, labelsgt_pos], axis=0)

        # Shuffle the data 
        self.X_raw, self.y = shuffle(X, y, random_state=random_state)
        self.X_gt, self.y_gt = shuffle(X_gt, y_gt, random_state=random_state)
        
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
        self.X = deepcopy(self.X_raw)

        excedded_time = 0
                
        if self.missingness_parameters['missingness_mechanism'] == 'MCAR':

            # Making sure that the total amount of missing coordinate does not exceed the threshold
            while not self.met_missingness_rate():
    
                # Simulate missing samples
                for i in range(self.X.shape[0]):  # randomly remove features
                    if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:
                        self.X[i,0] = np.nan

                    if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:
                        self.X[i,1] = np.nan

                    if self.met_missingness_rate():
                        break  
                    
            self.mask_missing = np.isnan(self.X)

        elif self.missingness_parameters['missingness_mechanism'] == 'MAR':

            if  self.missingness_parameters['missing_first_quarter']: 

                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate() and excedded_time < MAX_TRY_MISSSINGNESS:

                    # Simulate missing samples
                    for i in range(self.X.shape[0]):  # randomly remove features

                        if self.X_raw[i,0] > 0 and self.X_raw[i,1] > 0:

                            if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                # equal probability
                                self.X[i,0] = np.nan

                            if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                self.X[i,1] = np.nan

                        if self.met_missingness_rate():
                            break  
                    excedded_time+=1
                                        
            else:
                
                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate() and excedded_time < MAX_TRY_MISSSINGNESS:
                
                    for i in range(self.X.shape[0]):  # randomly remove features

                        if np.random.random() < self.missingness_parameters['ratio_of_missing_values']:

                            if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                self.X[i,0] = np.nan

                            if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_of_missing_values']:  
                                self.X[i,1] = np.nan  

                        if self.met_missingness_rate():
                            break  
                    excedded_time+=1
                                                    
        elif self.missingness_parameters['missingness_mechanism'] == 'MNAR':
            
            if self.missingness_parameters['missing_first_quarter']:

                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate(label=0) and not self.met_missingness_rate(label=1) and excedded_time < MAX_TRY_MISSSINGNESS:
                    for label in [0, 1]:
                        
                        # Simulate missing samples
                        for i in range(self.X.shape[0]):  # randomly remove features
                            
                            if self.y[i]==label and self.X_raw[i,0] > 0 and self.X_raw[i,1] > 0:

                                if self.missingness_parameters['missing_X1'] and np.random.random()  < self.missingness_parameters['ratio_missing_per_class'][label]:
                                    # equal probability
                                    self.X[i,0] = np.nan

                                if self.missingness_parameters['missing_X2'] and np.random.random()  < self.missingness_parameters['ratio_missing_per_class'][label]:
                                    self.X[i,1] = np.nan     

                            if self.met_missingness_rate(label=label): 
                                break
                    excedded_time+=1
                                    
            else:
                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate(label=0) and not self.met_missingness_rate(label=1) and excedded_time < MAX_TRY_MISSSINGNESS:
                    for label in [0, 1]:
                        
                            # Simulate missing samples
                            for i in range(self.X.shape[0]):  # randomly remove features

                                if self.y[i]==label:

                                    if self.missingness_parameters['missing_X1'] and np.random.random() < self.missingness_parameters['ratio_missing_per_class'][label]:
                                        self.X[i,0] = np.nan

                                    if self.missingness_parameters['missing_X2'] and np.random.random() < self.missingness_parameters['ratio_missing_per_class'][label]:
                                        self.X[i,1] = np.nan 

                                if self.met_missingness_rate(label=label): 
                                    break  
                    excedded_time+=1

        if excedded_time == MAX_TRY_MISSSINGNESS:
            print("/!\. Missingness constraints were ambitious. Try lower them to reach the desired criteria.")
            self.met_missingness_rate(verbose=True)
        
        self.mask_missing = np.isnan(self.X)
        if verbosity:
            self.plot(verbosity=verbosity)
                    
        return None

    def met_missingness_rate(self, label=None, verbose=False):

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
                print("Ratio of number-wise missing data {:.2f} (thres. {})".format(np.isnan(self.X).sum()/(self.num_samples*missing_dimension), self.missingness_parameters['ratio_of_missing_values']))
            else:
                for label in np.unique(self.y).astype(int):
                    print("Class {} - Ratio of number-wise missing data {:.5f} (thres. {})".format(label, np.isnan(self.X[(self.y==label).squeeze()]).sum() /((self.y==label).sum()*missing_dimension), self.missingness_parameters['ratio_missing_per_class'][label]))
        #while (np.isnan(self.X[(self.y==label).squeeze()]).sum(axis=1) > 0).sum()/(self.y==label).sum() < self.ratio_missing_per_class[label]:
        if label is None:
            return np.isnan(self.X).sum()/(self.num_samples*missing_dimension) >= self.missingness_parameters['ratio_of_missing_values'] 
        else:
            return np.isnan(self.X[(self.y==label).squeeze()]).sum() /((self.y==label).sum()*missing_dimension) >= self.missingness_parameters['ratio_missing_per_class'][label] 

    def save(self, experiment_path):
            
        #-------- Save dataset ----------#
        with open(os.path.join(experiment_path, 'dataset_log.json'), 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.__dict__)
        
    def load(self, dataset_data):

        for key, value in dataset_data.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else: 
                setattr(self, key, value)
    
    def get_data(self):
        return self.X, self.X_gt, self.y, self.y_gt 
        
    def plot(self, verbosity=1):

        colors, colors_gt = [self.cmap[0] if l==1 else self.cmap[1] for l in self.y], [self.cmap[0] if l==1 else self.cmap[1] for l in self.y_gt]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax1.scatter(self.X_gt[:,0], self.X_gt[:,1], c=colors_gt);ax1.axis('off');ax1.set_title("Ground Truth\n{}% imbalance ratio\n".format(int(self.imbalance_ratio*100)), weight='bold')
        ax2.scatter(self.X[:,0], self.X[:,1], c=colors);ax2.axis('off')
        ax2.set_title("{}{}".format(self.dataset_description, self.missingness_description), weight='bold')
        
        if verbosity > 0:
            ax2.scatter(self.X_raw[(self.mask_missing[:,0]) & (~self.mask_missing[:,1]), 0], self.X_raw[(self.mask_missing[:,0]) & (~self.mask_missing[:,1]), 1], c='g' if self.verbosity==4 else 'r', alpha=.7, label='Missing X1 ({})'.format(((self.mask_missing[:,0]) & (~self.mask_missing[:,1])).sum()))
            ax2.scatter(self.X_raw[(~self.mask_missing[:,0]) & (self.mask_missing[:,1]), 0], self.X_raw[(~self.mask_missing[:,0]) & (self.mask_missing[:,1]), 1], c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing X2 ({})'.format(((~self.mask_missing[:,0]) & (self.mask_missing[:,1])).sum()))
            ax2.scatter(self.X_raw[(self.mask_missing[:,0]) & (self.mask_missing[:,1]), 0], self.X_raw[(self.mask_missing[:,0]) & (self.mask_missing[:,1]), 1], c='r', alpha=.7, label='Missing both ({})'.format(((self.mask_missing[:,0]) & (self.mask_missing[:,1])).sum()))
            ax2.legend(prop={'size':10}, loc='lower left')
        plt.show()

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

            self.missingness_description = 'Pattern 2 - MAR quarter missing ({}) both X1,X2'.format(RATIO_OF_MISSING_VALUES)


        elif missingness_pattern==3:
            self.missingness_parameters = {'missingness_mechanism' : 'MAR', 
                                            'ratio_of_missing_values' : RATIO_OF_MISSING_VALUES,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : None}

            self.missingness_description = 'Pattern 3 - MAR quarter missing ({}) only X1'.format(RATIO_OF_MISSING_VALUES)

        elif missingness_pattern==4:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : False,
                                            'ratio_missing_per_class' : RATIO_MISSING_PER_CLASS}    

            self.missingness_description = 'Pattern 4 - MNAR ({} for pos. class {} for neg.class)'.format(RATIO_MISSING_PER_CLASS[0], RATIO_MISSING_PER_CLASS[1])

        elif missingness_pattern==5:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : RATIO_MISSING_PER_CLASS}  

            self.missingness_description = 'Pattern 5 - MNAR Quarter missing\n({} for pos. class {} for neg.class)'.format(RATIO_MISSING_PER_CLASS[0], RATIO_MISSING_PER_CLASS[1])

        elif missingness_pattern==6:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : RATIO_MISSING_PER_CLASS}  

            self.missingness_description = 'Pattern 4 - MNAR Quarter missing\n({} for pos. class {} for neg.class)\n Only X1'.format(RATIO_MISSING_PER_CLASS[0], RATIO_MISSING_PER_CLASS[1])



        elif missingness_pattern=='custom':
            self.missingness_parameters = {'missingness_mechanism' : 'MAR', 
                                            'ratio_of_missing_values' : kwargs['ratio_of_missing_values'],
                                            'missing_X1' : kwargs['missing_X1'],
                                            'missing_X2' : kwargs['missing_X2'],
                                            'missing_first_quarter' : kwargs['missing_first_quarter'],
                                            'ratio_missing_per_class' : kwargs['ratio_missing_per_class'],
                                            'allow_missing_both_coordinates' : kwargs['allow_missing_both_coordinates']}   

            self.missingness_description = 'Custom pattern - {}'.format(kwargs['allow_missing_both_coordinates'])
            
    

