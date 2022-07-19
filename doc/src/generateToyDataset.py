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
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

# add tools path and import our own tools
sys.path.insert(0, '../src')

from const import *
from utils import repr


class DatasetGenerator(object):
    """
    A class used to generate toy datasets with potential missing data. 

    It is understand in this work as the percentage of single number information that is missing in the dataset.
    E.g. if X is of shape 1000 x 3 (1000 subjects and 3 possible featurs per subjects), then a ratio of missing data of 20% mean there is .2x1000x3 = 600 numbers that are missing.
    This view of missing data is number-wise, although it could be subject-wise, group-wise, class-wise, or a mix! 

    Note on the feature management processing:
        - self._X_raw contains the features originally created from sklearn, WITHOUT missing values. 
        - self._X contains the features originally created from sklear, WITH missing values. Call `generate_missing_coordinates` to create.  
        - self._X_train contains the training set, and self._X_test the testing set.
        - self.X_train contains the training set with the missing data potentially encoded or imputed. 

        The data without _ before their name are visible from the outside of this class (ditributions estimation or model training). The other are necessary for internal use. 
        TODOREMOVE self.imp_X contains _X (so with missing values), but with the missing data imputed. Call `impute_missing_data` to create.  

        The methods: 
            - subset(class_used, imputation_mode) allows to change the face of X, (for training, selecting the correct class, or for prediction to allow using imputation, or to encode the NaNs with the DEFAULT_MISSING_VALUE.)

            - reset() reset the self.X features to the _X ones. 
    
    1) Missing Completely at Random (MCAR). 
    
    kwargs:
    ----------
        TODO: bool, default=False
            TODO
        
    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes

    TODO: think of removing self._Z as useless it seems
    """

    
    def __init__(self, 
                dataset_name, 
                num_samples=NUM_SAMPLES,
                imbalance_ratio=IMBALANCE_RATIO, 
                proportion_train=PROPORTION_TRAIN, 
                missing_data_handling=MISSING_DATA_HANDLING,
                imputation_method=DEFAULT_IMPUTATION_METHOD, 
                loading=False, 
                verbosity=1,
                debug=False, 
                random_state=RANDOM_STATE):
        
        self.dataset_name = dataset_name

        if loading:
            #TODOCHECK
            return 

        self.num_samples = num_samples
        
        #self.ratio_of_missing_values = ratio_of_missing_values
        self.imbalance_ratio = imbalance_ratio
        self.proportion_train = proportion_train
        self.missing_data_handling = missing_data_handling
        self.imputation_method = imputation_method
        
        self.missingness_pattern = None
        self.missingness_parameters = {'missingness_mechanism' : None, 
                                        'ratio_of_missing_values' : None,
                                        'missing_X1' : None,
                                        'missing_X2' : None,
                                        'missing_first_quarter' : None,
                                        'ratio_missing_per_class' : None,
                                        'allow_missing_both_coordinates' : None}
        
        
        self.dataset_description = 'Number of samples: {} ({} (#+/#-)'.format(self.num_samples, self.imbalance_ratio)
        self.missingness_description = ''
        self.cmap = sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))
        self.verbosity = verbosity    
        self.debug=debug
        self.random_state = random_state
        np.random.seed(random_state)

        # Generate a dataset with samples from both classes - stored for internal use and for keeping track of changes etc...
        self._X_raw, self._y = self._init_data()

        # Masked data - stored for internal use and for keeping track of changes etc...
        self._X = deepcopy(self._X_raw)

        # Imputed data (depend on the experiences/settings). If state is training, it contains imputation of 
        # the train set, otherwise the test set.
        self._imp_X_train, self._imp_X_test = None, None 

        # Data used by the other classes splitted.
        self.train_index, self.test_index = None, None 
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Prediciton on the test set
        self.y_pred = None


    def __call__(self):
        return repr(self)

    def split_test_train(self):
        """
        Split the dataset given some proportions, taking as input the see-able dataset with potentially missing data, having them either encoded of imputed.
        """
        self.train_index, self.test_index = train_test_split(np.arange(self.num_samples), test_size = (1-PROPORTION_TRAIN))


        self._X_train = deepcopy(self._X[self.train_index])
        self._X_test = deepcopy(self._X[self.test_index])

        self._y_train = deepcopy(self._y[self.train_index])
        self._y_test = deepcopy(self._y[self.test_index])


        # VBy default once the split is done, no matter the state of the dataset, data are pusshed to the forefront.
        self.X_train = deepcopy(self._X_train)
        self.X_test = deepcopy(self._X_test)
        self.y_train = deepcopy(self._y_train)
        self.y_test = deepcopy(self._y_test)

        print("Splitting dataset into test and train set.") if (self.debug or self.verbosity > 1) else None

        # We need to re-empute or encode data after this step, for model where several replicates of experiements are done and 
        # Who re-shuffle the data for each replicates. 
        self.impute_data()
        
        return

    def impute_data(self):
        """
            Must be called after the `split_test_train` method.
            Change the visible variable self.X_, depending on the class, and handling of missing data.

            If the approach are `bayesian` (omputing distributions), at training time (state=='training'), self.X_train will contain 
            the the data from the training set, with potential missing value, for a certain class. 
            At inference time it will contain the test set, with potential missing data, for all classes of course. 
        """

    
        if self.missing_data_handling == 'imputation':

            if self._imp_X_train is None and self._imp_X_test is None:
                self._imp_X_train, self._imp_X_test  = self._impute_missing_data()

            # Create X and y used for experiments
            self.X_train = deepcopy(self._imp_X_train)
            self.y_train = deepcopy(self._y_train)

            self.X_test = deepcopy(self._imp_X_test)
            self.y_test = deepcopy(self._y_test)
            
            print("Imputed {} values (train) and {} (test) using method {}.".format(len(np.isnan(self.X_train)), len(np.isnan(self.X_test)), self.imputation_method)) if (self.debug or self.verbosity > 2)  else None                

        elif self.missing_data_handling == 'encoding':

            self.X_train = deepcopy(self._X_train)
            self.X_train[np.isnan(self.X_train)] = DEFAULT_MISSING_VALUE
            self.y_train = deepcopy(self._y_train)

            self.X_test = deepcopy(self._X_test)
            self.X_test[np.isnan(self.X_test)] = DEFAULT_MISSING_VALUE
            self.y_test = deepcopy(self._y_test)

            print("Encoding {} (train) and {} (test) missing values with {}.".format(len(np.isnan(self._X_train)), len(np.isnan(self._X_test)), DEFAULT_MISSING_VALUE)) if (self.debug or self.verbosity > 2)  else None

        elif self.missing_data_handling == 'without':

            self.X_train = deepcopy(self._X_train)
            self.y_train = deepcopy(self._y_train)
            self.X_test = deepcopy(self._X_test)
            self.y_test = deepcopy(self._y_test)

        else:
            raise ValueError("Please use one of the following missing variables handling: imputation, encoding, or without")

        return 
        
    def change_imputation_approach(self, missing_data_handling, imputation_method):

        previous_imputation_method = self.imputation_method
        previous_missing_data_handling = self.missing_data_handling
        self.missing_data_handling = missing_data_handling
        self.imputation_method = imputation_method
        self.impute_data()

        print("MD handling: {} ({}) --> {} ({}), with changes of the data.".format(previous_missing_data_handling, previous_imputation_method, missing_data_handling, imputation_method)) if (self.debug or self.verbosity > 1)  else None

        return 

    def generate_missing_coordinates(self, missingness_pattern=None, **kwargs):

        """
        Example of the currently implemented Missingness mechanisms and settings.


        TODO: kwargs include: 
        missingness_mechanism='MCAR', ratio_of_missing_values=RATIO_OF_MISSING_VALUES, missing_first_quarter=False, missing_X1=False, missing_X2=False, ratio_missing_per_class=[.1, .5], missingness_pattern=None, verbosity=1)

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
            self._retrieve_missingness_parameters(missingness_pattern, **kwargs)

        else:
            self.missingness_parameters =   {'missingness_mechanism' : kwargs['missingness_mechanism'], 
                                            'ratio_of_missing_values' : kwargs['ratio_of_missing_values'], 
                                            'missing_X1' : kwargs['missing_X1'], 
                                            'missing_X2' : kwargs['missing_X2'], 
                                            'missing_first_quarter' : kwargs['missing_first_quarter'], 
                                            'ratio_missing_per_class' : kwargs['ratio_missing_per_class']}              
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
                    
        elif self.missingness_parameters['missingness_mechanism'] == 'MAR':

            if self.missingness_parameters['missing_first_quarter']: 

                # Making sure that the total amount of missing coordinate does not exceed the threshold
                while not self.met_missingness_rate() or excedded_time < MAX_TRY_MISSSINGNESS:

                    # Simulate missing samples
                    for i in range(self._X.shape[0]):  # randomly remove features

                        if (self.dataset_name!= 'blobs' and self._X_raw[i,1] > 0 and self._X_raw[i,1] > 0) or (self.dataset_name== 'blobs' and self._X_raw[i,1] > 0):

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
                while not self.met_missingness_rate() or excedded_time < MAX_TRY_MISSSINGNESS:
                
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
                while (not self.met_missingness_rate(label=0) or not self.met_missingness_rate(label=1)) and excedded_time < MAX_TRY_MISSSINGNESS:
                    for label in [0, 1]:
                        
                        # Simulate missing samples
                        for i in range(self._X.shape[0]):  # randomly remove features
                            
                            if (self.dataset_name!= 'blobs' and self._y[i]==label and self._X_raw[i,0] > 0 and self._X_raw[i,1] > 0) or (self.dataset_name== 'blobs' and self._y[i]==label and self._X_raw[i,1] > 0):

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
                while (not self.met_missingness_rate(label=0) or not self.met_missingness_rate(label=1)) and excedded_time < MAX_TRY_MISSSINGNESS:

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
        return None

    def _impute_missing_data(self, bandwidth=BANDWIDTH):
        """
            If state is training, we impute the missing data of the training set with itself.
            If state is inference, we impute the missing data of the test set with the training set.
            TODO: veriify that we are doing this once... (the imputation computation), because it's better to store than to recompute everytime.
        """
        from stats import impute_missing_data

        _imp_X_train = impute_missing_data(X_train=self.X_train, X_test=self.X_train, method=self.imputation_method, h=bandwidth)
        _imp_X_test = impute_missing_data(X_train=self.X_train, X_test=self.X_test, method=self.imputation_method, h=bandwidth)

        return _imp_X_train, _imp_X_test
    
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

        elif label is not None and not return_values:

            return np.isnan(self._X[(self._y==label).squeeze()]).sum() /((self._y==label).sum()*missing_dimension) >= self.missingness_parameters['ratio_missing_per_class'][label] 

        if return_values:
            if self.missingness_parameters['missingness_mechanism'] in ['MCAR', 'MAR']:
                return np.isnan(self._X).sum()/(self.num_samples*missing_dimension), None, None
            else:
                return None, np.isnan(self._X[(self._y==0).squeeze()]).sum() /((self._y==0).sum()*missing_dimension), np.isnan(self._X[(self._y==1).squeeze()]).sum() /((self._y==1).sum()*missing_dimension)

    def save(self, experiment_path):

        # Unneccesary to save.

        #-------- Save dataset ----------#
        with open(os.path.join(experiment_path, 'dataset_log.json'), 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.to_json(orient='records') if type(o) == pd.core.frame.DataFrame else o.__dict__)
            
        # Reload the object that were unsaved 

        return
    
    def load(self, dataset_data):

        for key, value in dataset_data.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else: 
                setattr(self, key, value)

    def get_data(self):
        return self.X, self.Z, self.y
        
    def plot(self, ax1=None, ax2=None, title=True, verbosity=1):

        colors = [self.cmap[0] if l==1 else self.cmap[1] for l in self.y_train]

        if ax1 is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            if title:
                fig.suptitle("{}\n{}".format(self.dataset_description, self.missingness_description), weight='bold')
        elif title:
            ax1.set_title(title, weight='bold')

        Z_train = np.isnan(self.X_train[:,0])

        train_df = pd.DataFrame({'X_1': self._X_raw[self.train_index][:,0], 
                                'X_2': self._X_raw[self.train_index][:,1],
                                'Z_1': (~np.isnan(self._X[self.train_index][:,0])).astype(int),
                                'Z_2': (~np.isnan(self._X[self.train_index][:,1])).astype(int)
                                })

        test_df = pd.DataFrame({'X_1':  self._X_raw[self.test_index][:,0], 
                                'X_2': self._X_raw[self.test_index][:,1],
                                'Z_1': (~np.isnan(self._X[self.test_index][:,0])).astype(int),
                                'Z_2': (~np.isnan(self._X[self.test_index][:,1])).astype(int)
                                })

        # Plot the training ans test sets
        ax1.scatter(self._X_train[:,0], self._X_train[:,1], c=colors);ax1.axis('off')

        ax1.scatter(train_df.query(" `Z_1`==0 & `Z_2`==1 ")['X_1'].to_list(), train_df.query(" `Z_1`==0 & `Z_2`==1 ")['X_2'].to_list(), c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing X1 ({})'.format(len(train_df.query(" `Z_1`==0 & `Z_2`==1 "))))
        ax1.scatter(train_df.query(" `Z_1`==1 & `Z_2`==0 ")['X_1'].to_list(), train_df.query(" `Z_1`==1 & `Z_2`==0 ")['X_2'].to_list(), c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing X2 ({})'.format(len(train_df.query(" `Z_1`==1 & `Z_2`==0 "))))
        ax1.scatter(train_df.query(" `Z_1`==0 & `Z_2`==0 ")['X_1'].to_list(), train_df.query(" `Z_1`==0 & `Z_2`==0 ")['X_2'].to_list(), c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing both ({})'.format(len(train_df.query(" `Z_1`==0 & `Z_2`==0 "))))
        ax1.legend(prop={'size':10}, loc='lower left')

        if ax2 is not None: 
            colors = [self.cmap[0] if l==1 else self.cmap[1] for l in self.y_test]

            ax2.scatter(self._X_test[:,0], self._X_test[:,1], c=colors);ax2.axis('off') 
            ax2.scatter(test_df.query(" `Z_1`==0 & `Z_2`==1 ")['X_1'].to_list(), test_df.query(" `Z_1`==0 & `Z_2`==1 ")['X_2'].to_list(), c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing X1 ({})'.format(len(test_df.query(" `Z_1`==0 & `Z_2`==1 "))))
            ax2.scatter(test_df.query(" `Z_1`==1 & `Z_2`==0 ")['X_1'].to_list(), test_df.query(" `Z_1`==1 & `Z_2`==0 ")['X_2'].to_list(), c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing X2 ({})'.format(len(test_df.query(" `Z_1`==1 & `Z_2`==0 "))))
            ax2.scatter(test_df.query(" `Z_1`==0 & `Z_2`==0 ")['X_1'].to_list(), test_df.query(" `Z_1`==0 & `Z_2`==0 ")['X_2'].to_list(), c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing both ({})'.format(len(test_df.query(" `Z_1`==0 & `Z_2`==0 "))))
            ax2.legend(prop={'size':10}, loc='lower left')

        return ax1, ax2

    def _retrieve_missingness_parameters(self, missingness_pattern, ratio_of_missing_values=RATIO_OF_MISSING_VALUES, ratio_missing_per_class=RATIO_MISSING_PER_CLASS):

        self.missingness_pattern = missingness_pattern

        if missingness_pattern==1:
            self.missingness_parameters = {'missingness_mechanism' : 'MCAR', 
                                            'ratio_of_missing_values' : ratio_of_missing_values,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : None,
                                            'ratio_missing_per_class' : None,
                                            'allow_missing_both_coordinates' : None}
            self.missingness_description = 'Pattern 1 - MCAR {} missing, only X1'.format(ratio_of_missing_values)

        elif missingness_pattern==2:
            self.missingness_parameters = {'missingness_mechanism' : 'MAR', 
                                            'ratio_of_missing_values' : ratio_of_missing_values,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : None}

            self.missingness_description = 'Pattern 2 - MAR quarter missing ({}%) both X1,X2'.format(int(100*ratio_of_missing_values))


        elif missingness_pattern==3:
            self.missingness_parameters = {'missingness_mechanism' : 'MAR', 
                                            'ratio_of_missing_values' : ratio_of_missing_values,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : None}

            self.missingness_description = 'Pattern 3 - MAR quarter missing ({}%) only X1'.format(int(100*ratio_of_missing_values))

        elif missingness_pattern==4:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : False,
                                            'ratio_missing_per_class' : ratio_missing_per_class}    

            self.missingness_description = 'Pattern 4 - MNAR ({}% for neg. class {}% for pos. class)'.format(int(100*ratio_missing_per_class[0]), int(100*ratio_missing_per_class[1]))

        elif missingness_pattern==5:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : True,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : [ratio_missing_per_class[0], ratio_missing_per_class[1]]}  

            self.missingness_description = 'Pattern 5 - MNAR Quarter missing\n({}% for neg. class {}% for pos. class)'.format(int(ratio_missing_per_class[0]), int(100*ratio_missing_per_class[1]))

        elif missingness_pattern==6:
            self.missingness_parameters = {'missingness_mechanism' : 'MNAR', 
                                            'ratio_of_missing_values' : None,
                                            'missing_X1' : True,
                                            'missing_X2' : False,
                                            'missing_first_quarter' : True,
                                            'ratio_missing_per_class' : ratio_missing_per_class}  

            self.missingness_description = 'Pattern 4 - MNAR Quarter missing\n({}% for neg. class {}% for pos. class)\n Only X1'.format(int(100*ratio_missing_per_class[0]), int(100*ratio_missing_per_class[1]))



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

            found=False
            while not found:

                X_all, labels = datasets.make_moons(n_samples=int(2*self.imbalance_ratio*(self.num_samples+num_samples_gt)), noise=.15, random_state=self.random_state)

                idx_out = np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )

                if len(idx_out) > 0:
                    idx_out = idx_out.squeeze() if len(idx_out)>1 else idx_out

                    X_all[idx_out], labels[idx_out] = datasets.make_moons(n_samples=len(idx_out), noise=.15)
                    
                if len(np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )) == 0:
                    found=True


        elif self.dataset_name=='circles':

            found=False
            while not found:
                X_all, labels = datasets.make_circles(n_samples=int(2*self.imbalance_ratio*(self.num_samples+num_samples_gt)), factor=.5, noise=.15, random_state=self.random_state)

    
                idx_out = np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )

                if len(idx_out) > 0:
                    idx_out = idx_out.squeeze() if len(idx_out)>1 else idx_out
                    X_all[idx_out], labels[idx_out] = datasets.make_circles(n_samples=len(idx_out), factor=.5, noise=.15, random_state=self.random_state)
                
                if len(np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )) == 0:
                    found=True


        elif self.dataset_name=='blobs':

            found=False
            while not found:
                
                X_all, labels = datasets.make_blobs(n_samples=int(2*self.imbalance_ratio*(self.num_samples+num_samples_gt)), centers=[[-1, 0],[1, 0]], cluster_std=.5, random_state=self.random_state)

                idx_out = np.argwhere( (X_all[:,0]>2.4) | (X_all[:,0] < -2.4) | (X_all[:,1]>2.4) | (X_all[:,1] < -2.4) )

                if len(idx_out) > 0:
                    idx_out = idx_out.squeeze() if len(idx_out)>1 else idx_out

                    X_all[idx_out], labels[idx_out] = datasets.make_blobs(n_samples=len(idx_out), centers=[[-1, 0],[1, 0]],
                                                                        cluster_std=.05)

                if len(np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )) == 0:
                    found=True

        else:
            raise ValueError("Please use 'moons', 'circles', or 'blobs' datasets.")             

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

            found=False
            while not found:

                X_all, labels = datasets.make_moons(n_samples=int(2*(1-self.imbalance_ratio)*(self.num_samples+num_samples_gt)), noise=.15, random_state=self.random_state)

                idx_out = np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )

                if len(idx_out) > 0:
                    idx_out = idx_out.squeeze() if len(idx_out)>1 else idx_out
                    X_all[idx_out], labels[idx_out] = datasets.make_moons(n_samples=len(idx_out), noise=.15)
                
                if len(np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )) == 0:
                    found=True

        elif self.dataset_name=='circles':

            found=False
            while not found:

                X_all, labels = datasets.make_circles(n_samples=int(2*(1-self.imbalance_ratio)*(self.num_samples+num_samples_gt)), factor=.5, noise=.15, random_state=self.random_state)

                idx_out = np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )

                if len(idx_out) > 0:
                    idx_out = idx_out.squeeze() if len(idx_out)>1 else idx_out
                    X_all[idx_out], labels[idx_out] = datasets.make_circles(n_samples=len(idx_out), factor=.5, noise=.15, random_state=self.random_state)
                
                if len(np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )) == 0:
                    found=True



        elif self.dataset_name=='blobs':

            found=False
            while not found:

                X_all, labels = datasets.make_blobs(n_samples=int(2*(1-self.imbalance_ratio)*(self.num_samples+num_samples_gt)), centers=[[-1, 0],[1, 0]], cluster_std=.5, random_state=self.random_state)

                idx_out = np.argwhere( (X_all[:,0]>2.4) | (X_all[:,0] < -2.4) | (X_all[:,1]>2.4) | (X_all[:,1] < -2.4) )

                if len(idx_out) > 0:
                    idx_out = idx_out.squeeze() if len(idx_out)>1 else idx_out
                    X_all[idx_out], labels[idx_out] = datasets.make_blobs(n_samples=len(idx_out), centers=[[-1, 0],[1, 0]],
                                                                        cluster_std=.05, random_state=self.random_state)

                if len(np.argwhere( (X_all[:,0]>2.49) | (X_all[:,0] < -2.49) | (X_all[:,1]>2.49) | (X_all[:,1] < -2.49) )) == 0:
                    found=True

        else:
            raise ValueError("Please use 'moons', 'circles', or 'blobs' datasets.")               

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