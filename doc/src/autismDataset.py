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
from const_autism import *
from utils import repr


class Dataset(object):
    """
    TODO.
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
                df,
                dataset_name='complete_autism',
                outcome_column='diagnosis',
                features_name = DEFAULT_PREDICTORS,
                missing_data_handling='encoding',
                imputation_method='without',
                proportion_train = PROPORTION_TRAIN,
                verbosity=4,
                debug=False, 
                random_state=RANDOM_STATE):



        self.dataset_name = dataset_name
        self.outcome_column = outcome_column
        self._features_name = features_name
        self.proportion_train = proportion_train

        self.missing_data_handling = missing_data_handling
        self.imputation_method = imputation_method
        
        self.df = self._post_process_df(df)
        self._raw_df = deepcopy(self.df)
        self.num_samples = len(self.df)

        self.verbosity = verbosity    

        # Generate a dataset with samples from both classes - stored for internal use and for keeping track of changes etc...
        self._X, self._y = self._init_data()

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

        self.dataset_description = 'Number of samples: {}'.format(self.num_samples)
        self.cmap = sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))


        self.debug=debug
        self.random_state = random_state
        np.random.seed(random_state)

    def __call__(self):
        return repr(self)

    @property
    def features_name(self):
        return self._features_name

    @features_name.setter  
    def features_name(self, value):
        self._features_name = value 
        self._X, _ = self._init_data()

    def reset(self):
        self.df = deepcopy(self._raw_df)
        return 

    def __repr__(self):
        display(self.df)
        return ''

    def filter(self, administration=None, features=None, validity=None, clinical=None, demographics=None, other=None, verbose=False):
        """
            self.filter(administration={'order': 'first', 
                                        'complete': True}, 
                        clinical={'diagnosis': [0, 1]}, 
                        demographics={'age':[18, 36], 
                                    'sex': 'Male'},
                        features={'having':['gaze', 'touch']},
                        other={'StateOfTheChild':['Slightly irritable', 'In a calm and/or good mood']}
                        )

        """

        if administration is not None:

            if 'completed' in administration.keys():

                indexes_to_drop = self.df[(self.df['validity_available']==1) & (self.df['completed']=='Incomplete (Readminister at next visit)') ].index

                if self.verbosity>1 or verbose:
                    print("Removing {}/{} incomplete administrations.".format(len(indexes_to_drop), len(self.df)))

                self.df.drop(index=indexes_to_drop, inplace=True)

            if 'order' in administration.keys():

                if administration['order'] == 'first':

                    indexes_to_drop = self.df[self.df.duplicated(subset=['id'], keep='first')].index
                    if self.verbosity>1 or verbose:
                        print("Removing {}/{} keeping first admin.".format(len(indexes_to_drop), len(self.df)))

                    self.df.drop(index=indexes_to_drop, inplace=True)
                
                elif administration['order'] == 'last':

                    indexes_to_drop = self.df[self.df.duplicated(subset=['id'], keep='last')].index

                    if self.verbosity>1 or verbose:
                        print("Removing {}/{} keeping last admin.".format(len(indexes_to_drop), len(self.df)))

                    self.df.drop(index=indexes_to_drop, inplace=True)
            
                elif administration['order'] == 'test-retest':
                    
                    indexes_to_drop = df[~df.duplicated(subset=['id'], keep=False)].index

                    if self.verbosity>1 or verbose:
                        print("Removing {}/{} keeping only subject with multiple administrations.".format(len(indexes_to_drop), len(self.df)))

                    self.df.drop(index=indexes_to_drop, inplace=True)

        if clinical is not None:

            if 'diagnosis' in clinical.keys():
                indexes_to_drop = self.df[~self.df['diagnosis'].isin(clinical['diagnosis'])].index

                if self.verbosity>1 or verbose:
                    print("Removing {}/{} keeping only subject with diagnosis: {}.".format(len(indexes_to_drop), len(self.df), clinical['diagnosis']))

            self.df.drop(index=indexes_to_drop, inplace=True)

        if demographics is not None:

            if 'age' in demographics.keys():
                indexes_to_drop =  self.df[~((self.df['age'] >= demographics['age'][0]) & (self.df['age'] < demographics['age'][1]))].index

                if self.verbosity>1 or verbose:
                    print("Removing {}/{} keeping only subject with age between {} and {} mo.".format(len(indexes_to_drop), len(self.df), demographics['age'][0], demographics['age'][1]))

                self.df.drop(index=indexes_to_drop, inplace=True)

            if 'sex' in demographics.keys():
                indexes_to_drop =  self.df[~(self.df['sex'] != demographics['sex'])].index

                if self.verbosity>1 or verbose:
                    print("Removing {}/{} keeping only subject with sex: {}.".format(len(indexes_to_drop), len(self.df), demographics['sex']))
                    
                self.df.drop(index=indexes_to_drop, inplace=True)

        if other is not None:

            for column, admissible_values in other.items():

                indexes_to_drop =  self.df[~self.df[column].isin(admissible_values)].index

                if self.verbosity>1 or verbose:
                    print("Removing {}/{} keeping only subject with column: {} in {}.".format(len(indexes_to_drop), len(self.df), column, admissible_values))
                    
                self.df.drop(index=indexes_to_drop, inplace=True)

        if self.verbosity > 1 or verbose:
            print("{} administrations left.".format(len(self.df)))
            
            display(self.df.groupby(self.outcome_column)[['id']].count())

        self.df.sort_values(by=['id', 'date'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.num_samples = len(self.df)
        self._X, self._y  = self._init_data(verbose=False)

        return

    def split_test_train(self):
        """
        Split the dataset given some proportions, taking as input the see-able dataset with potentially missing data, having them either encoded of imputed.
        """
        if self.proportion_train!=1:
            self.train_index, self.test_index = train_test_split(np.arange(self.num_samples), test_size = (1-self.proportion_train))
        else:
            self.train_index, self.test_index = np.arange(self.num_samples), []

        self._X_train = deepcopy(self._X[self.train_index])
        self._X_test = deepcopy(self._X[self.test_index])

        self._y_train = deepcopy(self._y[self.train_index])
        self._y_test = deepcopy(self._y[self.test_index])


        # By default once the split is done, no matter the state of the dataset, data are pusshed to the forefront.
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
        
    def plot(self):

        self._plot_missing()

        return

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

    def _post_process_df(self, df):

        # encode categorical variables
        df['diagnosis'].replace({'TD':0., 
                                'ASD':1., 
                                'DDLD':2., 
                                'ADHD':3.}, inplace = True)

        df['ethnicity'].replace({'Not Hispanic/Latino':0, 
                                'Hispanic/Latino':1, 
                                 'Unknown or not reported':np.nan}, inplace = True)

        df['race'].replace({'White':0., 
                    'White/Caucasian':0.,
                    'Black/African American':1., 
                    'More than one race':2.,
                    'American Indian/Alaskan Native':3.,
                    'Other':np.nan,
                    'Asian':np.nan,
                    'Unknown or not reported':np.nan,
                    'Unknown/Declined':np.nan,
                   }, inplace = True)

        df['sex'].replace({'M':0, 'F':1}, inplace=True)
        df.replace('N.A', np.nan, inplace=True)
        df.loc[df['SiblingsInTheRoom']==9, 'SiblingsInTheRoom'] = np.nan

        df['StateOfTheChild'].replace({'In a calm and/or good mood':1, 'Slightly irritable':2, 'Somewhat distressed':3, 'Crying and/or tantrum':4}, inplace = True)


        # Merge time information and create `administration_number` column
        df = self._retrieve_administration_timing(df)

        # Merge postural sway variables into social and non-social 
        df = self._compute_cva_condensed_variables(df)

        # Sort df
        df.sort_values(by=['id', 'date'], inplace=True)

        return df 
       
    def _init_data(self, verbose=None):

        y = self.df[self.outcome_column].to_numpy().astype(float)
        X_raw = self.df[self.features_name].to_numpy().astype(float)

        if self.verbosity>1 and verbose != False:
            print("Predicting {} based on {} features".format(self.outcome_column, len(self.features_name)))
        return X_raw, y

    def _retrieve_administration_timing(self, df):

        df.loc[df['time'].isna(), 'time'] = '00:00'
        df['date'] = pd.to_datetime(df["date"]+' '+df['time'])

        del df['time']

        df['administration_number'] = np.nan

        for i, d in df.sort_values(by=['date']).groupby('id'):
            if len(d) == 1:
                df.loc[d.index, 'administration_number'] = 1
                
            else:
                df.loc[d.index, 'administration_number'] = np.arange(1, len(d)+1).astype(int)

        return df

    def _compute_cva_condensed_variables(self, df):
        '''
            Merge postural sway variables into social and non-social 
        '''


        S_postural_sway = df[['ST_postural_sway', 'BB_postural_sway', 'RT_postural_sway', 'MML_postural_sway', 'FP_postural_sway']].mean(axis=1)
        NS_postural_sway = df[['DIGC_postural_sway', 'DIGRRL_postural_sway', 'FB_postural_sway', 'MP_postural_sway']].mean(axis=1)
        df['S_postural_sway'] = S_postural_sway
        df['NS_postural_sway'] = NS_postural_sway

        # Merge silhouette scores
        gaze_silhouette_score = df[['BB_gaze_silhouette_score','S_gaze_silhouette_score','FP_gaze_silhouette_score']].mean(axis=1)
        df['gaze_silhouette_score'] = gaze_silhouette_score

        # Merge percent right 
        inv_S_gaze_percent_right = 1-df['S_gaze_percent_right']
        df['inv_S_gaze_percent_right'] = inv_S_gaze_percent_right
        mean_gaze_percent_right = df[['BB_gaze_percent_right', 'inv_S_gaze_percent_right']].mean(axis=1)
        df['mean_gaze_percent_right'] = mean_gaze_percent_right
        return df

    def _plot_missing(self):
        if self.df.isnull().sum().sum() != 0:
            na_df = (self.df.isnull().sum() / len(self.df)) * 100      
            na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
            missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
            missing_data.plot(kind = "bar", figsize=(30, 8))
            plt.title("Percentage of values missing (the higher the more missing)", weight='bold', fontsize=18)
            plt.show()
        else:
            print('No NAs found')