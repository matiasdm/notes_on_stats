import os
import sys
import json 

from glob import glob
from tqdm import tqdm
from time import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, plot_roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from prg import prg

from xgboost import XGBClassifier, plot_importance, plot_tree
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

import torch


# add tools path and import our own tools
sys.path.insert(0, '../src')

from const import *
from const_autism import *
from utils import fi, repr
from metrics import f1score, average_precision, bestf1score, calc_auprg, create_prg_curve

from generateToyDataset import DatasetGenerator
from model.bayesian.distributions import Distributions

from model.neural_additive_models.nam import NAM
from model.neural_additive_models.visualization import plot_roc_curves_nam, plot_shape_functions
from model.neural_additive_models.utils import train_model, eval_model
from model.neural_additive_models.dataset import TabularData

from model.xgboost.visualization import plot_roc_curves_xgboost


class Experiments(object):

    def __init__(self, 
                dataset_name, 
                dataset=None, 
                purpose='classification', 
                approach='multi_distributions', 
                resolution=RESOLUTION, 
                bandwidth=BANDWIDTH, 
                sampling_method=DEFAULT_SAMPLING_METHOD,
                previous_experiment=None,        
                save_experiment=True, 
                verbosity=1, 
                debug=False, 
                random_state=RANDOM_STATE):

        # Set definitions attributes (also used for log purposes)
        self.dataset_name = dataset_name
        self.debug=debug
        self.verbosity=verbosity

        # Dataset 
        self.dataset = dataset
        self.sampling_method = sampling_method

        # Context of the experiment
        self.purpose = purpose
        self.approach = approach
        self.save_experiment = save_experiment

        # Beyesian-related attributes
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.num_cv = None
        self.estimation_time = None
        self.dist = None
        self.dist_pos = None
        self.dist_neg = None

        # NAMs and XGboost related attributs
        self._init_model()
        self.fitted = False

        # Interesting byproducts
        self.predictions_df = None
        self.performances_df = None

        if previous_experiment is not None:
            self.load(previous_experiment)
            return

        # Create experiment folder
        if self.save_experiment:
            self.experiment_number, self.experiment_path, self.json_path = self._init_experiment_path()
        else:
            self.experiment_number, self.experiment_path, self.json_path = -1, None, None

        self.description = '({}) Dataset name {}\nApproach:{} Imputation technics: {}'.format(self.experiment_number, self.dataset_name, self.approach, self.dataset.imputation_method)

        # Define colors, level of verbosity, and random_state
        self.verbosity = verbosity 
        self.debug=debug
        self.random_state = random_state

    def __call__(self):
        return repr(self)       
    
    @property
    def features_name(self):
        return self.dataset._features_name

    @features_name.setter  
    def features_name(self, features_name):
        self.dataset.features_name = features_name
    
    def fit_predict(self, **kwargs):

        """ 
            This function fit the model and predict score.
            It generates the `predictions_df` attribute, that predict score on all the dataset.

        """

        self.dataset.impute_data()

        if self.verbosity >1:
            print("Predicting {} based on {} features using {} approach.".format(self.dataset.outcome_column, len(self.dataset.features_name), self.approach))
            print(*self.dataset.features_name, sep='\n')
            
        t0 = time()

        if self.approach in ['single_distribution', 'multi_distributions']:

            if self.purpose == 'classification':
            
                # Estimate distributions
                self.dist_pos.estimate_pdf(resolution=self.resolution, bandwidth=self.bandwidth)
                self.dist_neg.estimate_pdf(resolution=self.resolution, bandwidth=self.bandwidth)
            
            
            elif self.purpose == 'estimation':

                # Estimate distributions
                self.dist.estimate_pdf(resolution=self.resolution, bandwidth=self.bandwidth)

            # Do the prediction for the test set.
            self._predict_map()

        elif self.approach == 'nam':

            # Be careful this one is containing all the replicates, if you want to access results on a test set 
            # with the best model, you should call `self.predict` and y_pred and proper coordinates will be accessible
            # Through self.dataset.y_pred. Note also this is called when plotting! 
            self.predictions_df = self._train_nam(**kwargs)

        elif self.approach == 'xgboost':

            self.predictions_df = self._fit_predict_vanilla(**kwargs)

        elif self.approach == 'ebm':

            self.predictions_df = self._fit_predict_vanilla(**kwargs)

        elif self.approach == 'DecisionTree':

            self.predictions_df = self._fit_predict_vanilla(**kwargs)

        elif self.approach == 'LogisticRegression':

            self.predictions_df = self._fit_predict_vanilla(**kwargs)
            
        self.estimation_time = time() - t0

        self._performances()
            
        self.fitted = True

        return 

    def plot(self,  *args, **kwargs):

        display(self.performances_df)

        if self.approach in ['single_distribution', 'multi_distributions']:

            if self.purpose == 'classification':

                self._plot_classification(*args, **kwargs)

            elif self.purpose == 'estimation':

                self._plot_estimation()

        elif self.approach == 'nam':
            
            self._plot_nam()
            
        elif self.approach == 'xgboost':
            
            self._plot_xgboost()    

        elif self.approach == 'ebm':
            self._plot_ebm()   

        elif self.approach in ['DecisionTree', 'LogisticRegression']:
            self._plot_lr_decision_tree()
            print("Not implemented yet.")

        return

    def save(self):
        
        #-------- Save dataset ----------#
        self.dataset.save(experiment_path=self.experiment_path)

        if self.approach in ['single_distribution', 'multi_distributions']:

            #-------- Save Distributions ----------#
            self.dist_pos.save(experiment_path=self.experiment_path)
            self.dist_neg.save(experiment_path=self.experiment_path)

        #-------- Save json information ----------#

        # Store here the objects that cannot be saved as json objects (saved and stored separately)
        dataset = self.dataset
        dist_pos = self.dist_pos
        dist_neg = self.dist_neg
        model = self.model if hasattr(self, 'model') else None #TODO MAJOR
        performances_df = self.performances_df
        predictions_df = self.predictions_df

        self.dataset = None 
        self.dist_pos = None
        self.dist_neg = None
        self.model = None

        if self.approach in ['single_distribution', 'multi_distributions']:
            self.predictions_df = None

        # TODOMAJOR: deal both the same way!!!!
        self.performances_df = self.performances_df.to_dict(orient='list') if self.performances_df is not None else None

        with open(self.json_path, 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.to_json(orient='records') if type(o) == pd.core.frame.DataFrame else o.__dict__) 

        # Reload the object that were unsaved 
        self.dataset = dataset
        self.dist_pos = dist_pos
        self.dist_neg = dist_neg
        self.model = model
        self.performances_df = performances_df
        self.predictions_df = predictions_df

        return    
    
    def load(self, previous_experiment=None):
        """
            This functions aims at retrieving the best model from all the experiments. 
            You can either predefince the experiment number and the epoch, or look over all the losses of all experiments and pick the model having the best performances.
            
            /!\ Not implemented yet  
        """

        experiment_path = os.path.join(DATA_DIR,  EXPERIMENT_FOLDER_NAME, self.dataset_name, str(previous_experiment), 'experiment_log.json')
        dataset_path = os.path.join(DATA_DIR,  EXPERIMENT_FOLDER_NAME, self.dataset_name, str(previous_experiment), 'dataset_log.json')
        dist_None_path = os.path.join(DATA_DIR,  EXPERIMENT_FOLDER_NAME, self.dataset_name, str(previous_experiment), 'distributions_None_log.json')
        dist_pos_path = os.path.join(DATA_DIR,  EXPERIMENT_FOLDER_NAME, self.dataset_name, str(previous_experiment), 'distributions_1_log.json')
        dist_neg_path = os.path.join(DATA_DIR,  EXPERIMENT_FOLDER_NAME, self.dataset_name, str(previous_experiment), 'distributions_0_log.json')
        model_path = os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME, self.dataset_name, str(previous_experiment), 'best_model.pt')


        #---------------- Loading Experiment  ----------------#

        if os.path.isfile(experiment_path):

            with open(experiment_path) as experiment_json:

                # Load experiment data
                experiment_data = json.load(experiment_json)
                
                # Load experiment attributes
                self._load(experiment_data)
            print("Loaded experiment at '{}'".format(experiment_path)) if (self.debug or self.verbosity > 1)  else None
        else:
            print("/!\ No previous experiment found at '{}'".format(experiment_path)) if (self.debug or self.verbosity > 1)  else None


        #---------------- Loading Dataset   ----------------#


        if os.path.isfile(dataset_path):
    
            with open(dataset_path) as data_json:

                # Load experiment data
                dataset_data = json.load(data_json)

                self.dataset = DatasetGenerator(dataset_name=dataset_data['dataset_name'], loading=True)
                
                # Load experiment attributes
                self.dataset.load(dataset_data)
            print("Loaded dataset at '{}'".format(dataset_path)) if (self.debug or self.verbosity > 1)  else None
        else:
            print("/!\ No previous dataset found at '{}'".format(dataset_path)) if (self.debug or self.verbosity > 1)  else None


        #---------------- Loading none dist   ----------------#


        if os.path.isfile(dist_None_path):

            with open(dist_None_path) as dist_json:

                # Load experiment data
                dist_data = json.load(dist_json)

                self.dist = Distributions(dataset=self.dataset)
                
                # Load experiment attributes
                self.dist.load(dist_data)

            print("Loaded associated distribution with '{}'".format(dist_None_path)) if (self.debug or self.verbosity > 1)  else None
        else:
            print("/!\ No previous computed distribution found at '{}'".format(dist_None_path)) if (self.debug or self.verbosity > 1)  else None

        #---------------- Loading Pos dist   ----------------#

        if os.path.isfile(dist_pos_path):

            with open(dist_pos_path) as dist_json:

                # Load experiment data
                dist_data = json.load(dist_json)

                self.dist_pos = Distributions(dataset=self.dataset)
                
                # Load experiment attributes
                self.dist_pos.load(dist_data)

            print("Loaded associated distribution with '{}'".format(dist_pos_path)) if (self.debug or self.verbosity > 1)  else None
        else:
            print("/!\ No previous computed distribution found at '{}'".format(dist_pos_path)) if (self.debug or self.verbosity > 1)  else None

        #---------------- Loading Neg dist   ----------------#

        if os.path.isfile(dist_neg_path):
    
            with open(dist_neg_path) as dist_json:

                # Load experiment data
                dist_data = json.load(dist_json)

                self.dist_neg = Distributions(dataset=self.dataset)
                
                # Load experiment attributes
                self.dist_neg.load(dist_data)

            print("Loaded associated distribution with '{}'".format(dist_neg_path)) if (self.debug or self.verbosity > 1)  else None
        else:
            print("/!\ No previous computed distribution found at '{}'".format(dist_neg_path)) if (self.debug or self.verbosity > 1)  else None

        print(self.approach)

        if self.approach in ['single_distribution', 'multi_distributions']:
            self.predict()      

        elif self.approach in ['DecisionTree', 'LogisticRegression']:
            print("yo")
            self.predictions_df = pd.DataFrame(json.loads(self.predictions_df))
            self.performance_df = pd.DataFrame(self.performances_df, index=[0])
            self._init_model()

        elif self.approach == 'nam':
            
            self.predictions_df = pd.DataFrame(json.loads(self.predictions_df))
            self.performance_df = pd.DataFrame(self.performances_df, index=[0])

            if os.path.isfile(model_path):
                
                if self.dataset.use_missing_indicator_variables :
                    
                    NAM_DEFAULT_PARAMETERS['model']['num_features'] = 4
                    
                else:
                    
                    NAM_DEFAULT_PARAMETERS['model']['num_features'] = 2
                    
                self.model = NAM(**NAM_DEFAULT_PARAMETERS['model'])
                self.model = self.model.double()
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                
            else:
                
                raise ValueError("No model stored for this experiment.")
                
        elif self.approach == 'xgboost':
            
            self.predictions_df = pd.DataFrame(json.loads(self.predictions_df))
            self.performance_df = pd.DataFrame(self.performances_df, index=[0])
            
            self.model = XGBClassifier(use_label_encoder=False, # TODO ADD PLAYING WITH PARAMETERS 
                                      learning_rate=0.01,
                                      verbosity=1,
                                      objective='binary:logistic',
                                      eval_metric='auc',
                                      booster='gbtree',
                                      tree_method='exact',
                                      subsample=1,
                                      colsample_bylevel=.8,
                                      alpha=0)    
            
            if os.path.isfile(model_path):
                
                self.model = pickle.load(open(model_path, "rb"))
                
        elif self.approach == 'ebm':
            
            self.predictions_df = pd.DataFrame(json.loads(self.predictions_df))
            self.performance_df = pd.DataFrame(self.performances_df, index=[0])
                                                                                                       
            self.model = ExplainableBoostingClassifier(feature_names=self.features_name, n_jobs=-1, random_state=RANDOM_STATE)
                        
            self._fit_vanilla()

    

        print("Experiment {} loaded successfully! :-)".format(previous_experiment))
        return  

    ################################ Computing predictions (predictions_df attribute) ###########################

    def _train_nam(self, num_cv=None, **kwargs):
        """
            This function is a helper function to fit and predict score using loocv. 
            It generates the `predictions_df` attribute, that predict score on all the dataset.

        """
        if num_cv is not None:
            return self._train_predict_nam_cv(num_cv)
             

        replicates_results = []
        best_roc_auc = 0
        NAM_DEFAULT_PARAMETERS['model']['num_features'] = len(self.features_name)

        for i in range(NAM_DEFAULT_PARAMETERS['num_replicates']):      
            print('\t===== Replicate no. {} =====\n'.format(i + 1)) if (self.debug or self.verbosity > 1)  else None

            # Split test and train dataset
            self.dataset.split_test_train()

            # Init data
            X_train, X_test = self.dataset.X_train, self.dataset.X_train
            y_train, y_test = self.dataset._y.squeeze(), self.dataset._y.squeeze()
                
            # Create the df associated to the test sample 
            test_dict = {i:X_test[:,i] for i in range(X_test.shape[1])}
            test_dict['y_true'] = y_test
            test_df = pd.DataFrame.from_dict(test_dict);test_df.columns = self.features_name + ["y_true"]

            # Create the PyTorch Datasets
            data_train = TabularData(X=X_train, y=y_train)
            data_test = TabularData(X=X_test, y=y_test) 

            # Init. the model
            model = NAM(**NAM_DEFAULT_PARAMETERS['model'])
            model = model.double()

            train_model(model, data_train, verbosity=self.verbosity, **NAM_DEFAULT_PARAMETERS['training'])
            y_, p_ = eval_model(model, data_test)

            res = (pd.DataFrame(p_, columns = self.features_name, index=test_df.index)
                        .add_suffix('_partial')
                        .join(test_df)
                        .assign(y_pred = y_)
                        .assign(replicate = i))

            replicates_results.append(res)

            # Save best model by computing auroc
            y_true = res['y_true'].to_numpy(); y_pred = res['y_pred'].to_numpy()
            fpr_auc, tpr_auc, _ = roc_curve(y_true, y_pred)      
            roc_auc = auc(fpr_auc, tpr_auc)  
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc

                if self.save_experiment:
                    torch.save(model.state_dict(), os.path.join(self.experiment_path, 'best_model.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'tmp', 'best_model.pt'))


        # Finally load the best model
        
        self.model = NAM(**NAM_DEFAULT_PARAMETERS['model'])
        self.model = self.model.double()
        if self.save_experiment:
            self.model.load_state_dict(torch.load(os.path.join(self.experiment_path, 'best_model.pt')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'tmp', 'best_model.pt')))

        self.model.eval() 
        
        return  pd.concat(replicates_results)

    def _train_predict_nam_cv(self, num_cv, **kwargs):
        """
            This function is a helper function to fit and predict score using loocv. 
            It generates the `predictions_df` attribute, that predict score on all the dataset.
        """

        self.num_cv = num_cv
        cv = StratifiedKFold(n_splits=num_cv, shuffle=True, random_state=0)
        print('Performing {} fold cross-validation.'.format(num_cv)) if self.verbosity > 1 else None
            
        replicates_results = []
        best_roc_auc = 0
        NAM_DEFAULT_PARAMETERS['model']['num_features'] = len(self.features_name)

        for i in range(NAM_DEFAULT_PARAMETERS['num_replicates']):      
            #print('\t===== Replicate no. {} =====\n'.format(i + 1)) if (self.debug or self.verbosity > 1)  else None
            

            y_pred_all = np.zeros(self.dataset.num_samples).astype('float32') 
            feature_net_values_all = np.zeros((self.dataset.num_samples, len(self.features_name))).astype('float32') 

            test_dict = { k :np.zeros(self.dataset.num_samples) for k in range(len(self.features_name))}
            test_dict['y_true'] = np.zeros(self.dataset.num_samples)
            for _, (train, test) in enumerate(cv.split(self.dataset._X, self.dataset._y)):
            
                # Init data
                X_train, X_test = self.dataset.X_train[train], self.dataset.X_train[test]
                y_train, y_test = self.dataset._y[train].squeeze(), self.dataset._y[test].squeeze()

                # Create the df associated to the test sample 
                for k in range(X_test.shape[1]):
                    test_dict[k][test] = X_test[:,k]
                                    
                test_dict['y_true'][test] = y_test
                test_df = pd.DataFrame.from_dict(test_dict);test_df.columns = self.features_name + ["y_true"]

                # Create the PyTorch Datasets
                data_train, data_test = TabularData(X=X_train, y=y_train), TabularData(X=X_test, y=y_test) 

                # Init. the model
                model = NAM(**NAM_DEFAULT_PARAMETERS['model'])
                model = model.double()

                train_model(model, data_train, verbosity=self.verbosity, **NAM_DEFAULT_PARAMETERS['training'])
                y_pred_all[test], feature_net_values_all[test] = eval_model(model, data_test)

                
            res = (pd.DataFrame(feature_net_values_all, columns = self.features_name, index=test_df.index)
                        .add_suffix('_partial')
                        .join(test_df)
                        .assign(y_pred = y_pred_all)
                        .assign(replicate = i))

            replicates_results.append(res)

            # Save best model by computing auroc
            y_true = res['y_true'].to_numpy(); y_pred = res['y_pred'].to_numpy()
            fpr_auc, tpr_auc, _ = roc_curve(y_true, y_pred)      
            roc_auc = auc(fpr_auc, tpr_auc)  
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc

                if self.save_experiment:
                    torch.save(model.state_dict(), os.path.join(self.experiment_path, 'best_model.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'tmp', 'best_model.pt'))

        # Finally load the best model

        self.model = NAM(**NAM_DEFAULT_PARAMETERS['model'])
        self.model = self.model.double()
        if self.save_experiment:
            self.model.load_state_dict(torch.load(os.path.join(self.experiment_path, 'best_model.pt')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'tmp', 'best_model.pt')))

        self.model.eval()

        return  pd.concat(replicates_results)

    def _fit_predict_vanilla(self, num_cv=None, **kwargs): #TODO not NAM
        """
            This function is a helper function to fit and predict score. The train set is used to train 
            the algorithm, the test set for evaluating performances.
            It generates the `predictions_df` attribute, that predict score on all the dataset.

        """

        if num_cv is not None:
            return self._fit_predict_vanilla_cv(num_cv)
             
        # Init data
        X_train, X_test = self.dataset.X_train, self.dataset.X_train
        y_train, y_test = self.dataset._y.squeeze(), self.dataset._y.squeeze()

        # Fit model 
        self.model.fit(X_train, y_train)

        # Create the df associated to the test sample 
        test_dict = {i:X_test[:,i] for i in range(X_test.shape[1])}
        test_dict['y_true'] = y_test
        test_dict['y_pred'] = self.model.predict_proba(X_test)[:,1]

        test_df = pd.DataFrame.from_dict(test_dict);test_df.columns = self.features_name + ["y_true", "y_pred"]

        if self.save_experiment and self.approach != 'ebm': # TODOADD
            pickle.dump(self.model, open(os.path.join(self.experiment_path, 'best_model.pt'), "wb"))
        
        return test_df

    def _fit_predict_vanilla_cv(self, num_cv, **kwargs): #TODO not NAM
        """
            This function is a helper function to fit and predict score using loocv. 
            It generates the `predictions_df` attribute, that predict score on all the dataset.

        """    
        self.num_cv = num_cv
        y_pred_score = -1*np.ones_like(self.dataset._y).astype('float32')  # init prediction scores 

        cv = StratifiedKFold(n_splits=num_cv, shuffle=True, random_state=0)
        print('Performing {} fold cross-validation.'.format(num_cv)) if self.verbosity > 1 else None
        
        for _, (train, test) in enumerate(cv.split(self.dataset._X, self.dataset._y)):
    
            # Init data
            X_train, X_test = self.dataset.X_train[train], self.dataset.X_train[test]
            y_train, y_test = self.dataset._y[train].squeeze(), self.dataset._y[test].squeeze()

            X_train, y_train = self.dataset.upsample_minority(X_train, y_train)

            # Reset model 
            self._init_model()
            # Fit model 
            self.model.fit(X_train, y_train)

            # Predict samples on the test set
            y_pred_score[test] = self.model.predict_proba(X_test)[:,1]
        

        # Create the df associated to the test sample 
        n_features = self.dataset.X_train.shape[1]
            
        test_dict = {k: self.dataset._X_train[:,k] for k in range(n_features)}
        test_dict['y_true'] = self.dataset._y
        test_dict['y_pred'] = y_pred_score

        test_df = pd.DataFrame.from_dict(test_dict)
        test_df.columns = self.features_name + ["y_true", "y_pred"]

        # Finally fit the model with all the data for plotting purposes 
        self._init_model()
        
        X_train = self.dataset.X_train
        y_train = self.dataset._y.squeeze()

        # Fit model 
        self.model.fit(X_train, y_train)

        if self.save_experiment and self.approach != 'ebm': # TODOADD
            pickle.dump(self.model, open(os.path.join(self.experiment_path, 'best_model.pt'), "wb"))

        return test_df
    
    def _predict_map(self):
    
        #################################################################
        #  Prediction using maximum likelihood estimation
        #################################################################

        _, step = np.linspace(-3, 3, self.dist_pos.resolution, retstep=True)

        # Contains for each sample of the Test set, the corresponding x and y index coordinates, in the matrix of the 2D pdf... 
        coord_to_index = np.floor_divide(self.dataset.X_test+2.5, step)


        # Init. the array of prediction
        y_pred = np.zeros(shape=self.dataset.y_test.shape[0]); arr = []



        #----------- Treat the case of when none coordinates are known

        # Index of the samples in the test set where none of the coordinates are known 
        X_indexes_none_known = np.argwhere(((~np.isnan(coord_to_index)).sum(axis=1)==0)).squeeze()

        # Compare likelihood to do the prediction and assign the label in the prediction

        # If equal prior, we assign labels randomly
        if self.dist_pos.f_0  == self.dist_neg.f_0:
            y_pred[X_indexes_none_known] = np.random.randint(0, 2, len(X_indexes_none_known))
        # Otherwise  the label is based on the a posterior on the missingness ratio
        else:
            y_pred[X_indexes_none_known] = np.array( len(X_indexes_none_known) * [int(self.dist_pos.f_0  > self.dist_neg.f_0)])

        arr.extend(list(X_indexes_none_known))


        #----------- Treat the case of when both coordinates are known

        # Index of the samples in the test set where both first coordinates are known 
        X_indexes_both_known = np.argwhere((~np.isnan(coord_to_index)).sum(axis=1)==2).squeeze(); arr.extend(list(X_indexes_both_known))

        # Coordinates of indexes in the feature space of the samples in the test set where both first coordinates are known 
        hat_f_coordinates = coord_to_index[X_indexes_both_known].astype(int)
        inds_array = np.moveaxis(np.array(list(map(tuple, hat_f_coordinates))), -1, 0)

        # Compare likelihood to do the prediction
        y_pred_both_known = (self.dist_pos.f[tuple(inds_array)] > self.dist_neg.f[tuple(inds_array)]).astype(int)

        # Assign predictions 
        y_pred[X_indexes_both_known] = y_pred_both_known


        #----------- Treat the case of when only the first coordinate is known

        # Index of the samples in the test set where only the first coordinate is known 
        X_indexes_first_known = np.argwhere(~np.isnan(coord_to_index[:,0]) & np.isnan(coord_to_index[:,1])).squeeze(); arr.extend(list(X_indexes_first_known))

        # Coordinates of index in the feature space of the samples in the test set where only the first coordinate is known 
        hat_f_coordinates = coord_to_index[X_indexes_first_known][:,0].astype(int)

        # Compare likelihood to do the prediction
        if self.approach == 'multi_distributions':
            y_pred_first_known = (self.dist_pos.f_1[hat_f_coordinates] > self.dist_neg.f_1[hat_f_coordinates]).astype(int)
        else:
            y_pred_first_known = (self.dist_pos.f_1_marginal[hat_f_coordinates] > self.dist_neg.f_1_marginal[hat_f_coordinates]).astype(int)

        # Assign predictions 
        y_pred[X_indexes_first_known] = y_pred_first_known

        #----------- Treat the case of when only the second coordinate is known


        # Index of the samples in the test set where only the first coordinate is known 
        X_indexes_second_known = np.argwhere(np.isnan(coord_to_index[:,0]) & ~np.isnan(coord_to_index[:,1])).squeeze(); arr.extend(list(X_indexes_second_known))

        # Coordinates of index in the feature space of the samples in the test set where only the first coordinate is known 
        hat_f_coordinates = coord_to_index[X_indexes_second_known][:,1].astype(int)

        # Compare likelihood to do the prediction
        if self.approach == 'multi_distributions':
            y_pred_second_known = (self.dist_pos.f_2[hat_f_coordinates] > self.dist_neg.f_2[hat_f_coordinates]).astype(int)
        else:
            y_pred_second_known = (self.dist_pos.f_2_marginal[hat_f_coordinates] > self.dist_neg.f_2_marginal[hat_f_coordinates]).astype(int)

        # Assign predictions 
        y_pred[X_indexes_second_known] = y_pred_second_known

        assert len(arr) == self.dataset.y_test.shape[0], "/!\. Not enough predictions made, check this out!"
            
        print("Sanity check: number of predictions: {} == {}: Num samples\n".format(len(arr), self.dataset.y_test.shape[0])) if (self.debug or self.verbosity > 1)  else None

        self.dataset.y_pred = y_pred

        return
    
    ################################ Computing performances (performcances_df attribute) ###########################

    def _performances(self):
        """
            This function generates the `performances_df` dataframe, using the prediction scores of `predictions_df`

        """

        assert self.purpose == 'classification', "/!\. Purpose mode is set to `estimation`, you should set it to `classification`. :-)"
        # TODO: if we don't enlarge the scope of those classes (only use fit and predict with distributions), then no need to distinguish here...

        if self.approach == 'nam':
            
            # Compute perf.
            self._performances_nam()
            
        elif self.approach in ['multi_distributions', 'single_distribution', 'xgboost', 'ebm', 'DecisionTree', 'LogisticRegression']:
            
            # Compute perf.
            self._performances_vanilla()
            
        else:
            raise ValueError("Please use 'single_distribution', 'multi_distributions' or 'nam', 'xgboost', or 'ebm' approach.")
        
        if self.save_experiment:
            self.save() 

        return

    def _performances_vanilla(self): 

        if self.dataset.dataset_name in ['blobs', 'circles', 'moons']:
            y_true = self.dataset.y_test.squeeze()
            y_pred = self.dataset.y_pred

        else:
            
            y_true = self.predictions_df['y_true'].to_numpy()
            y_pred = self.predictions_df['y_pred'].to_numpy()

        # Compute first AUROC
        auroc = roc_auc_score(y_true, y_pred)

        # Compute the AUC-PR
        auc_pr = average_precision_score(y_true, y_pred)

        # Compute the AUC-PR Corrected
        auc_pr_corrected = average_precision(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO)

        # Compute the AUC-PR Gain
        auc_pr_g = prg.calc_auprg(prg.create_prg_curve(y_true, y_pred))

        # Compute the AUC-PR Gain corrected
        auc_pr_g_corrected = calc_auprg(create_prg_curve(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO))

        # Compute the F1 score
        f1, optimal_threshold = bestf1score(y_true, y_pred, pi0=None)

        # Compute the corrected F1 score
        f1_corrected, _ = bestf1score(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred > optimal_threshold).ravel()

        acc = (tp + tn) / (tp + tn + fp +  fn)
        mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        tpr =  tp / (tp+fn)
        tnr = tn / (tn+fp)
        ppv = tp / (tp+fp)
        npv = tn / (tn+fn)
        fnr = fn / (tp+fn)

        performances_dict = {'AUROC':round(auroc, 3),
                            'AUC-PR': round(auc_pr, 3),
                            'AUC-PR-Gain': round(auc_pr_g, 3),
                            'AUC-PR-Corrected': round(auc_pr_corrected, 3),
                            'AUC-PR-Gain-Corrected' :round(auc_pr_g_corrected, 3),
                            'F1 score (2 PPVxTPR/(PPV+TPR))': round(f1, 3),
                            'F1 score Corrected': round(f1_corrected, 3),
                            'Accuracy' : round(acc, 3),
                            'Matthews correlation coefficient (MCC)': round(mcc, 3),
                            'Sensitivity, recall, hit rate, or true positive rate (TPR)': round(tpr, 3),
                            'Specificity, selectivity or true negative rate (TNR)': round(tnr, 3),
                            'Precision or positive predictive value (PPV)': round(ppv, 3),
                            'Negative predictive value (NPV)': round(npv, 3),
                            'Miss rate or false negative rate (FNR)': round(fnr, 3),
                            'False discovery rate (FDR=1-PPV)': round(1-ppv, 3),
                            'False omission rate (FOR=1-NPV)': round(1-npv, 3)}

        self.performances_df = pd.DataFrame(performances_dict, index=['0'])  

        return 

    def _performances_nam(self):
        """
            Create the predictions_df and performances_df based on the dataframe that recap all the results for the replciates. 
        """

        performances_dict = {'AUROC':[], 
                            'AUC-PR': [],
                            'AUC-PR-Gain': [],
                            'AUC-PR-Corrected':[],
                            'AUC-PR-Gain-Corrected' : [],
                            'F1 score (2 PPVxTPR/(PPV+TPR))': [],
                            'F1 score Corrected': [],
                            'Accuracy' : [],
                            'Matthews correlation coefficient (MCC)': [],
                            'Sensitivity, recall, hit rate, or true positive rate (TPR)': [],
                            'Specificity, selectivity or true negative rate (TNR)': [],
                            'Precision or positive predictive value (PPV)': [],
                            'Negative predictive value (NPV)': [],
                            'Miss rate or false negative rate (FNR)': [],
                            'False discovery rate (FDR=1-PPV)': [],
                            'False omission rate (FOR=1-NPV)': [],
                            'Area Under the Curve (AUC)': []}


        for _, res in self.predictions_df.groupby('replicate'):
            y_true = res['y_true'].to_numpy()
            y_pred = res['y_pred'].to_numpy()


            # Compute first AUROC
            auroc = roc_auc_score(y_true, y_pred)

            # Compute the AUC-PR
            auc_pr = average_precision_score(y_true, y_pred)

            # Compute the AUC-PR Corrected
            auc_pr_corrected = average_precision(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO)

            # Compute the AUC-PR Gain
            auc_pr_g = prg.calc_auprg(prg.create_prg_curve(y_true, y_pred))

            # Compute the AUC-PR Gain corrected
            auc_pr_g_corrected = calc_auprg(create_prg_curve(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO))

            # Compute the F1 score
            f1, optimal_threshold = bestf1score(y_true, y_pred, pi0=None)

            # Compute the corrected F1 score
            f1_corrected, _ = bestf1score(y_true, y_pred, pi0=REFERENCE_IMBALANCE_RATIO)

            
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred > optimal_threshold).ravel()

            acc = (tp + tn) / (tp + tn + fp +  fn)
            mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            tpr =  tp / (tp+fn)
            tnr = tn / (tn+fp)
            ppv = tp / (tp+fp)
            npv = tn / (tn+fn)
            fnr = fn / (tp+fn)
            
            fpr_auc, tpr_auc, _ = roc_curve(y_true, y_pred)      
            roc_auc = auc(fpr_auc, tpr_auc)    

            performances_dict['AUROC'].append(round(auroc, 3))
            performances_dict['AUC-PR'].append(round(auc_pr, 3))
            performances_dict['AUC-PR-Gain'].append(round(auc_pr_g, 3))
            performances_dict['AUC-PR-Corrected'].append(round(auc_pr_corrected, 3))
            performances_dict['AUC-PR-Gain-Corrected'].append(round(auc_pr_g_corrected, 3))
            performances_dict['F1 score (2 PPVxTPR/(PPV+TPR))'].append(round(f1, 3))
            performances_dict['F1 score Corrected'].append(round(f1_corrected, 3))
            performances_dict['Accuracy'].append(round(acc, 3))
            performances_dict['F1 score (2 PPVxTPR/(PPV+TPR))'].append(round(f1, 3))
            performances_dict['Matthews correlation coefficient (MCC)'].append(round(mcc, 3))
            performances_dict['Sensitivity, recall, hit rate, or true positive rate (TPR)'].append(round(tpr, 3))
            performances_dict['Specificity, selectivity or true negative rate (TNR)'].append(round(tnr, 3))
            performances_dict['Precision or positive predictive value (PPV)'].append(round(ppv, 3))
            performances_dict['Negative predictive value (NPV)'].append(round(npv, 3))
            performances_dict['Miss rate or false negative rate (FNR)'].append(round(fnr, 3))
            performances_dict['False discovery rate (FDR=1-PPV)'].append(round(1-ppv, 3))
            performances_dict['False omission rate (FOR=1-NPV)'].append(round(1-npv, 3))
            performances_dict['Area Under the Curve (AUC)'].append(round(roc_auc, 3))
            
        # Create the performace df
        performances_dict = {metric: np.mean(values) for metric, values in performances_dict.items()}
        self.performances_df = pd.DataFrame(performances_dict, index = [0])
        return

    ################################ Plotting functions ###########################

    def _plot_estimation(self):
    
        # Create the pannel 
        fig, axes = plt.subplots(3, 5, figsize=(30, 12));axes = axes.flatten()
        fig.suptitle("{}\n{}".format(int(self.experiment_number), self.description, self.dataset.missingness_description), y=1.05, weight='bold', fontsize=12)


        # Plot the dataset 
        axes[1], axes[3] = self.dataset.plot(ax1=axes[1], ax2=axes[3], title=False)
        axes[1].set_title("Training set"); axes[3].set_title("Test set")

        # Plot the distributions
        axes = self.dist.plot(axes=axes)

        bar = axes[12].bar([0], self.dist.f_0, color = 'tab:orange', label="P(Z_1=0, Z_2=0)");label_bar(bar,axes[12])
        axes[12].set_title("H)\nBoth coord. missing");axes[7].set_xlim([-2, 2])

        # Plot the points on the distributions
        _ = [ax.legend(prop={'size':10}, loc='lower right') for i,ax in enumerate(axes) if i in [12]]; [axes[i].axis('off') for i in range(len(axes))]
        plt.tight_layout()

        plt.show()


        return      

    def _plot_classification(self, *args, **kwargs):
    
        # Create the pannel 
        fig, axes = plt.subplots(5, 5, figsize=(20, 14)); axes = axes.flatten()
        fig.suptitle("{}\n{}".format(int(self.experiment_number), self.description, self.dataset.missingness_description), y=1.05, weight='bold', fontsize=12)

        # Plot the dataset 
        axes[1], axes[3] = self.dataset.plot(ax1=axes[1], ax2=axes[3], title=False)
        axes[1].set_title("Training set ({})".format(self.dataset.X_train.shape[0])); axes[3].set_title("Test set ({})".format(self.dataset.X_test.shape[0]))

        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].tolist(), self.predictions_df['y_pred'].tolist())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes[22])
        disp.im_.colorbar.remove()    
        
        axes = self.dist_pos.plot(axes=axes, predictions_df=self.predictions_df, *args, **kwargs)
        axes = self.dist_neg.plot(axes=axes, predictions_df=self.predictions_df, *args, **kwargs)

        # Handle legend and set axis off
        axes_with_legend = [5, 7, 9, 12, 15, 17, 19] if self.approach == 'multi_distributions' else [6, 7, 8, 12, 16, 17, 18]
        _ = [ax.legend(prop={'size':10}) for i,ax in enumerate(axes) if i in axes_with_legend]; [axes[i].axis('off') for i in range(len(axes)) if i!=22 ]
        
        plt.tight_layout();plt.show()

        #Compute metrics of interest  
        tn, fp, fn, tp = confusion_matrix(self.predictions_df['y_true'].tolist(), self.predictions_df['y_pred'].tolist()).ravel()
        print('Sample: {} positive and {} negative samples (#p/#n={:3.0f}%)\n'.format(tp+fn, tn+fp, 100*(tp+fn)/(tn+fp)))

        display(self.performances_df.transpose())

        #for item, value in performances_metrics.items(): TODOREMMOVE
        #    print("  {0:70}\t {1}".format(item, value))

        return     

    def _plot_nam(self):
        
        if self.dataset.dataset_name not in ['blobs', 'circles', 'moons']:
            self._plot_nam_autism()
            return
        
        #Select the best replicate as the predictions_df for plotting reasons. Note this is based on AUC.
        best_replicate = np.argmax(self.performances_df['Area Under the Curve (AUC)'])
        best_predictions_df = self.predictions_df.query(" `replicate` == @best_replicate")

        # Create the pannel 
        fig, axes = plt.subplots(2, 5, figsize=(25, 8)); axes = axes.flatten()
        fig.suptitle("({}) {}\n{}".format(int(self.experiment_number), self.description, self.dataset.missingness_description), y=1.1, weight='bold', fontsize=12)

        # Plot the dataset 
        axes[0], axes[1] = self.dataset.plot(ax1=axes[0], ax2=axes[1], title=False)
        axes[0].set_title("Training set ({})".format(self.dataset.X_train.shape[0])); axes[1].set_title("Test set ({})".format(self.dataset.X_test.shape[0]))

        # Plot the performances 
        cm = confusion_matrix(best_predictions_df['y_true'].to_numpy(), best_predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes[2]); disp.im_.colorbar.remove()    
                                                                                                    
        # Plot the roc curves
        axes[3] = plot_roc_curves_nam(self.predictions_df, ax=axes[3]) 
        axes = plot_shape_functions(self.predictions_df, self.features_name, axes=axes, ncols=5, start_axes_plotting=5) 

        # Plot the dataset with the errors

        y_true = self.dataset.y_test.squeeze()
        y_pred = (self.dataset.y_pred > CLASSIFICATION_THRESHOLD).astype(int)

        # Creation of a df for the prediction
        predictions_df = pd.DataFrame({'X1':self.dataset._X_raw[self.dataset.test_index][:,0], 
                                    'X2':self.dataset._X_raw[self.dataset.test_index][:,1], 
                                    'Z1':[1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,0]],
                                    'Z2': [1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,1]],
                                    'Have missing' : [(np.isnan(x).sum()>0).astype(int)  for x in self.dataset._X_test],
                                    'y_true': y_true, 
                                    'y_pred': y_pred, 
                                    'True Positive': [1 if y_true==1 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'True Negative': [1 if y_true==0 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Positive': [1 if y_true==0 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Negative': [1 if y_true==1 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    })

        alpha=1
        axes[4].set_title("Classification result (th={})".format(CLASSIFICATION_THRESHOLD));axes[4].axis('off')

        # Plot the sample points without missing data
        axes[4].scatter(predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes[4].scatter(predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes[4].scatter(predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes[4].scatter(predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))


        # Plot the sample points without missing data
        axes[4].scatter(predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes[4].scatter(predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes[4].scatter(predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes[4].scatter(predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))

        # Plot the sample points without missing data
        axes[9].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes[9].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes[9].scatter([-5], [-5],color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes[9].scatter([-5], [-5],color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))
        axes[9].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes[9].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes[9].scatter([-5], [-5],color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes[9].scatter([-5], [-5], color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))
        axes[9].set_xlim([0,1]);axes[9].set_ylim([0,1]);axes[9].axis('off');axes[9].legend(loc='center', prop={'size':15})

        if not self.dataset.use_missing_indicator_variables:
            [axes[i].axis('off') for i in [7, 8, 9]]
        else:
            [axes[i].axis('off') for i in [9]]


        plt.tight_layout();plt.show()
        
        return 

    def _plot_nam_autism(self):
        
        #Select the best replicate as the predictions_df for plotting reasons. Note this is based on AUC.
        best_replicate = np.argmax(self.performances_df['Area Under the Curve (AUC)'])
        best_predictions_df = self.predictions_df.query(" `replicate` == @best_replicate")

        n_plots = 5 + len(self.features_name)
        
        # Create the pannel 
        fig, axes = plt.subplots(n_plots//5+1, 5, figsize=(25, 8)); axes = axes.flatten()
        fig.suptitle("({}) {}".format(int(self.experiment_number), self.description), y=1.1, weight='bold', fontsize=12)

        # Plot the performances 
        cm = confusion_matrix(best_predictions_df['y_true'].to_numpy(), best_predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes[3])
        disp.im_.colorbar.remove()    
                                                                                                                     

        # Plot the roc curves
        axes[1] = plot_roc_curves_nam(self.predictions_df, ax=axes[1]) 
        axes = plot_shape_functions(self.predictions_df, self.features_name, axes=axes, ncols=5, start_axes_plotting=5) 

    
        [axes[i].axis('off') for i in [0, 2, 4]]


        plt.tight_layout();plt.show()
        
        return 
    
    def _plot_xgboost(self):
        
        if self.dataset.dataset_name not in ['blobs', 'circles', 'moons']:
            self._plot_xgboost_autism()
            return        
        
        # Create the pannel 
        fig_mosaic = """
                        ABCD
                        EGHI
                        FFFF
                        FFFF
                    """

        fig, axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(25,18))
        
        fig.suptitle("({}) {}\n{}".format(int(self.experiment_number), self.description, self.dataset.missingness_description), y=1.1, weight='bold', fontsize=12)

        # Plot the dataset 
        axes['A'], axes['B'] = self.dataset.plot(ax1=axes['A'], ax2=axes['B'], title=False)
        axes['A'].set_title("Training set ({})".format(self.dataset.X_train.shape[0])); axes['B'].set_title("Test set ({})".format(self.dataset.X_test.shape[0]))

        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].to_numpy(), self.predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes['C']);disp.im_.colorbar.remove()    
                                                                                                    
        # Plot the roc curves
        axes['D'] = plot_roc_curves_xgboost(self.predictions_df, ax=axes['D']) 
        
        # Plot features importance
        self.model.get_booster().feature_names = self.features_name
        axes['E'] = plot_importance(self.model.get_booster(),  height=0.5, ax = axes['E'])
        
        # Plot Tree
        axes['F'] = plot_tree(self.model.get_booster(), num_trees=self.model.best_iteration, ax=axes['F'])
        
        

        y_true = self.dataset.y_test.squeeze()
        y_pred = (self.dataset.y_pred > CLASSIFICATION_THRESHOLD).astype(int)

        # Creation of a df for the prediction
        predictions_df = pd.DataFrame({'X1':self.dataset._X_raw[self.dataset.test_index][:,0], 
                                    'X2':self.dataset._X_raw[self.dataset.test_index][:,1], 
                                    'Z1':[1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,0]],
                                    'Z2': [1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,1]],
                                    'Have missing' : [(np.isnan(x).sum()>0).astype(int)  for x in self.dataset._X_test],
                                    'y_true': y_true, 
                                    'y_pred': y_pred, 
                                    'True Positive': [1 if y_true==1 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'True Negative': [1 if y_true==0 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Positive': [1 if y_true==0 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Negative': [1 if y_true==1 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    })

        alpha=1
        axes['G'].set_title("Classification result (th={})".format(CLASSIFICATION_THRESHOLD));axes['G'].grid()#;axes['G'].axis('off')

        # Plot the sample points without missing data
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))


        # Plot the sample points without missing data
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))

        # Plot the sample points without missing data
        axes['H'].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes['H'].scatter([-5], [-5], color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))
        axes['H'].set_xlim([0,1]);axes['H'].set_ylim([0,1]);axes['H'].axis('off');axes['H'].legend(loc='center', prop={'size':15})

        #if not self.dataset.use_missing_indicator_variables:
        #    [axes[i].axis('off') for i in [7, 8, 9]]
        #else:
        axes['I'].axis('off')


        plt.tight_layout();plt.show()

        return 

    def _plot_xgboost_autism(self):
        
        # Create the pannel 
        fig_mosaic = """
                        ABCC
                        FFFF
                        FFFF
                    """

        fig, axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(25,18))
        
        fig.suptitle("({}) {}".format(int(self.experiment_number), self.description), y=1.1, weight='bold', fontsize=12)

        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].to_numpy(), self.predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes['B']);disp.im_.colorbar.remove()    
                                                                                                    
        # Plot the roc curves
        axes['A'] = plot_roc_curves_xgboost(self.predictions_df, ax=axes['A']) 
        
        # Plot features importance
        self.model.get_booster().feature_names = self.features_name
        axes['C'] = plot_importance(self.model.get_booster(),  height=0.5, ax = axes['C'])
        
        # Plot Tree
        axes['F'] = plot_tree(self.model.get_booster(), num_trees=self.model.best_iteration, ax=axes['F'])

        plt.tight_layout();plt.show()

        return 

    def _plot_ebm(self):


        if self.dataset.dataset_name not in ['blobs', 'circles', 'moons']:
            return self._plot_ebm_autism()
            

        # Create the pannel 
        fig_mosaic = """
                        ABC
                        DGH
                    """

        fig, axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(20,12))
        
        fig.suptitle("({}) {}\n{}".format(int(self.experiment_number), self.description, self.dataset.missingness_description), y=1.1, weight='bold', fontsize=12)

        # Plot the dataset 
        axes['A'], axes['B'] = self.dataset.plot(ax1=axes['A'], ax2=axes['B'], title=False)
        axes['A'].set_title("Training set ({})".format(self.dataset.X_train.shape[0])); axes['B'].set_title("Test set ({})".format(self.dataset.X_test.shape[0]))

        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].to_numpy(), self.predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes['C']);disp.im_.colorbar.remove()    
                                                                                                                                                                

        # Plot the roc curves
        axes['D'] = plot_roc_curves_xgboost(self.predictions_df, ax=axes['D']) 
        
        y_true = self.dataset.y_test.squeeze()
        y_pred = (self.dataset.y_pred > CLASSIFICATION_THRESHOLD).astype(int)

        # Creation of a df for the prediction
        predictions_df = pd.DataFrame({'X1':self.dataset._X_raw[self.dataset.test_index][:,0], 
                                    'X2':self.dataset._X_raw[self.dataset.test_index][:,1], 
                                    'Z1':[1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,0]],
                                    'Z2': [1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,1]],
                                    'Have missing' : [(np.isnan(x).sum()>0).astype(int)  for x in self.dataset._X_test],
                                    'y_true': y_true, 
                                    'y_pred': y_pred, 
                                    'True Positive': [1 if y_true==1 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'True Negative': [1 if y_true==0 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Positive': [1 if y_true==0 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Negative': [1 if y_true==1 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    })

        alpha=1
        axes['G'].set_title("Classification result (th={})".format(CLASSIFICATION_THRESHOLD));axes['G'].grid()#;axes['G'].axis('off')

        # Plot the sample points without missing data
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))


        # Plot the sample points without missing data
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))

        # Plot the sample points without missing data
        axes['H'].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes['H'].scatter([-5], [-5], color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))
        axes['H'].set_xlim([0,1]);axes['H'].set_ylim([0,1]);axes['H'].axis('off');axes['H'].legend(loc='center', prop={'size':15})

        #if not self.dataset.use_missing_indicator_variables:
        #    [axes[i].axis('off') for i in [7, 8, 9]]
        #else:
        #axes['I'].axis('off')


        plt.tight_layout();plt.show()


        ebm_global = self.model.explain_global()

        return show(ebm_global)

    def _plot_ebm_autism(self):

        # Create the pannel 
        fig_mosaic = """
                        AB
                    """

        fig, axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(20,8))
        
        fig.suptitle("({}) {}".format(int(self.experiment_number), self.description), y=1.1, weight='bold', fontsize=12)

        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].to_numpy(), self.predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes['B']);disp.im_.colorbar.remove()    

        # Plot the roc curves
        axes['A'] = plot_roc_curves_xgboost(self.predictions_df, ax=axes['A']) 

        plt.tight_layout();plt.show()

        ebm_global = self.model.explain_global()

        return show(ebm_global)

    def _plot_lr_decision_tree(self):

        if self.dataset.dataset_name not in ['blobs', 'circles', 'moons']:
            self._plot_lr_decision_tree_autism()
            return

        # Create the pannel 
        fig_mosaic = """
                        ABC
                        DGH
                    """

        fig, axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(20,12))

        fig.suptitle("({}) {}\n{}".format(int(self.experiment_number), self.description, self.dataset.missingness_description), y=1.1, weight='bold', fontsize=12)

        # Plot the dataset 
        axes['A'], axes['B'] = self.dataset.plot(ax1=axes['A'], ax2=axes['B'], title=False)
        axes['A'].set_title("Training set ({})".format(self.dataset.X_train.shape[0])); axes['B'].set_title("Test set ({})".format(self.dataset.X_test.shape[0]))

        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].to_numpy(), self.predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes['C']);disp.im_.colorbar.remove()    
                                                                                                    
        # Plot the roc curves
        axes['D'] = plot_roc_curves_xgboost(self.predictions_df, ax=axes['D']) 
        
        y_true = self.dataset.y_test.squeeze()
        y_pred = (self.dataset.y_pred > CLASSIFICATION_THRESHOLD).astype(int)

        # Creation of a df for the prediction
        predictions_df = pd.DataFrame({'X1':self.dataset._X_raw[self.dataset.test_index][:,0], 
                                    'X2':self.dataset._X_raw[self.dataset.test_index][:,1], 
                                    'Z1':[1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,0]],
                                    'Z2': [1 if not np.isnan(x) else 0 for x in self.dataset._X_test[:,1]],
                                    'Have missing' : [(np.isnan(x).sum()>0).astype(int)  for x in self.dataset._X_test],
                                    'y_true': y_true, 
                                    'y_pred': y_pred, 
                                    'True Positive': [1 if y_true==1 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'True Negative': [1 if y_true==0 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Positive': [1 if y_true==0 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    'False Negative': [1 if y_true==1 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                                    })

        alpha=1
        axes['G'].set_title("Classification result (th={})".format(CLASSIFICATION_THRESHOLD));axes['G'].grid()#;axes['G'].axis('off')

        # Plot the sample points without missing data
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Positive`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `True Negative`==1")['X2'],
                    color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==0 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))


        # Plot the sample points without missing data
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Positive`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `True Negative`==1")['X2'],
                    color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Positive`==1")['X2'],
                    color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes['G'].scatter(predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X1'], 
                    predictions_df.query(" `Have missing`==1 and `False Negative`==1")['X2'],
                    color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))

        # Plot the sample points without missing data
        axes['H'].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', alpha=alpha, label="TN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `True Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:orange', s=100, alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:red', s=100, alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Have missing`==0 and `False Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Positive`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:blue', facecolors='none', alpha=alpha, label="TN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `True Negative`==1"))))
        axes['H'].scatter([-5], [-5],color='tab:orange', s=100, facecolors='none', alpha=alpha, label="FP (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Positive`==1"))))
        axes['H'].scatter([-5], [-5], color='tab:red', s=100, facecolors='none', alpha=alpha, label="FN (n={}) with Missing".format(len(predictions_df.query(" `Have missing`==1 and `False Negative`==1"))))
        axes['H'].set_xlim([0,1]);axes['H'].set_ylim([0,1]);axes['H'].axis('off');axes['H'].legend(loc='center', prop={'size':15})

        #if not self.dataset.use_missing_indicator_variables:
        #    [axes[i].axis('off') for i in [7, 8, 9]]
        #else:
        #axes['I'].axis('off')


        plt.tight_layout();plt.show()

        return 

    def _plot_lr_decision_tree_autism(self):
    
        # Create the pannel 
        fig_mosaic = """
                        AB
                    """

        fig, axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(18,6))

        fig.suptitle("({}) {}".format(int(self.experiment_number), self.description), y=1.1, weight='bold', fontsize=12)

        # Plot the roc curves
        axes['A'] = plot_roc_curves_xgboost(self.predictions_df, ax=axes['A']) 
        
        
        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].to_numpy(), self.predictions_df['y_pred'].to_numpy()> CLASSIFICATION_THRESHOLD)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes['B']);disp.im_.colorbar.remove()    
                                                                                                                                                                          
        plt.tight_layout();plt.show()

        return 

    ################################  Experiments maintenance ################################ 

    def _init_experiment_path(self, suffix=None):
        """
            This method create the experiment path and folders associated with experiments. 
            It creates into the DATA_DIR location - usually "*/data/ - several objects (here is an exemple for the autism project):

                data/
                ├── experiments/
                │   ├── README.md
                │   └── 0
                │   │   ├── experiments_log.json
                │   │   ├── model
                |   |   |     ├── model.gz
                │   │   └── distributions
                |   |   |     ├── hat_f.npy
                |   |   |     ├── hat_f_1.npy
                |   |   |     ├── hat_f_2.npy
                └── *Whatever you have here*



            You have to figure out: 
                - The name of the different sub-folders contained in each experiments (model (as in the example), fisher_vectors, images, etc.)
                - The attributes of this classes which cannot be simply saved/loaded in json files. Those will be handled separately.

        """
        
        # Create experiment folder if not already created

        if not os.path.isdir(os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME)):
            os.mkdir(os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME))

        if not os.path.isdir(os.path.join(DATA_DIR, 'tmp')):
            os.mkdir(os.path.join(DATA_DIR, 'tmp'))  

        # Create dataset experiment folder  if not already created
        if not os.path.isdir(os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME, self.dataset_name)):
            os.mkdir(os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME, self.dataset_name))

        if not os.path.isdir(os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME, self.dataset_name, '0')):
            os.mkdir(os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME, self.dataset_name, '0'))
    
        # Looking for the number of the new experiment number
        experiment_number = np.max([int(os.path.basename(path)) for path in glob(os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME, self.dataset_name, '*'))])+1

        experiment_path = os.path.join(DATA_DIR, EXPERIMENT_FOLDER_NAME, self.dataset_name, str(experiment_number))
        print('Doing experiment {}!'.format(experiment_number))

        # Create experiment folder 
        os.mkdir(experiment_path)

        # Create sub-folders associated to the project
        #os.mkdir(os.path.join(experiment_path, 'dataset'))

        # Create json path
        json_path = os.path.join(experiment_path, 'experiment_log.json')

        return experiment_number, experiment_path, json_path
   
    def _init_model(self, **kwargs):
    
        """ 
            This function fit the model and predict score.
            It generates the `predictions_df` attribute, that predict score on all the dataset.

        """

        if self.approach in ['single_distribution', 'multi_distributions']:

            if self.purpose == 'classification':
                    #Estimation of the distributions for the positive and negative class
                self.dist_pos = Distributions(dataset=self.dataset, 
                                            class_used=1, 
                                            approach=self.approach,
                                            cmap='Blues',
                                            debug=self.debug, 
                                            verbosity=1)

                self.dist_neg = Distributions(dataset=self.dataset, 
                                            class_used=0, 
                                            approach=self.approach,
                                            cmap='Greens',
                                            debug=self.debug, 
                                            verbosity=1)

            elif self.purpose == 'estimation':
                #Estimation of the distribution
                self.dist = Distributions(dataset=self.dataset, 
                                        class_used=None, 
                                        cmap='Oranges',
                                        verbosity=1)

        elif self.approach == 'nam':

            pass

        elif self.approach == 'xgboost':
            
            self.model = XGBClassifier(use_label_encoder=False, # TODO ADD PLAYING WITH PARAMETERS 
                                      learning_rate=0.01,
                                       n_estimators= 1000,
                                      max_depth = 3,
                                      verbosity=1,
                                      objective='binary:logistic',
                                      eval_metric='auc',
                                      booster='gbtree',
                                      tree_method='exact',
                                      subsample=1,
                                      colsample_bylevel=.8,
                                      alpha=0)


        elif self.approach == 'ebm':

            self.model = ExplainableBoostingClassifier(feature_names=self.features_name, 
                                                        max_bins=256,
                                                        max_interaction_bins=256,
                                                        binning='quantile',
                                                        mains='all',
                                                        interactions=40,
                                                        outer_bags=8,
                                                        inner_bags=0,
                                                        learning_rate=0.01,
                                                        validation_size=0.15,
                                                        early_stopping_rounds=50,
                                                        early_stopping_tolerance=0.0001,
                                                        max_rounds=10000,
                                                        min_samples_leaf=2,
                                                        max_leaves=3,
                                                        n_jobs=-2,
                                                        random_state=RANDOM_STATE)


        elif self.approach == 'DecisionTree':
                                                                                           
            self.model = DecisionTreeClassifier(criterion='gini',
                                splitter='best',
                                max_depth=3,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features=None,
                                random_state=None,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                class_weight=None,
                                ccp_alpha=0.0)


        elif self.approach == 'LogisticRegression':
            self.model = LogisticRegression(random_state=0)

        return 
    
    def _load(self, data):

        for key, value in data.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else: 
                setattr(self, key, value)
