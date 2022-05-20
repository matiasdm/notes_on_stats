import os
import sys
import json 

from glob import glob
from tqdm import tqdm
from time import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# add tools path and import our own tools
sys.path.insert(0, '../tools')

from const import *
from utils import fi, label_bar

from generateToyDataset import DatasetGenerator
from distributions import Distributions


class Experiments(object):


    def __init__(self, dataset_name, dataset_train=None, dataset_test=None, purpose='classification', previous_experiment=None, save_experiment=True, verbosity=1, debug=False, proportion_train=PROPORTION_TRAIN, resolution=RESOLUTION, bandwidth=BANDWIDTH, random_state=RANDOM_STATE):

        # Set definitions attributes (also used for log purposes)
        self.dataset_name = dataset_name

        if previous_experiment is not None:
            self.load(previous_experiment)
            return

        # Dataset 
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

        # Distributions
        self.resolution = resolution
        self.bandwidth = bandwidth = bandwidth
        self.dist_pos = None
        self.dist_neg = None
        
        self.purpose = purpose
        self.save_experiment = save_experiment

        # Create experiment folder
        if self.save_experiment:
            self.experiment_number, self.experiment_path, self.json_path = self._init_experiment_path()
        else:
            self.experiment_number, self.experiment_path, self.json_path = -1, None, None

        self.description = '({} Dataset name: {} ({})'.format(self.experiment_number, self.dataset_name, self.purpose)
        self.predictions_df = None
        self.performances_df = None
        self.fitted = False

        # Define colors, level of verbosity, and random_state
        self.verbosity = verbosity 
        self.debug=debug
        self.random_state = random_state

    def fit(self):

        if self.purpose == 'classification':
            #Estimation of the distributions for the positive and negative class
            self.dist_pos = Distributions(dataset=self.dataset_train, 
                                        class_used=1, 
                                        cmap='Blues',
                                        debug=self.debug, 
                                        verbosity=1)

            self.dist_neg = Distributions(dataset=self.dataset_train, 
                                        class_used=0, 
                                        cmap='Greens',
                                        debug=self.debug, 
                                        verbosity=1)

            # Estimate distributions
            self.dist_pos.estimate_pdf(resolution=self.resolution, bandwidth=self.bandwidth)
            self.dist_neg.estimate_pdf(resolution=self.resolution, bandwidth=self.bandwidth)
        
        elif self.purpose == 'estimation':
            #Estimation of the distribution
            self.dist = Distributions(dataset=self.dataset_train, 
                                      class_used=None, 
                                      cmap='Oranges',
                                      verbosity=1)

            # Estimate distributions
            self.dist.estimate_pdf(resolution=self.resolution, bandwidth=self.bandwidth)
        self.fitted = True

    def predict(self):

        assert self.purpose == 'classification', "/!\. Purpose mode is set to `estimation`, you should set it to `classification`. :-)"

        #################################################################
        #  Prediction using maximum likelihood estimation
        #################################################################

        _, step = np.linspace(-2.5,2.5, self.dist_pos.resolution, retstep=True)

        # Contains for each sample of the Test set, the corresponding x and y index coordinates, in the matrix of the 2D pdf... 
        coord_to_index = np.floor_divide(self.dataset_test.X+2.5, step)


        # Init. the array of prediction
        y_pred = np.zeros(shape=self.dataset_test.y.shape[0]); arr = []

        #----------- Treat the case of when both coordinates are known


        # Index of the samples in the test set where both first coordinates are known 
        X_indexes_both_known = np.argwhere((~np.isnan(coord_to_index)).sum(axis=1)==2).squeeze(); arr.extend(X_indexes_both_known)

        # Coordinates of indexes in the feature space of the samples in the test set where both first coordinates are known 
        hat_f_coordinates = coord_to_index[X_indexes_both_known].astype(int)
        inds_array = np.moveaxis(np.array(list(map(tuple, hat_f_coordinates))), -1, 0)

        # Compare likelihood to do the prediction
        y_pred_both_known = (self.dist_pos.f[tuple(inds_array)] > self.dist_neg.f[tuple(inds_array)]).astype(int)

        # Assign predictions 
        y_pred[X_indexes_both_known] = y_pred_both_known


        #----------- Treat the case of when only the first coordinate is known

        # Index of the samples in the test set where only the first coordinate is known 
        X_indexes_first_known = np.argwhere(~np.isnan(coord_to_index[:,0]) & np.isnan(coord_to_index[:,1])).squeeze(); arr.extend(X_indexes_first_known)

        # Coordinates of index in the feature space of the samples in the test set where only the first coordinate is known 
        hat_f_coordinates = coord_to_index[X_indexes_first_known][:,0].astype(int)

        # Compare likelihood to do the prediction
        y_pred_first_known = (self.dist_pos.f_1[hat_f_coordinates] > self.dist_neg.f_1[hat_f_coordinates]).astype(int)

        # Assign predictions 
        y_pred[X_indexes_first_known] = y_pred_first_known

        #----------- Treat the case of when only the second coordinate is known


        # Index of the samples in the test set where only the first coordinate is known 
        X_indexes_second_known = np.argwhere(np.isnan(coord_to_index[:,0]) & ~np.isnan(coord_to_index[:,1])).squeeze(); arr.extend(X_indexes_second_known)

        # Coordinates of index in the feature space of the samples in the test set where only the first coordinate is known 
        hat_f_coordinates = coord_to_index[X_indexes_second_known][:,1].astype(int)

        # Compare likelihood to do the prediction
        y_pred_second_known = (self.dist_pos.f_2[hat_f_coordinates] > self.dist_neg.f_2[hat_f_coordinates]).astype(int)

        # Assign predictions 
        y_pred[X_indexes_second_known] = y_pred_second_known

        print("Sanity check: number of predictions: {} == {}: Num samples\n".format(len(arr), self.dataset_test.y.shape[0])) if self.debug else None

        y_true = self.dataset_test.y.squeeze()

        # Creation of a df for the prediction
        predictions_df = pd.DataFrame({'X1':self.dataset_test.X[:,0], 
                      'X2':self.dataset_test.X[:,1], 
                      'Z1':[1 if not np.isnan(x) else 0 for x in self.dataset_test.X[:,0]],
                      'Z2': [1 if not np.isnan(x) else 0 for x in self.dataset_test.X[:,1]],
                      'y_true': y_true, 
                      'y_pred': y_pred, 
                      'True Positive': [1 if y_true==1 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      'True Negative': [1 if y_true==0 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      'False Positive': [1 if y_true==0 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      'False Negative': [1 if y_true==1 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      })

        
        #Compute metrics of interest  
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc = (tp + tn) / (tp + tn + fp +  fn)
        f1 = 2*tp / (2*tp + fp + fn)
        mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        tpr =  tp / (tp+fn)
        tnr = tn / (tn+fp)
        ppv = tp / (tp+fp)
        npv = tn / (tn+fn)
        fnr = fn / (tp+fn)

        performances_dict = {'Accuracy' : round(acc, 3),
                            'F1 score (2 PPVxTPR/(PPV+TPR))': round(f1, 3),
                            'Matthews correlation coefficient (MCC)': round(mcc, 3),
                            'Sensitivity, recall, hit rate, or true positive rate (TPR)': round(tpr, 3),
                            'Specificity, selectivity or true negative rate (TNR)': round(tnr, 3),
                            'Precision or positive predictive value (PPV)': round(ppv, 3),
                            'Negative predictive value (NPV)': round(npv, 3),
                            'Miss rate or false negative rate (FNR)': round(fnr, 3),
                            'False discovery rate (FDR=1-PPV)': round(1-ppv, 3),
                            'False omission rate (FOR=1-NPV)': round(1-npv, 3)}

        self.predictions_df = predictions_df
        self.performances_df = pd.DataFrame(performances_dict, index=['0'])

        if self.save_experiment:
            self.save()        
            
        return 

    def plot(self):

        if self.purpose == 'classification':
            self.plot_classification()
        elif self.purpose == 'estimation':
            self.plot_estimation()

    def plot_estimation(self):
        # TODOOOOOO
        # Create the pannel 
        fig, axes = plt.subplots(3, 5, figsize=(30, 12));axes = axes.flatten()
        fig.suptitle("({}) Training dataset: {}\n{}".format(int(self.experiment_number), self.dataset_train.dataset_description, self.dataset_train.missingness_description), weight='bold', fontsize=20)


        # Put the dataset 
        axes[1 if df is not None else 2] = self.dataset_test.plot(ax=axes[1 if df is not None else 2])


        # Put the performances
        if df is not None:

                    cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax if ax is not None else None)
        disp.im_.colorbar.remove()        
        print('Sample: {} positive and {} negative samples (#p/#n={:3.0f}%)'.format(tp+fn, tn+fp, 100*(tp+fn)/(tn+fp)))
        for item, value in performances_metrics.items():
            print("  {0:70}\t {1}".format(item, value))


            from utils import my_classification_report
            axes[3], _ = my_classification_report(df['Y'].tolist(), df['Prediction'].tolist(), ax=axes[3])
            
            _ = [ax.legend(prop={'size':10}, loc='lower right') for i,ax in enumerate(axes) if i in [5, 7, 9, ]]; [axes[i].axis('off') for i in range(len(axes)) if i!=3 ]; plt.tight_layout()

        if df is None:
            _ = [axes[i].axis('off') for i in range(len(axes))]; plt.tight_layout()   


        if self.dist_pos is not None:
            self.dist_pos.plot(axes=axes)
            self.dist_neg.plot(axes=axes)
        else:
            self.dist.plot(axes=axes)
        return      

    def plot_classification(self):
    
        # Create the pannel 
        fig, axes = plt.subplots(5, 5, figsize=(20, 14)); axes = axes.flatten()
        fig.suptitle("({}) Training dataset: {}\n{}".format(int(self.experiment_number), self.dataset_train.dataset_description, self.dataset_train.missingness_description), weight='bold', fontsize=12)

        # Plot the dataset 
        axes[2] = self.dataset_test.plot(ax=axes[2])

        # Plot the performances 
        cm = confusion_matrix(self.predictions_df['y_true'].tolist(), self.predictions_df['y_pred'].tolist())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=axes[22])
        disp.im_.colorbar.remove()    
        
        axes = self.dist_pos.plot(axes=axes, predictions_df=self.predictions_df)
        axes = self.dist_neg.plot(axes=axes, predictions_df=self.predictions_df)

        # Plot the distributions of points

        bar = axes[12].bar([0], self.dist_pos.f_0, color = 'tab:blue', label="Pos.");label_bar(bar,axes[12])
        bar = axes[12].bar([1], self.dist_neg.f_0, color = 'tab:green', label="Neg.");label_bar(bar,axes[12])
        axes[12].set_title("H)\nBoth coord. missing");axes[12].set_xlim([-4, 4])
        _ = [ax.legend(prop={'size':10}, loc='lower right') for i,ax in enumerate(axes) if i in [5, 7, 9, 12]]; [axes[i].axis('off') for i in range(len(axes)) if i!=22 ]; plt.tight_layout()

        plt.show()

        #Compute metrics of interest  
        tn, fp, fn, tp = confusion_matrix(self.predictions_df['y_true'].tolist(), self.predictions_df['y_pred'].tolist()).ravel()
        acc = (tp + tn) / (tp + tn + fp +  fn)
        f1 = 2*tp / (2*tp + fp + fn)
        mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        tpr =  tp / (tp+fn)
        tnr = tn / (tn+fp)
        ppv = tp / (tp+fp)
        npv = tn / (tn+fn)
        fnr = fn / (tp+fn) 

        print('Sample: {} positive and {} negative samples (#p/#n={:3.0f}%)\n'.format(tp+fn, tn+fp, 100*(tp+fn)/(tn+fp)))

        display(self.performances_df.transpose())

        #for item, value in performances_metrics.items(): TODOREMMOVE
        #    print("  {0:70}\t {1}".format(item, value))





        return            
    
    def save(self):
        
        #-------- Save dataset ----------#
        self.dataset_train.save(experiment_path=self.experiment_path)
        self.dataset_test.save(experiment_path=self.experiment_path)

        #-------- Save Distributions ----------#
        self.dist_pos.save(experiment_path=self.experiment_path)
        self.dist_neg.save(experiment_path=self.experiment_path)

        #-------- Save json information ----------#

        # Store here the objects that cannot be saved as json objects (saved and stored separately)
        dataset_train = self.dataset_train
        dataset_test = self.dataset_test
        dist_pos = self.dist_pos
        dist_neg = self.dist_neg

        performances_df = self.performances_df
        predictions_df = self.predictions_df

        self.dataset_train = None 
        self.dataset_test = None 
        self.dist_pos = None
        self.dist_neg = None
        self.performances_df = self.performances_df.to_dict(orient='list') if self.performances_df is not None else None

        with open(self.json_path, 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.to_json(orient='records') if type(o) == pd.core.frame.DataFrame else o.__dict__) 

        # Reload the object that were unsaved 
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dist_pos = dist_pos
        self.dist_neg = dist_neg
        self.performances_df = performances_df
        self.predictions_df = predictions_df

        return    
    
    def load(self, previous_experiment=None):
        """
            This functions aims at retrieving the best model from all the experiments. 
            You can either predefince the experiment number and the epoch, or look over all the losses of all experiments and pick the model having the best performances.
            
            /!\ Not implemented yet  
        """

        experiment_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(previous_experiment), 'experiment_log.json')
        dataset_train_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(previous_experiment), 'dataset_train_log.json')
        dataset_test_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(previous_experiment), 'dataset_test_log.json')
        dist_None_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(previous_experiment), 'distributions_None_log.json')
        dist_pos_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(previous_experiment), 'distributions_1_log.json')
        dist_neg_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(previous_experiment), 'distributions_0_log.json')

        #---------------- Loading Experiment  ----------------#

        if os.path.isfile(experiment_path):

            with open(experiment_path) as experiment_json:

                # Load experiment data
                experiment_data = json.load(experiment_json)
                
                # Load experiment attributes
                self._load(experiment_data)
            print("Loaded experiment at '{}'".format(experiment_path)) if self.debug else None
        else:
            print("/!\ No previous experiment found at '{}'".format(experiment_path)) if self.debug else None


        #---------------- Loading Test Dataset   ----------------#


        if os.path.isfile(dataset_test_path):

            with open(dataset_test_path) as data_json:

                # Load experiment data
                dataset_test_data = json.load(data_json)

                self.dataset_test = DatasetGenerator(dataset_name=dataset_test_data['dataset_name'])
                
                # Load experiment attributes
                self.dataset_test.load(dataset_test_data)
            print("Loaded test dataset at '{}'".format(dataset_test_path)) if self.debug else None
        else:
            print("/!\ No previous dataset found at '{}'".format(dataset_test_path)) if self.debug else None

        #---------------- Loading Train Dataset  ----------------#

        if os.path.isfile(dataset_train_path):
    
            with open(dataset_train_path) as data_json:

                # Load experiment data
                dataset_train_data = json.load(data_json)

                self.dataset_train = DatasetGenerator(dataset_name=dataset_train_data['dataset_name'])
                
                # Load experiment attributes
                self.dataset_train.load(dataset_train_data)
            print("Loaded train dataset at '{}'".format(dataset_train_path)) if self.debug else None
        else:
            print("/!\ No previous dataset found at '{}'".format(dataset_train_path)) if self.debug else None


        #---------------- Loading none dist   ----------------#


        if os.path.isfile(dist_None_path):

            with open(dist_None_path) as dist_json:

                # Load experiment data
                dist_data = json.load(dist_json)

                self.dist = Distributions(dataset=self.dataset_train)
                
                # Load experiment attributes
                self.dist.load(dist_data)

            print("Loaded associated distribution with '{}'".format(dist_None_path)) if self.debug else None
        else:
            print("/!\ No previous computed distribution found at '{}'".format(dist_None_path)) if self.debug else None

        #---------------- Loading Pos dist   ----------------#

        if os.path.isfile(dist_pos_path):

            with open(dist_pos_path) as dist_json:

                # Load experiment data
                dist_data = json.load(dist_json)

                self.dist_pos = Distributions(dataset=self.dataset_train)
                
                # Load experiment attributes
                self.dist_pos.load(dist_data)

            print("Loaded associated distribution with '{}'".format(dist_pos_path)) if self.debug else None
        else:
            print("/!\ No previous computed distribution found at '{}'".format(dist_pos_path)) if self.debug else None

        #---------------- Loading Neg dist   ----------------#

        if os.path.isfile(dist_neg_path):
    
            with open(dist_neg_path) as dist_json:

                # Load experiment data
                dist_data = json.load(dist_json)

                self.dist_neg = Distributions(dataset=self.dataset_train)
                
                # Load experiment attributes
                self.dist_neg.load(dist_data)

            print("Loaded associated distribution with '{}'".format(dist_neg_path)) if self.debug else None
        else:
            print("/!\ No previous computed distribution found at '{}'".format(dist_neg_path)) if self.debug else None

        self.predict()      
        
        print("Experiment {} loaded successfully! :-)".format(previous_experiment))
        return  

        

    def _init_experiment_path(self):
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
        if not os.path.isdir(os.path.join(DATA_DIR, 'experiments')):
            os.mkdir(os.path.join(DATA_DIR, 'experiments'))
            
        # Create dataset experiment folder  if not already created
        if not os.path.isdir(os.path.join(DATA_DIR, 'experiments', self.dataset_name)):
            os.mkdir(os.path.join(DATA_DIR, 'experiments', self.dataset_name))

        if not os.path.isdir(os.path.join(DATA_DIR, 'experiments', self.dataset_name, '0')):
            os.mkdir(os.path.join(DATA_DIR, 'experiments', self.dataset_name, '0'))
    
        # Looking for the number of the new experiment number
        experiment_number = np.max([int(os.path.basename(path)) for path in glob(os.path.join(DATA_DIR, 'experiments', self.dataset_name, '*'))])+1

        experiment_path = os.path.join(DATA_DIR, 'experiments', self.dataset_name, str(experiment_number))
        print('Doing experiment {}!'.format(experiment_number))

        # Create experiment folder 
        os.mkdir(experiment_path)

        # Create sub-folders associated to the project
        #os.mkdir(os.path.join(experiment_path, 'dataset'))

        # Create json path
        json_path = os.path.join(experiment_path, 'experiment_log.json')

        return experiment_number, experiment_path, json_path

    def _load(self, data):

        for key, value in data.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else: 
                setattr(self, key, value)
