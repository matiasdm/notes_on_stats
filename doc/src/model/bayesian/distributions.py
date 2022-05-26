import sys
import json
from time import time


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


# add tools path and import our own tools
sys.path.insert(0, '../../src')

from const import *
from utils import fi, label_bar, repr
from .utils import estimate_pdf
from generateToyDataset import DatasetGenerator

class Distributions(object):


    def __init__(self, dataset, class_used=None, approach='multi_distributions', cmap='Blues', verbosity=1, debug=False, random_state=RANDOM_STATE):

        # Dataset 
        self.dataset_name = dataset.dataset_name
        self.dataset = dataset
        self.class_used = class_used
        self.approach = approach

        # Kernel Density estimation parameters
        bandwidth = None 
        resolution = None


        # Outputs
        self.f = None
        self.f_0 = None 
        self.f_1 = None; self.f_2 = None
        self.f_1_marginal = None; self.f_2_marginal = None
        self.f_z1 = None; self.f_z2 = None

        self.computed = False
        self.estimation_time = None

        # Define colors, level of verbosity, and random_state
        self.cmap = cmap# sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))
        self.verbosity = verbosity 
        self.debug=debug
        self.random_state = random_state

    def __call__(self):
        return repr(self)

    def estimate_pdf(self, resolution=20, bandwidth=.2):

        assert self.dataset.X_train is not None, "/!\. You need to generate missing data first.\n call `experiment.dataset.generate_missing_coordinates(missingness_mechanism='MCAR')` for instance! :-)"

        self.bandwidth = bandwidth
        self.resolution = resolution

        #---------------------- Verbose ----------------------#

        if self.debug or self.verbosity > 1:

            if self.class_used is not None:
                print("Estimating distribution for class {}.".format(int(self.class_used)))
            else:
                print("Estimating distributions for both classes.")

        #---------------------- Select correct set of data for the estimation ------------#

        # For training/estimation, we may want self.X and self.y to be the ones of a certain class. 
        X = self.dataset.X_train[(self.dataset.y_train==self.class_used).squeeze()]

        #---------------------- Estimation ----------------------#
        start_time = time()

        if self.approach == 'multi_distributions':

            # Estimation all the distributions
            self.f, self.f_0, self.f_1, self.f_2  = estimate_pdf(X=X, approach=self.approach, imputation_method=self.dataset.imputation_method, bandwidth=self.bandwidth, resolution=self.resolution)

        elif self.approach == 'single_distribution':

            self.f = estimate_pdf(X=X, approach=self.approach, imputation_method=self.dataset.imputation_method,  bandwidth=self.bandwidth, resolution=self.resolution)

            # Computation of the prior when no data at all are available.
            self.f_0 = np.sum(np.isnan(X).sum(axis=1) == X.shape[1]) / X.shape[0]

        else:
            raise ValueError("Please use 'multi_distributions', 'single_distribution' approach.")


        #----------------------------------------------------------------------------------
        # Computation of the other distributions
        #----------------------------------------------------------------------------------

        # Computation of the marginals
        self.f_1_marginal = self.f.sum(axis=0)
        self.f_2_marginal = self.f.sum(axis=1)

        if self.approach == 'multi_distributions':
            # Z_prior reflects P(Z_1=1, Z_2=1, ... Z_k=1)
            Z_prior = np.array([np.mean(~np.isnan(X[:,i])) for i in range(X.shape[1])])

            # Estimation of f(Z_1=0|X_2)
            self.f_z1 = self.f_2_marginal * Z_prior[0]/(self.f_2_marginal * Z_prior[0]  + self.f_2*(1-Z_prior[0]))
            self.f_z1 /= (self.f_z1.sum()+EPSILON)

            # Estimation of f(Z_1=1|X_2)
            self.f_z1_bar = self.f_2 * (1-Z_prior[0])/(self.f_2_marginal * Z_prior[0]  + self.f_2*(1-Z_prior[0]))
            self.f_z1_bar /= (self.f_z1_bar.sum()+EPSILON)

            # Estimation of f(Z_2=0|X_1)
            self.f_z2 = self.f_1_marginal * Z_prior[1]/(self.f_1_marginal * Z_prior[1]  + self.f_1*(1-Z_prior[0]))
            self.f_z2 /= self.f_z2.sum()

            # Estimation of f(Z_2=1|X_1)
            self.f_z2_bar = self.f_1 * (1-Z_prior[1])/(self.f_1_marginal * Z_prior[1]  + self.f_1*(1-Z_prior[0]))
            self.f_z2_bar /= (self.f_z2_bar.sum()+EPSILON)

        end_time = time()
        self.estimation_time = end_time - start_time
        self._print_time() if (self.debug or self.verbosity > 1)  else None

        self.computed = True

    def plot(self, axes=None, predictions_df=None, **kwargs):
        
        if axes is None:
            fig, axes = plt.subplots(3, 5, figsize=(30, 12));axes = axes.flatten()

        if self.class_used is None or self.class_used == 1:
            shift = 0
        elif self.class_used == 0:
            shift = 10

        alpha=.3

        # Plot the distributions common to every distributions-based approaches
        axes[6+shift].imshow(self.f_2_marginal[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[6+shift].set_title("B)\nf(X_2|Z_1=1)")
        axes[7+shift].imshow(self.f, cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -2.5, 2.5]);axes[7].set_title("C)\nf(X_1, X_2|Z_1=1, Z_2=1)")
        axes[8+shift].imshow(self.f_1_marginal[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[8+shift].set_title("D)\nf(X_1|Z_2=1)")

        if self.approach == 'multi_distributions':

            axes[5+shift].imshow(self.f_2[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[5+shift].set_title("A)\nf(X_2|Z_1=0)")

            axes[9+shift].imshow(self.f_1[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[9+shift].set_title("E)\nf(X_1|Z_2=0)")
            
            axes[10+shift].imshow(self.f_z1_bar[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[10+shift].set_title("F)\nf(Z_1=0|X_2)")
            axes[11+shift].imshow(self.f_z1[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[11+shift].set_title("G)\nf(Z_1=1|X_2)")
            axes[13+shift].imshow(self.f_z2[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[13+shift].set_title("I)\nf(Z_2=1|X_1)")
            axes[14+shift].imshow(self.f_z2_bar[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[14+shift].set_title("J)\nf(Z_2=0|X_1)")        

        if predictions_df is not None and self.class_used == 1:

            # plot on the A) plot the sample having only X2 
            axes[5 if self.approach == 'multi_distributions' else 6].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1")['X2'], 
                            color='b', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1"))))
            axes[5 if self.approach == 'multi_distributions' else 6].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")['X2'], 
                        color='g', alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1"))))
            axes[5 if self.approach == 'multi_distributions' else 6].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")),  
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")['X2'], 
                            color='r', alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1"))))

            axes[7].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `True Positive`==1")['X1'], 
                            predictions_df.query(" `Z1`==1 and  `Z2`==1 and `True Positive`==1")['X2'], 
                            color='b', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `True Positive`==1"))))
            axes[7].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X1'], 
                            predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X2'], 
                        color='g', alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1"))))
            axes[7].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X1'],  
                            predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X2'], 
                            color='r', alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1"))))

            axes[9 if self.approach == 'multi_distributions' else 8].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1")['X1'],
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1")),  
                            color='b', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1"))))
            axes[9 if self.approach == 'multi_distributions' else 8].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")['X1'], 
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")),
                            color='g', alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1"))))
            axes[9 if self.approach == 'multi_distributions' else 8].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")['X1'], 
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")),  
                            color='r', alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1"))))

            bar = axes[12].bar([0], self.f_0, color = 'tab:blue', label="Pos. {} TP {} FP".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==0 and `True Positive`==1")), len(predictions_df.query(" `Z1`==0 and  `Z2`==0 and `False Positive`==1")) ));label_bar(bar,axes[12])


        elif predictions_df is not None and self.class_used == 0:
            # plot on the A) plot the sample having only X2 
            axes[5+shift if self.approach == 'multi_distributions' else 6+shift].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1")['X2'], 
                            color='g', label="TN (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1"))))
            axes[5+shift if self.approach == 'multi_distributions' else 6+shift].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")['X2'], 
                        color='b', label="FN (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1"))))
            axes[5+shift if self.approach == 'multi_distributions' else 6+shift].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")),  
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")['X2'], 
                            color='r', label="FP (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1"))))


            axes[7+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `True Negative`==1")['X1'], 
                            predictions_df.query(" `Z1`==1 and  `Z2`==1 and `True Negative`==1")['X2'], 
                            color='g', label="TN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `True Negative`==1"))))
            axes[7+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X1'], 
                            predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X2'], 
                        color='b', label="FN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1"))))
            axes[7+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X1'],  
                            predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X2'], 
                            color='r', label="FP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1"))))

            axes[9+shift if self.approach == 'multi_distributions' else 8+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1")['X1'],
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1")),  
                            color='g', label="TN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1"))))
            axes[9+shift if self.approach == 'multi_distributions' else 8+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")['X1'], 
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")),
                            color='b', label="FN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1"))))
            axes[9+shift if self.approach == 'multi_distributions' else 8+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")['X1'], 
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")),  
                            color='r', label="FP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")))) 

            bar = axes[12].bar([1], self.f_0, color = 'tab:green', label="Neg. {} TN {} FN".format( len(predictions_df.query(" `Z1`==0 and  `Z2`==0 and `True Negative`==1")), len(predictions_df.query(" `Z1`==0 and  `Z2`==0 and `False Negative`==1")) ));label_bar(bar,axes[12])
            
            axes[12].set_title("H)\nBoth coord. missing");axes[12].set_xlim([-4, 4])
    
        return axes

    def save(self, experiment_path):
            
        #-------- Save dataset ----------#
        with open(os.path.join(experiment_path, 'distributions_{}_log.json'.format(self.class_used)), 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.to_json(orient='records') if type(o) == pd.core.frame.DataFrame else o.__dict__)

    def load(self, dataset_data):

        for key, value in dataset_data.items():

            if key != 'dataset':
                if isinstance(value, list):
                    setattr(self, key, np.array(value))
                else: 
                    setattr(self, key, value)
    
    def _print_time(self):

        hours, rest = divmod( self.estimation_time, 3600)
        minutes, seconds = divmod(rest, 60)
        print("Done. ({}h {}m and {:.2f}s)".format(int(hours), int(minutes), seconds))

