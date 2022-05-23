import sys
import json
from time import time


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


# add tools path and import our own tools
sys.path.insert(0, '../tools')

from const import *
from utils import fi, label_bar
from generateToyDataset import DatasetGenerator

class Distributions(object):


    def __init__(self, dataset, class_used=None, cmap='Blues', verbosity=1, debug=False, random_state=RANDOM_STATE):

        # Dataset 
        self.dataset_name = dataset.dataset_name
        self.dataset = dataset
        self.class_used = class_used

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

    # Decorator function that switch the data of the dataset from class 0 or 1, and reset it properly at the end.
    def switchdataset(func):
        def switch(self, **kargs):

            self.dataset.subset(self.class_used)
            output = func(self,**kargs)
            self.dataset.reset()
            return output

        return switch


    @switchdataset
    def estimate_pdf(self, resolution=20, bandwidth=.2):

        assert self.dataset.X is not None, "/!\. You need to generate missing data first.\n call `experiment.dataset.generate_missing_coordinates(missingness_mechanism='MCAR')` for instance! :-)"

        self.bandwidth = bandwidth
        self.resolution = resolution

        if self.class_used is not None:
            print("Estimating distribution for class {}.".format(int(self.class_used))) if self.debug else None
        else:
            print("Estimating distributions for both classes.") if self.debug else None

        from stats import kernel_based_pdf_estimation_xz


        start_time = time()

        # Estimation all the distributions
        self.f, self.f_0, self.f_1, self.f_2, self.f_1_marginal, self.f_2_marginal, self.f_z1, self.f_z2, self.f_z1_bar, self.f_z2_bar = kernel_based_pdf_estimation_xz(X=self.dataset.X, h=bandwidth, resolution=resolution, verbose=0)

        end_time = time()
        self.estimation_time = end_time - start_time
        self._print_time() if self.debug else None

        self._compute_distance()

        self.computed = True

    @switchdataset
    def plot(self, axes=None, predictions_df=None, **kwargs):
        
        if axes is None:
            fig, axes = plt.subplots(3, 5, figsize=(30, 12));axes = axes.flatten()

        if self.class_used is None or self.class_used == 1:
            shift = 0
        elif self.class_used == 0:
            shift = 10

        alpha=.3

        # Plot the distributions

        axes[5+shift].imshow(self.f_2[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[5+shift].set_title("A)\nf(X_2|Z_1=0)")
        axes[6+shift].imshow(self.f_2_marginal[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[6+shift].set_title("B)\nf(X_2|Z_2=1)")
        axes[7+shift].imshow(self.f, cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -2.5, 2.5]);axes[7].set_title("C)\nf(X_1, X_2|Z_1=1, Z_2=1)")
        axes[8+shift].imshow(self.f_1_marginal[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[8+shift].set_title("D)\nf(X_1|Z_1=1)")
        axes[9+shift].imshow(self.f_1[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[9+shift].set_title("E)\nf(X_1|Z_2=0)")
        
        axes[10+shift].imshow(self.f_z1_bar[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[10+shift].set_title("F)\nf(Z_1=0|X_2)")
        axes[11+shift].imshow(self.f_z1[:,None].repeat(2, axis=1), cmap=self.cmap, origin='lower', extent=[-.5, .5, -2.5, 2.5]);axes[11+shift].set_title("G)\nf(Z_1=1|X_2)")
        axes[13+shift].imshow(self.f_z2[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[13+shift].set_title("I)\nf(Z_2=1|X_1)")
        axes[14+shift].imshow(self.f_z2_bar[None, :].repeat(2, axis=0), cmap=self.cmap, origin='lower', extent=[-2.5, 2.5, -.5, .5]);axes[14+shift].set_title("J)\nf(Z_2=0|X_1)")        

        if predictions_df is not None and self.class_used == 1:

            # plot on the A) plot the sample having only X2 
            axes[5].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1")['X2'], 
                            color='b', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1"))))
            axes[5].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")['X2'], 
                        color='g', alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1"))))
            axes[5].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")),  
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

            axes[9].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1")['X1'],
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1")),  
                            color='b', alpha=alpha, label="TP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1"))))
            axes[9].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")['X1'], 
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")),
                            color='g', alpha=alpha, label="FP (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1"))))
            axes[9].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")['X1'], 
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")),  
                            color='r', alpha=alpha, label="FN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1"))))

            bar = axes[12].bar([0], self.f_0, color = 'tab:blue', label="Pos. {} TP {} FP".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==0 and `True Positive`==1")), len(predictions_df.query(" `Z1`==0 and  `Z2`==0 and `False Positive`==1")) ));label_bar(bar,axes[12])


        elif predictions_df is not None and self.class_used == 0:
            # plot on the A) plot the sample having only X2 
            axes[5+shift].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1")['X2'], 
                            color='g', label="TN (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1"))))
            axes[5+shift].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")), 
                            predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")['X2'], 
                        color='b', label="FN (n={})".format(len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1"))))
            axes[5+shift].scatter([0]*len(predictions_df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")),  
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

            axes[9+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1")['X1'],
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1")),  
                            color='g', label="TN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1"))))
            axes[9+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")['X1'], 
                            [0]*len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")),
                            color='b', label="FN (n={})".format(len(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1"))))
            axes[9+shift].scatter(predictions_df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")['X1'], 
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
    
    @switchdataset
    def _compute_distance(self):
        self.f_1_jensenshannon = distance.jensenshannon(self.f_1, self.f_1_marginal, 2.0)
        self.f_1_wasserstein = wasserstein_distance(self.f_1, self.f_1_marginal)
        self.f_2_jensenshannon = distance.jensenshannon(self.f_2, self.f_2_marginal, 2.0)
        self.f_2_wasserstein = wasserstein_distance(self.f_2, self.f_2_marginal)

    def _print_time(self):

        hours, rest = divmod( self.estimation_time, 3600)
        minutes, seconds = divmod(rest, 60)
        print("Done. ({}h {}m and {:.2f}s)".format(int(hours), int(minutes), seconds))

