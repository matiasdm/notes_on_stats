import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm
from glob import glob
import json 
from time import time

from scipy.spatial import distance
from scipy.stats import wasserstein_distance


# add tools path and import our own tools
sys.path.insert(0, '../tools')

from const import *
from utils import fi
from generateToyDataset import DatasetGenerator

class Experiment(object):
    def __init__(self, dataset, previous_experiment=None, create_experiment=True, verbosity=1, debug=False, random_state=47):

        # Set definitions attributes (also used for log purposes)
        self.dataset_name = dataset.dataset_name
        self.create_experiment=create_experiment


        if previous_experiment is not None:
            self.retrieve_experiment(previous_experiment)
            return

        # Dataset 
        self.dataset = dataset

        # Kernel Density estimation parameters
        bandwidth = None 
        resolution = None
            

        # Create experiment folder
        if self.create_experiment:
            self.experiment_number, self.experiment_path, self.json_path = self._init_experiment_path()
        else:
            self.experiment_number, self.experiment_path, self.json_path = -1, None, None

        self.description = '({} Dataset name: {}\n'.format(self.experiment_number, self.dataset_name)
        self.computed = False


        # Define colors, level of verbosity, and random_state
        self.cmap = sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))
        self.verbosity = verbosity 
        self.debug=debug
        self.random_state = random_state

        # Initial saving of the experiment
        if self.create_experiment:
            self.save_experiment()

        # Outputs
        self.f = None
        self.f_0 = None 
        self.f_1 = None; self.f_2 = None
        self.f_1_marginal = None; self.f_2_marginal = None
        self.f_z1 = None; self.f_z2 = None
        
        self.estimation_time = None

        self.f_1_jensenshannon = None;self.f_1_wasserstein = None
        self.f_2_jensenshannon = None;self.f_2_wasserstein = None

    def generate_missing_data(missingness_mechanism='MCAR', ratio_of_missing_values=RATIO_OF_MISSING_VALUES, missing_first_quarter=False, missing_X1=False, missing_X2=False, ratio_missing_per_class=[.1, .5], missingness_pattern=None, verbosity=1):
        """
        Example of the currently implemented Missingness mechanisms and settings.


        # There are no mutual information between Z and X, Y
        self.generate_missing_coordinates(missingness_mechanism='MCAR')

        # There are mutual information between Z and X (if X_1=0 then Z_2=1 and vice-versa), but not between Z and Y
        self.generate_missing_coordinates(missingness_mechanism='MAR', allow_missing_both_coordinates=False)

        # There are mutual information between Z and X (Z_1 and Z_2 depend on X_1 and X_2), but not between Z and Y
        self.generate_missing_coordinates(missingness_mechanism='MAR', missing_first_quarter=True)

        # There are no mutual information between Z and X, but there are between Z and Y (one class has higher rate of missing value)
        self.generate_missing_coordinates(missingness_mechanism='MNAR', missing_first_quarter=False, ratio_missing_per_class=[.1, .5])

        # There are mutual information between Z and X (Z_1 and Z_2 depend on X_1 and X_2), and between Z and Y (one class has higher rate of missing value)
        self.generate_missing_coordinates(missingness_mechanism='MNAR', missing_first_quarter=True, ratio_missing_per_class=[.1, .5])

        """



        self.dataset.generate_missing_coordinates(missingness_mechanism=missingness_mechanism, ratio_of_missing_values=ratio_of_missing_values, missing_first_quarter=missing_first_quarter, missing_X1=missing_X1, missing_X2=missing_X2, ratio_missing_per_class=ratio_missing_per_class, missingness_pattern=missingness_pattern, debug=debug, verbosity=verbosity)
        return 

    def estimate_pdf(self, resolution=20, bandwidth=.2):

        assert self.dataset.X is not None, "/!\. You need to generate missing data first.\n call `experiment.dataset.generate_missing_coordinates(missingness_mechanism='MCAR')` for instance! :-)"

        self.bandwidth = bandwidth
        self.resolution = resolution

        from stats import kernel_based_pdf_estimation_xz


        start_time = time()

        # Estimation all the distributions
        self.f, self.f_0, self.f_1, self.f_2, self.f_1_marginal, self.f_2_marginal, self.f_z1, self.f_z2, self.f_z1_bar, self.f_z2_bar = kernel_based_pdf_estimation_xz(X=self.dataset.X, h=bandwidth, resolution=resolution, verbose=0)

        end_time = time()
        self.estimation_time = end_time - start_time
        self._print_time()

        self._compute_distance()
        if self.create_experiment:
            self.save_experiment()
            
    def retrieve_experiment(self, experiment_number=None):
        """
            This functions aims at retrieving the best model from all the experiments. 
            You can either predefince the experiment number and the epoch, or look over all the losses of all experiments and pick the model having the best performances.
            
            /!\ Not implemented yet  
        """
        experiment_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(experiment_number), 'experiment_log.json')
        dataset_path = os.path.join(DATA_DIR,  'experiments', self.dataset_name, str(experiment_number), 'dataset_log.json')

        if os.path.isfile(experiment_path) and os.path.isfile(dataset_path):

            with open(experiment_path) as experiment_json:

                # Load experiment data
                experiment_data = json.load(experiment_json)
                
                # Load experiment attributes
                self._load_attributes(experiment_data)

            with open(dataset_path) as data_json:

                # Load experiment data
                dataset_data = json.load(data_json)

                self.dataset = DatasetGenerator(dataset_name=experiment_data['dataset_name'])
                
                # Load experiment attributes
                self.dataset.load(dataset_data)

            print("Loaded previous experiment '{}'".format(experiment_path))

        else:

            print("/!\ No previous experiment found at '{}'".format(experiment_path))
        return  
    
    def save_experiment(self):
        

        #-------- Save dataset ----------#
        self.dataset.save(experiment_path=self.experiment_path)

        #-------- Save json information ----------#

        # Store here the objects that cannot be saved as json objects (saved and stored separately)
        dataset = self.dataset
        
        self.dataset = None

        with open(self.json_path, 'w') as outfile:
            json.dump(self, outfile, default=lambda o: o.astype(float) if type(o) == np.int64 else o.tolist() if type(o) == np.ndarray else o.__dict__)

        # Reload the object that were unsaved 
        self.dataset = dataset

        return

    def summary(self):
        #TODO IMPLEMENT BETTER 
        for item in list(self.__dict__.keys()):
            print("  {0:40}\t {1}".format(item, self.__getattribute__(item)))
        return

    def plot(self):

        # Plot first the dataset that
        self.dataset.plot()

        cmap = 'Blues'

        fig, axes = plt.subplots(2, 5, figsize=(20, 8));axes = axes.flatten()
        fig.suptitle("({}) {}{}".format(int(self.experiment_number), self.dataset.dataset_description, self.dataset.missingness_description), weight='bold', fontsize=20)
        axes[0].imshow(self.f_2[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[0].set_title("A)\nf(X_2|Z_1=0)")
        axes[1].imshow(self.f_2_marginal[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[1].set_title("B)\nf(X_2|Z_2=1)")
        axes[2].imshow(self.f, cmap=cmap, origin='lower');axes[2].set_title("C)\nf(X_1, X_2|Z_1=1, Z_2=1)")
        axes[3].imshow(self.f_1_marginal[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[3].set_title("D)\nf(X_1|Z_1=1)")
        axes[4].imshow(self.f_1[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[4].set_title("E)\nf(X_1|Z_2=0)")
        
        axes[5].imshow(self.f_z1_bar[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[5].set_title("F)\nf(Z_1=0|X_2)")
        axes[6].imshow(self.f_z1[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[6].set_title("G)\nf(Z_1=1|X_2)")
        axes[8].imshow(self.f_z2[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[8].set_title("H)\nf(Z_2=1|X_1)")
        axes[9].imshow(self.f_z2_bar[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[9].set_title("I)\nf(Z_2=0|X_1)")

        _ = [ax.axis('off') for ax in axes]; plt.tight_layout()

        axes[7].text(0.5,0.5, "P(Z_1=0, Z_2=0)={:.3f}%".format(100*self.f_0), size=18, ha="center", transform=axes[7].transAxes)
        plt.show()

        fi(20, 3)
        plt.plot(np.linspace(-2.5,2.5, self.resolution), self.f_1, color='tab:blue',label='Computed on the hyperplane')
        plt.plot(np.linspace(-2.5,2.5, self.resolution), self.f_1_marginal, color='tab:orange', label='Marginal')
        #plt.title("f(X_1|...)\nJS dist.: {:.3f}\nWasss. dist.: {:.2e}".format(self.f_1_jensenshannon, self.f_1_wasserstein), weight='bold', fontsize=18)TODO
        plt.title("f(X_1|Z_2=1) (blue) and f(X_1|Z_2=0) (orange) \nJS dist.: {:.3f}\nWasss. dist.: {:.2e}".format(distance.jensenshannon(self.f_1, self.f_1_marginal, 2.0), wasserstein_distance(self.f_1, self.f_1_marginal)), weight='bold', fontsize=18)
        plt.legend()
        fi(20, 3)
        plt.plot(np.linspace(-2.5,2.5, self.resolution), self.f_2, color='tab:blue', label='Computed on the hyperplane')
        plt.plot(np.linspace(-2.5,2.5, self.resolution), self.f_2_marginal, color='tab:orange', label='Marginal')
        #plt.title("f(X_2|...)\nJS dist.: {:.3f}\nWasss. dist.: {:.2e}".format(self.f_2_jensenshannon, self.f_2_wasserstein), weight='bold', fontsize=18)TODO
        plt.title("f(X_2|Z_1=0)  (blue) and f(X_2|Z_1=0) (orange) \nJS dist.: {:.3f}\nWasss. dist.: {:.2e}".format(distance.jensenshannon(self.f_2, self.f_2_marginal, 2.0), wasserstein_distance(self.f_2, self.f_2_marginal)), weight='bold', fontsize=18)
        plt.legend()
        plt.show()
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

    def _load_attributes(self, data):

        for key, value in data.items():
            if isinstance(value, list):
                setattr(self, key, np.array(value))
            else: 
                setattr(self, key, value)

    def _compute_distance(self):
        self.f_1_jensenshannon = distance.jensenshannon(self.f_1, self.f_1_marginal, 2.0)
        self.f_1_wasserstein = wasserstein_distance(self.f_1, self.f_1_marginal)
        self.f_2_jensenshannon = distance.jensenshannon(self.f_2, self.f_2_marginal, 2.0)
        self.f_2_wasserstein = wasserstein_distance(self.f_2, self.f_2_marginal)

    def _print_time(self):

        hours, rest = divmod( self.estimation_time, 3600)
        minutes, seconds = divmod(rest, 60)
        print("Computation time was {}h {}m and {:.2f}s".format(int(hours), int(minutes), seconds))


