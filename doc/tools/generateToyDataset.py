import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle 
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

class DatasetGenerator(object):
    """
    A class used to generate toy datasets with potential missing data. 
    
    Available missingness mechanisms include:
    
    1) Missing Completely at Random (MCAR). 
    
    kwargs TODO: WRONG!!!! THIS IS MAR!!!
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

    
    def __init__(self, name, num_samples=1000, ratio_of_missing_values=.2, imbalance_ratio=.5, num_samples_gt=2000, verbosity=1, random_state=47):
        
        self.name = name
        self.num_samples = num_samples
        self.num_samples_gt = num_samples_gt
        
        
        self.ratio_of_missing_values = ratio_of_missing_values
        self.imbalance_ratio = imbalance_ratio
        
        self.missingness_mechanism = None
        self.missing_one_coordinates = None
        self.missing_first_quarter = None
        self.ratio_missing_per_class = None
        self.mask_missing = None
        
        self.X_gt = None
        self.y_gt = None
        
        self.X_raw = None
        self.X = None
        self.y = None
        
        self.description = 'Number of samples: {}\n'.format(self.num_samples)
        self.cmap = sns.color_palette(plt.get_cmap('tab20')(np.arange(0,2)))
        self.verbosity = verbosity    
        self.random_state = random_state
        np.random.seed(random_state)
        
        
        ################################
        # Generate the positive examples
        ################################
        if name=='moons':
            X_all, labels = datasets.make_moons(n_samples=int(2*imbalance_ratio*(num_samples+num_samples_gt)), noise=.05, random_state=random_state)
        elif name=='circles':
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
        if name=='moons':
            X_all, labels = datasets.make_moons(n_samples=int(2*(1-imbalance_ratio)*(num_samples+num_samples_gt)), noise=.05, random_state=random_state)
        elif name=='circles':
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
        
        
    def generate_missing_coordinates(self, missingness_mechanism='MCAR', missing_one_coordinates=False, missing_first_quarter=True, ratio_missing_per_class=[.1, .5]):
        
        
        self.missingness_mechanism = missingness_mechanism
        self.missing_one_coordinates = missing_one_coordinates
        self.missing_first_quarter = missing_first_quarter
        self.ratio_missing_per_class = ratio_missing_per_class
        
        self.X = deepcopy(self.X_raw)
                
        if missingness_mechanism == 'MCAR':

            # Simulate missing samples
            for i in range(self.X.shape[0]):  # randomtly remove features
                if np.random.random() < self.ratio_of_missing_values:
                    self.X[i,0] = np.nan

                if np.random.random() < self.ratio_of_missing_values:
                    self.X[i,1] = np.nan
                    
            self.mask_missing = np.isnan(self.X)
            self.description += 'MCAR {}% missing'.format(int(self.ratio_of_missing_values*100))
                        
        elif missingness_mechanism == 'MAR':

            if missing_one_coordinates:
                # Simulate missing samples
                for i in range(self.X.shape[0]):  # randomly remove features
                    if np.random.random() < self.ratio_of_missing_values:
                        if np.random.random() < .5:  # remove samples from x or y with 
                            # equal probability
                            self.X[i,0] = np.nan
                        else:
                            self.X[i,1] = np.nan
                            
                self.description += 'MAR {}% missing\nMissing one coordinate or the other'.format(int(self.ratio_of_missing_values*100))
                            
            elif missing_first_quarter:
                while (np.isnan(self.X).sum(axis=1) > 0).sum()/self.num_samples < self.ratio_of_missing_values:
                    # Simulate missing samples
                    for i in range(self.X.shape[0]):  # randomly remove features
                        if self.X[i,0] > 0 and self.X[i,1] > 0 and  np.random.random() < self.ratio_of_missing_values:
                            if np.random.random() < .5:  # remove samples from x or y with 
                                # equal probability
                                self.X[i,0] = np.nan
                            if np.random.random() < .5:  # remove samples from x or y with 
                                self.X[i,1] = np.nan
                self.description += 'MAR {}% missing\nMissing only on the first quarter'.format(int(self.ratio_of_missing_values*100))
                
        elif missingness_mechanism == 'MNAR':
            
            if missing_first_quarter:
                for label in [0, 1]:
                    while (np.isnan(self.X[(self.y==label).squeeze()]).sum(axis=1) > 0).sum()/(self.y==label).sum() < self.ratio_missing_per_class[label]:
                        
                        # Simulate missing samples
                        for i in range(self.X.shape[0]):  # randomly remove features
                            
                            if self.y[i]==label and self.X[i,0] > 0 and self.X[i,1] > 0 and  np.random.random() < self.ratio_of_missing_values:
                                if np.random.random() < .5:  # remove samples from x or y with 
                                    # equal probability
                                    self.X[i,0] = np.nan
                                if np.random.random() < .5:  # remove samples from x or y with 
                                    self.X[i,1] = np.nan     
                self.description += 'MNAR {}% positive class missing\n {}% negative class missing\nMissing only on the first quarter'.format(100*self.ratio_missing_per_class[0],
                                                                                                                   100*self.ratio_missing_per_class[1])
                                    
            else:
                for label in [0, 1]:
                    while (np.isnan(self.X[(self.y==label).squeeze()]).sum(axis=1) > 0).sum()/(self.y==label).sum() < self.ratio_missing_per_class[label]:

                        # Simulate missing samples
                        for i in range(self.X.shape[0]):  # randomly remove features

                            if self.y[i]==label and np.random.random() < self.ratio_of_missing_values:
                                if np.random.random() < .5:  # remove samples from x or y with 
                                    # equal probability
                                    self.X[i,0] = np.nan
                                if np.random.random() < .5:  # remove samples from x or y with 
                                    self.X[i,1] = np.nan   
                self.description += 'MNAR {}% positive class missing\n {}% negative class missing'.format(100*self.ratio_missing_per_class[0],
                                                                                                                   100*self.ratio_missing_per_class[1])            
        self.mask_missing = np.isnan(self.X)
                    
        return self.X, self.y
    
    def get_data(self):
        return self.X, self.X_gt, self.y, self.y_gt 
        
    def plot(self, verbosity=1):

        colors, colors_gt = [self.cmap[0] if l==1 else self.cmap[1] for l in self.y], [self.cmap[0] if l==1 else self.cmap[1] for l in self.y_gt]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax1.scatter(self.X_gt[:,0], self.X_gt[:,1], c=colors_gt);ax1.axis('off');ax1.set_title("Ground Truth\n{}% imbalance ratio\n".format(int(self.imbalance_ratio*100)), weight='bold')
        ax2.scatter(self.X[:,0], self.X[:,1], c=colors);ax2.axis('off');ax2.set_title(self.description, weight='bold')
        
        if verbosity > 0:
            ax2.scatter(self.X_raw[(self.mask_missing[:,0]) & (~self.mask_missing[:,1]), 0], self.X_raw[(self.mask_missing[:,0]) & (~self.mask_missing[:,1]), 1], c='g' if self.verbosity==4 else 'r', alpha=.7, label='Missing X1 ({})'.format(((self.mask_missing[:,0]) & (~self.mask_missing[:,1])).sum()))
            ax2.scatter(self.X_raw[(~self.mask_missing[:,0]) & (self.mask_missing[:,1]), 0], self.X_raw[(~self.mask_missing[:,0]) & (self.mask_missing[:,1]), 1], c='purple' if self.verbosity==4 else 'r',alpha=.7, label='Missing X2 ({})'.format(((~self.mask_missing[:,0]) & (self.mask_missing[:,1])).sum()))
            ax2.scatter(self.X_raw[(self.mask_missing[:,0]) & (self.mask_missing[:,1]), 0], self.X_raw[(self.mask_missing[:,0]) & (self.mask_missing[:,1]), 1], c='r', alpha=.7, label='Missing both ({})'.format(((self.mask_missing[:,0]) & (self.mask_missing[:,1])).sum()))
            ax2.legend(prop={'size':10}, loc='lower left')
        plt.show()
        
    

