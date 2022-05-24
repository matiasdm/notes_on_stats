
import os
import sys
import json
from glob import glob

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Import local packages
from const import *


def estimate_pdf(X=None, method='our', resolution=20, bandwidth=None):

    xygrid = np.meshgrid(np.linspace(-2.5,2.5,resolution),np.linspace(-2.5,2.5,resolution))
    H,W = xygrid[0].shape
    hat_f = np.zeros_like(xygrid[0])  # init. the pdf estimation
    h = bandwidth

    if method=='our' or method=='custom_imputations':
        # See documentation
        from stats import kernel_based_pdf_estimation
    
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(X=X, x=[x,y], h=h)
                
    if method=='missing':
        # See documentation
        from stats import kernel_based_pdf_estimation_z_prior

        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation_z_prior(X=X, x=[x,y] ,h=h)  

    if method=='missing_limited_range':
        # See documentation
        from stats import kernel_based_pdf_estimation_z_prior_limited_range

        # Compute the space mask to be sure not to add contribution on expty space, based on the resolution of the space
        m = [not np.isnan(np.sum(X[i,:])) for i in range(X.shape[0])]
        X_prior = X[m,:]
        hist2d, _, _ = np.histogram2d(X_prior[:,0], X_prior[:,1], bins=[np.linspace(-2.5,2.5,resolution), np.linspace(-2.5,2.5,resolution)])
        hist2d_up = np.concatenate([np.concatenate([hist2d, np.zeros((1, W-1))], axis=0), np.zeros((H, 1))], axis=1)
        mask_space = hist2d_up>0
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation_z_prior_limited_range(X=X, x=[x,y], put_weight=mask_space[i,j], h=h)  

    if method=='no_imputations':

        #----------------------------------------------------------------------------------
        #  Estimation of f(X_1,X_2|Z_1=1, Z_2=1), f(X_2|Z_1=0,Z_2=1) and f(X_1|Z_1=1,Z_2=0)
        #----------------------------------------------------------------------------------

        from stats import kernel_based_pdf_estimation_side_spaces

        hat_f_0 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_1 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_2 = np.zeros_like(xygrid[0])  # init. the pdf estimation

        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                # Computing contribution on coordinates i, j of hat_f, and coordinate i of hat_f_1 and coordinate j of hat_f_2
                hat_f[i,j], hat_f_0[i,j], hat_f_1[i,j], hat_f_2[i,j] =  kernel_based_pdf_estimation_side_spaces(X=X, x=[x, y], h=h)
                
        # Average the contribution of all i's and j's coordinate
        hat_f_0 = np.mean(hat_f_0)

        # Average the contribution of all j's coordinate on this horizontal line
        hat_f_1 = np.mean(hat_f_1, axis=0)
        
        # Average the contribution of all i's coordinate to form the vertical line
        hat_f_2 = np.mean(hat_f_2, axis=1) 

        # Normalization of the distributions
        hat_f /= (hat_f.sum()+EPSILON);hat_f_1 /= (hat_f_1.sum()+EPSILON);hat_f_2 /= (hat_f_2.sum()+EPSILON)

        return hat_f, hat_f_0, hat_f_1, hat_f_2
                
    if method=='naive':
        # Ignore missing values
        from stats import kernel_based_pdf_estimation
        imp_X = X[~np.isnan(X[:,0]),:]
        imp_X = imp_X[~np.isnan(imp_X[:,1]),:]        
                
    if method=='mean':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_X = imp.fit_transform(X)
      
    if method=='median':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_X = imp.fit_transform(X)
      
    if method=='knn':
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer()
        imp_X = knn_imputer.fit_transform(X)
        
    if method=='mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
        imp_X = imp.fit_transform(X)
       
    if method in ['mice', 'knn', 'median', 'mean', 'naive']:
        from stats import kernel_based_pdf_estimation
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(imp_X, x=[x,y],h=h)

    # Except for the `no_imputations` method, return only hat_f

    return hat_f



