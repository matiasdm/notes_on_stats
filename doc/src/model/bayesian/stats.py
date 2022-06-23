"""
Tools for statistics data analysis. 
TODO: complete this description. 
----
Refs:
[1] Larry Wasserman, "All of Statistics, A Concise Course in Statistical Inference"
[2] Peter Bruce And Andrew Bruce, "Practical Statistics for Data Scientists"
[3] Alexander Gordon et al., "Control of the Mean Number of False Discoveries, Bonferroni and Stability of Multiple Testing".
[4] Jacob Cohen, "Things I have Learned (So Far)"
[5] Thomas Cover and Joy Thomas, "Elements of Information Theory"
[6] Richard Duda et al., "Pattern Classification"
[7] Judea Pearl et al., "Causal Inference in Statistics"
[8] Steven Kay, "Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory"----
---- 
matias.di.martino.uy@gmail.com,    Durham 2020. 
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from copy import deepcopy       
from const import EPSILON
#from numba.typed import List TODO: work on fighting against the deprecation of list in Numba, cf: https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types


def feature_values_positive_to_negative_ratio(Xp=None, Xn=None, x_range=None, y_range=None, num_bins=50, verbose=1):
    """
    This functions computes the ratio 
     P(x|y=1)/P(x|y=0) and shows it in log scale. This ratio is equivalent to 
     P(x|y=1)/P(x|y=0) = (P(y=1|x)/P(y=0|x))/(P(y=1|x)/P(y=0|x)). This can be interpreted as: how more likely I expect to find a positive samples compared to a random sampling. 
     ----- 
     
    Keyword Arguments:
        Xp {1 or 2D array} -- 1 or 2D array, values of the feature for samples of the positive class. 
        Xn {1 or 2D array} -- 1 or 2D array, values of the feature for samples of the negative class. 
        verbose -- 1 display plots, 
        x_range -- [min_X, max_X] range of X feature. If none min(X) and max(X) is used. 
        y_range -- [min_Y, max_Y] range of Y feature. If none min(Y) and max(Y) is used. 
        num_bins -- Number of bins used when estimating the pdfs. 
    """
    
    if Xp.shape[1]==1:
        return feature_values_positive_to_negative_ratio_1D(Xp=Xp, Xn=Xn, verbose=verbose, x_range=x_range, num_bins=num_bins)
    elif Xp.shape[1]==2:
        return feature_values_positive_to_negative_ratio_2D(Xp=Xp, Xn=Xn, verbose=verbose, x_range=x_range, y_range=y_range, num_bins=num_bins)

def feature_values_positive_to_negative_ratio_1D(Xp=None, Xn=None, x_range=None, num_bins=50, verbose=1):
    """
    This functions computes the ratio 
     P(x|y=1)/P(x|y=0) and shows it in log scale. This ratio is equivalent to 
     P(x|y=1)/P(x|y=0) = (P(y=1|x)/P(y=0|x))/(P(y=1|x)/P(y=0|x)). This can be interpreted as: how more likely I expect to find a positive samples compared to a random sampling. 
     ----- 
     
    Keyword Arguments:
        Xp {1D array} -- 1D array, values of the feature for samples of the positive class. 
        Xn {1D array} -- 1D array, values of the feature for samples of the negative class. 
        verbose -- 1 display plots, 
        x_range -- [min_X, max_X] range of X feature. If none min(X) and max(X) is used. 
    """
    
    # 1) Estimate P(x|y) from the input data. To that end, we follows a frequentist approach, and approximate the pdf by a discrete function, with num_bins number of bins. The domain is set accordint to the x_range.  
    if x_range is None:
        X = np.hstack((Xp,Xn))
        xmin = np.quantile(X,0)  
        xmax = np.quantile(X,1)
        x_range = [xmin, xmax]
        # If the data has noise or outliers, you may want to consider useing 5% lower and upper quantiles to define x_range. Just replace "0" by "0.05" and "1" by "0.95".
        
    num_p = len(Xp)  # Number of positive samples in this set
    num_n = len(Xn)  # Number of negative samples in this set
    
    count_Xp, edges = np.histogram(Xp, bins=num_bins, range=x_range)  # Count samples per bin
    pdf_Xp = count_Xp / num_p  # Normalize to have an estimation of the prob. 
    count_Xn, _     = np.histogram(Xn, bins=num_bins, range=x_range)
    pdf_Xn = count_Xn / num_n

    eps = 1e-5  # A very small number just for numerical stability. 
    Q = np.log10( (pdf_Xp / (pdf_Xn + 1e-5) ) + 1e-5)  # P(x|y=1)/P(x|y=0) in logarithmic scale. 
    
    if verbose>0:  # Show plots 
        # Define a colormap function 
        min_val = -2; max_val = 1.5  # Recall these are in a log scale!
        colormap = define_colormap(min_value=min_val, max_value=max_val, zero=0., num_tones=20)
        plt.subplot(121)  # Plot raw histogram distribution per-class
        plt.bar(edges[:-1], pdf_Xn, width = edges[1]-edges[0], alpha=.5, color=colormap(-1)); 
        plt.bar(edges[:-1], pdf_Xp, width = edges[1]-edges[0], alpha=.5, color=colormap(.7))
        plt.xlabel('X'); plt.ylabel('Estimation of P(X|Y)'), plt.title('Positive and Negative empirical distribution of X')
                       
        plt.subplot(122);  # Plot Q
        colors = [colormap(q) for q in Q]
        plt.bar(edges[:-1], Q , width = edges[1]-edges[0], alpha=.5, color=colors)
        plt.plot(edges, 0*edges, '-k', linewidth=3)
        plt.ylim([min_val,max_val]); plt.xlim(x_range); ax = plt.gca(); plt.grid(axis='y')
        plt.xlabel('X'); plt.ylabel('Q'); plt.title('$Q = log_{10}( P(X|y=1)/P(X|y=-1) )$'); 
        
    return Q

def feature_values_positive_to_negative_ratio_2D(Xp=None, Xn=None, x_range=None, y_range=None, num_bins=50, verbose=1):
    """
    This functions computes the ratio 
     P(x|y=1)/P(x|y=0) and shows it in log scale. This ratio is equivalent to 
     P(x|y=1)/P(x|y=0) = (P(y=1|x)/P(y=0|x))/(P(y=1|x)/P(y=0|x)). This can be interpreted as: how more likely I expect to find a positive samples compared to a random sampling. 
     ----- 
     
    Keyword Arguments:
        Xp {2D array} -- 2D array, values of the feature for samples of the positive class. 
        Xn {2D array} -- 2D array, values of the feature for samples of the negative class. 
        verbose -- 1 display plots, 
        x_range -- [min_X, max_X] range of X feature. If none min(X) and max(X) is used. 
    """
    

    # 1) Estimate P(x|y) from the input data. To that end, we follows a frequentist approach, and approximate the pdf by a discrete function, with num_bins number of bins. The domain is set accordint to the x_range.  
    if x_range is None:
        X_combined = np.concatenate((Xp,Xn))
        xmin, xmax = np.quantile(X_combined[:,0],0), np.quantile(X_combined[:,0],1)
        x_range = [xmin, xmax]
    if y_range is None:
        X_combined = np.concatenate((Xp,Xn))
        ymin, ymax = np.quantile(X_combined[:,1],0), np.quantile(X_combined[:,1],1)
        y_range = [ymin, ymax]        
        # If the data has noise or outliers, you may want to consider useing 5% lower and upper quantiles to define x_range. Just replace "0" by "0.05" and "1" by "0.95".


    num_p = len(Xp)  # Number of positive samples in this set
    num_n = len(Xn)  # Number of negative samples in this set

    count_Xp, xedges, yedges = np.histogram2d(Xp[:,0], Xp[:,1], bins=num_bins, range=[x_range, y_range])  # Count samples per bin
    pdf_Xp = count_Xp / num_p  # Normalize to have an estimation of the prob. 
    count_Xn, _, _ = np.histogram2d(Xn[:,0], Xn[:,1], bins=num_bins, range=[x_range, y_range])  # Count samples per bin
    pdf_Xn = count_Xn / num_n

    eps = 1e-5  # A very small number just for numerical stability. 
    Q = np.log10( (pdf_Xp / (pdf_Xn + 1e-5) ) + 1e-5)  # P(x|y=1)/P(x|y=0) in logarithmic scale. 

    if verbose:
        from matplotlib.colors import LinearSegmentedColormap
        import seaborn as sns

        # This colormap has been designed using the awesome following cite: https://eltos.github.io/gradient/#1F77B4-FFFFFF-FF7F0E
        cmap = LinearSegmentedColormap.from_list('my_gradient', (
            # Edit this gradient at https://eltos.github.io/gradient/#1F77B4-FFFFFF-FF7F0E
            (0.000, (0.122, 0.467, 0.706)),
            (0.500, (1.000, 1.000, 1.000)),
            (1.000, (1.000, 0.498, 0.055))))     
        # Define a colormap function 
        min_val = -2; max_val = 1.5  # Recall these are in a log scale!
        colormap = define_colormap(min_value=min_val, max_value=max_val, zero=0., num_tones=20)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))

        # Plot raw histogram distribution per-class
        ax1 = sns.histplot(x=Xp[:,0], y=Xp[:,1], bins=num_bins, color = colormap(.7), ax=ax1)
        sns.histplot(x=Xn[:,0], y=Xn[:,1], bins=num_bins, color = colormap(-1), ax=ax1)
        ax1.set_xlabel('x'); ax1.set_ylabel('y');ax1.set_title('Positive and Negative empirical distribution of X\nEstimation of P(X|Y)')

        # Plot Q
        xx, yy = np.mgrid[xmin:xmax:complex(0, num_bins), ymin:ymax:complex(0, num_bins)]
        ax2.set_xlim(xmin, xmax);ax2.set_ylim(ymin, ymax)
        # Contourf plot
        cfset = ax2.contourf(xx, yy, Q, cmap=cmap)
        ## Or kernel density estimate plot instead of the contourf plot
        #ax.imshow(np.rot90(Q), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        # Contour plot
        #cset = ax2.contour(xx, yy, Q, colors='k')
        # Label plot
        #ax.clabel(cset, inline=5, fontsize=20)
        cbar = plt.colorbar(cfset);ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_title('$Q = log_{10}( P(X|y=1)/P(X|y=-1) )$'); 
        plt.show()
    return Q

def define_colormap(min_value=-1., max_value=1., zero=0., num_tones=10):
    """
    This is a shortcut to color quantities in tones of blue and orange. In this "ASD" screening examples, we associated orange tones with risk of ASD and blue tones with indications of TD. This function is just to simply this color mapping across different experiments. Zero is the "neutral value", and is mapped to the color white. Max value is the larges value an ASD risk factor can take, this value (and any value above this value) is mapped to the darkest orange. Min value is the lower value the risk indicator can take, and any value lower or equal is mapped to the darkest blue tone. This function returns a mapping function that you can use to compute the color of each new feature values. For example:
    colormap = define_colormap(min_value=-1, max_value=1, zero=0, num_tones=10)
    x = .5
    color_x = colormap(x)  --> returns the color "orange tone" [0.9, 0.5, 0.2, 1.0] and so on. 
    ------ 
    Arguments:
        min_value {float} -- min value associated to the lower asd risk
        max_value {float} -- max value associated to the highest asd risk
        zero {float} -- zero value, associated to the neutral value (no bias)
        num_tones {int} -- number of different tones (the more the more continuos)   
    Returns:
        colormap [function R->R^4] -- Map values to their color. 
    """
    from matplotlib import cm

    blue = cm.get_cmap('Blues', num_tones)   #  grab blue pallette 
    orange = cm.get_cmap('Oranges', num_tones)  # grab orange pallette
    
    def colormap(z):
        if z>zero:
            i = min(max_value-zero, z-zero)/(max_value-zero)  # [zero, max_value] --> [0,1] 
            color = orange(i)
        else:
            i = max(min_value-zero, z-zero)/(min_value-zero)  # [min_value,0] --> [1,0] 
            color = blue(i)
        return color
    
    return colormap

############## Kernel_based_pdf_estimation imputing the missing values 

def kernel_based_pdf_estimation(X,x=None,h=1,verbose=0):
    """
    Estimate the pdf distribution of "X" at "x" using the set of observations X[i,:]. x has lenght k (the dimension of the problem), X has shape nxk (n observations of dimension k). X can have missing values which should be filled with np.nan. A Kernel approximation is computed when the coordinates of the observations are know. If a coordinate is unknown, the contribution of this term is replaced by a weighed average prior computed from the subset of observation for which we have complete data. . 
    
    Example: 
    X = np.random.random((10,3))  # 10 Observations of a 3d problem
    X[0,1] = np.nan; X[4,2] = np.nan  # We don't know some entries. 
    h = .1  # bandwidth of the gaussian kernel
    x = [0.1, 0.1, 0.1]  # where we want to evaluate the pdf (in the 3d space)
    pdf_x = kernel_based_pdf_estimation(X,x=x,h=h,verbose=0)
    print('The prob at {} is {}.format(x,pdf_x))

    """
    n = X.shape[0]  # number "training" samples   
    
    # Define the set of samples for which all the data is available. 
    # (This set is used to compute priors)
    m = [not np.isnan(np.sum(X[i,:])) for i in range(X.shape[0])]
    X_prior = X[m,:]
    hat_f = F(X=X, X_prior=X_prior, x=x, h=h)
    return hat_f

@jit(nopython=True, parallel=True)
def F(X=None, X_prior=None, x=None, h=1, verbose=0):
    n = X.shape[0]  # number "training" samples       
    # Average the contribution of each individual sample. 
    hat_f = []  # init 
    #hat_f = List() TODO: work on fighting against the deprecation of list in Numba, cf: https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    for X_i in X:
        hat_f.append(f_xi(X_prior,X_i,x,h))
        
    hat_f = np.mean(np.array(hat_f))         
        
    return hat_f

@jit(nopython=True, parallel=True)
def f_xi(X_prior, X_i, x, h):
    """
    Contribution of the X_i sample to the estimation of the pdf of X at x. 
    X_prior contains the samples for which there is no missing values, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    k = X_prior.shape[1]  # dimension of the space of samples. 
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel
    
    def W(X_1,X_2):
        """
        Weight between two samples X_1 and X_2 to measure the proximity of the hyperplane (in the case of missing values.)
        W(X_1,X_2) = e^( -1/2 1/h**2 sum_k(x_1k-x_2k)**2 )  for k s.t. x_1k and x_2k isn't nan. 
        """
        def dist(X_1,X_2):
            ks = [i for i in range(len(X_1)) if not np.isnan(X_1[i]) and not np.isnan(X_2[i])]
            if not ks:  # if ks is empty the distance can't be computed
                return np.nan
            
            d = 0
            for k in ks:
                x_1 = X_1[k]
                x_2 = X_2[k]
                d += (x_1-x_2)**2
            return np.sqrt(d)
        
        d = dist(X_1, X_2)
        if np.isnan(d):
            # Is the vectors don't share at least one common coordinate 
            return 0
        
        W = K( d/h )
        return W
        
    # Since we are using a isomorph kernel, each axis can be handeled independently. 
    hat_fi = 1
    coords_missing = np.isnan(X_i)  # unknown coordinates of X_i
    for j in range(k):
        if not coords_missing[j]: # we know the j-th coordinate of X_i 
            # We can compute the contribution of the jth coordinate using the standard term
            hat_fi *= 1/h * K( (x[j]-X_i[j])/h )
        
        if coords_missing[j]:  # we don't know the j-th coordinate, 
            # We use the term associate to the j-th coordinate for the 
            # rest of the samples in the training set (for which the j-th component is know).
            # The contribution of each term is weighted with the distance to the sample hyperplane.
            hat_fip = 0
            Ws = 1e-10  # eps
            for X_p in X_prior:
                w_p = W(X_i, X_p)
                hat_fip += w_p * 1/h * K( (x[j]-X_p[j])/h )
                Ws += w_p
            hat_fip /= Ws
            
            hat_fi *= hat_fip    
    return hat_fi

############## Weighted scheme data imputation            

def impute_missing_data(X_train, X_test, method='custom_imputations', h=.2):
    """
    Imputation of the missing values of the different rows of X_test based on X_train. 
    X_prior contains the samples for which there is no missing values, which are used as prior when coputing the missing coordinate of a sample.
    """

    k = X_test.shape[1]  # dimension of the space of samples. 
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel
    
    if method == 'custom_imputations':
        def W(X_1,X_2):
            """
            Weight between two samples X_1 and X_2 to measure the proximity of the hyperplane (in the case of missing values.)
            W(X_1,X_2) = e^( -1/2 1/h**2 sum_k(x_1k-x_2k)**2 )  for k s.t. x_1k and x_2k isn't nan. 
            """
            def dist(X_1,X_2):
                ks = [i for i in range(len(X_1)) if not np.isnan(X_1[i]) and not np.isnan(X_2[i])]
                if not ks:  # if ks is empty the distance can't be computed
                    return np.nan

                d = 0
                for k in ks:
                    x_1 = X_1[k]
                    x_2 = X_2[k]
                    d += (x_1-x_2)**2
                return np.sqrt(d)

            d = dist(X_1, X_2)
            if np.isnan(d):
                # Is the vectors don't share at least one common coordinate 
                return 0

            W = K( d/h )
            return W
            
        # Compute prior set.
        m = [not np.isnan(np.sum(X_train[i,:])) for i in range(X_train.shape[0])]
        X_prior = X_train[m,:]

        # Init. the imputed test set.
        imp_X_test = deepcopy(X_test)

        # TODO: Keep track of the weights ? As a measure of confidence for the imputation ?

        for i, X_i in enumerate(X_test):

            # Perform imputation if needed
            coords_missing = np.isnan(X_i)  # unknown coordinates of X_i
            for j in range(k):        
                if coords_missing[j]:  # we don't know the j-th coordinate, we need to impute it

                    # We use the term associate to the j-th coordinate for the 
                    # rest of the samples in the training set (for which the j-th component is know).
                    # The contribution of each term is weighted with the distance to the sample hyperplane.
                    hat_X_ij = 0
                    Ws = 1e-10  # eps
                    for X_p in X_prior:
                        w_p = W(X_i, X_p)
                        hat_X_ij += w_p * X_p[j]
                        Ws += w_p
                    hat_X_ij /= Ws

                    imp_X_test[i, j] = hat_X_ij

    elif method=='naive':
            # Ignore missing values
        from stats import kernel_based_pdf_estimation
        imp_X_test = X_test[~np.isnan(X_test[:,0]),:]
        imp_X_test = imp_X_test[~np.isnan(imp_X_test[:,1]),:]        
                

    elif method=='mean':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_X_test = imp.fit_transform(X_test)
      
    elif method=='median':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_X_test = imp.fit_transform(X_test)
      
    elif method=='knn':
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer()
        imp_X_test = knn_imputer.fit_transform(X_test)
        
    elif method=='mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
        imp_X_test = imp.fit_transform(X_test)

        
    return imp_X_test



############## Kernel_based_pdf_estimation_with_missing_priors 

def kernel_based_pdf_estimation_z_prior(X, x=None, h=.2, verbose=0):
    """
    Estimate the pdf distribution of "X" at "x" using the set of observations X[i,:]. x has lenght k (the dimension of the problem), X has shape nxk (n observations of dimension k). 
    X can have missing values which should be filled with np.nan. A Kernel approximation is computed when the coordinates of the observations are know. 
    If a coordinate is unknown, the contribution of this term is replaced by a prior on the missingness distribution of that coordinate, computed from all the other samples. 
    
    Example: 
    X = np.random.random((10,3))  # 10 Observations of a 3d problem
    X[0,1] = np.nan; X[4,2] = np.nan  # We don't know some entries. 
    h = .1  # bandwidth of the gaussian kernel
    x = [0.1, 0.1, 0.1]  # where we want to evaluate the pdf (in the 3d space)
    pdf_x = kernel_based_pdf_estimation(X,x=x,h=h,verbose=0)
    print('The prob at {} is {}.format(x,pdf_x))

    """
    # Computation of the missingness priors: [P(Z_1 = 1), ... ,  P(Z_K = 1)]
    Z_prior = np.array([np.mean(~np.isnan(X[:,i])) for i in range(X.shape[1])])
    hat_f = F_z_prior(X=X, Z_prior=Z_prior, x=x, h=h)
    return hat_f

#@jit(nopython=True, parallel=True)
def F_z_prior(X=None, Z_prior=None, x=None, h=.2, verbose=0):
    """
    Computation of the pdf at x, using the prior on the missingness meachanism of each features. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    # number "training" samples 
    n = X.shape[0]      

    # init 
    hat_f = []  

    # Compute contribution of each samples to the estimation of the pdf at point x 
    for X_i in X:
        hat_f.append(f_xi_z_prior(X_i, x, Z_prior, h))
    hat_f = np.mean(np.array(hat_f))         
    return hat_f

#@jit(nopython=True, parallel=True)
def f_xi_z_prior(X_i, x, Z_prior, h):
    """
    Contribution of the X_i sample to the estimation of the pdf of X at x. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    k = Z_prior.shape[0]  # dimension of the space of samples. 
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel

    # Since we are using a isomorph kernel, each axis can be handeled independently. 
    hat_fi = 1
    coords_missing = np.isnan(X_i)  # unknown coordinates of X_i

    for j in range(k):
        if not coords_missing[j]: # we know the j-th coordinate of X_i 
            # We can compute the contribution of the jth coordinate using the standard term
            hat_fi *= 1/h * K( (x[j]-X_i[j])/h )

        if coords_missing[j]:  # we don't know the j-th coordinate, 
            # We use prior on the missingness mechanism associated to this coordinates. 
            hat_fi *= Z_prior[j]
    return hat_fi



####### Kernel_based_pdf_estimation_with_missing_priors_limited_range


def kernel_based_pdf_estimation_z_prior_limited_range(X, x=None, h=.2, put_weight=1, verbose=0):
    """
    Estimate the pdf distribution of "X" at "x" using the set of observations X[i,:]. x has lenght k (the dimension of the problem), X has shape nxk (n observations of dimension k). 
    X can have missing values which should be filled with np.nan. A Kernel approximation is computed when the coordinates of the observations are know. 
    If a coordinate is unknown, the contribution of this term is replaced by a prior on the missingness distribution of that coordinate, computed from all the other samples. *The contribution is limited to the range of values, based on the rest of the dataset.*
    
    Example: 
    X = np.random.random((10,3))  # 10 Observations of a 3d problem
    X[0,1] = np.nan; X[4,2] = np.nan  # We don't know some entries. 
    h = .1  # bandwidth of the gaussian kernel
    x = [0.1, 0.1, 0.1]  # where we want to evaluate the pdf (in the 3d space)
    pdf_x = kernel_based_pdf_estimation_z_prior_limited_range(X,x=x,h=h,verbose=0)
    print('The prob at {} is {}.format(x,pdf_x))

    """

    # Computation of the missingness priors: [P(Z_1 = 1), ... ,  P(Z_K = 1)]
    Z_prior = np.array([np.mean(~np.isnan(X[:,i])) for i in range(X.shape[1])])
    hat_f = F_z_prior_limited_range(X=X, Z_prior=Z_prior, put_weight=put_weight, x=x, h=h)
    return hat_f


#@jit(nopython=True, parallel=True)
def F_z_prior_limited_range(X=None, Z_prior=None, x=None, put_weight=0,  h=.2, verbose=0):
    """
    Computation of the pdf at x, using the prior on the missingness meachanism of each features. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    # number "training" samples 
    n = X.shape[0]      

    # init 
    hat_f = []  

    # Compute contribution of each samples to the estimation of the pdf at point x 
    for X_i in X:
        hat_f.append(f_xi_z_prior_limited_range(X_i, x, Z_prior, put_weight, h))
    hat_f = np.mean(np.array(hat_f))         
    return hat_f

#@jit(nopython=True, parallel=True)
def f_xi_z_prior_limited_range(X_i, x, Z_prior, put_weight, h):
    """
    Contribution of the X_i sample to the estimation of the pdf of X at x. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    k = Z_prior.shape[0]  # dimension of the space of samples. 
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel

    # Since we are using a isomorph kernel, each axis can be handeled independently. 
    hat_fi = 1
    coords_missing = np.isnan(X_i)  # unknown coordinates of X_i

    for j in range(k):
        if not coords_missing[j]: # we know the j-th coordinate of X_i 
            # We can compute the contribution of the jth coordinate using the standard term
            hat_fi *= 1/h * K( (x[j]-X_i[j])/h )

        if coords_missing[j]:  # we don't know the j-th coordinate, 
            if put_weight: # We put weight only if this part of the space is explored by the distribution.
                # We use prior on the missingness mechanism associated to this coordinates.
                hat_fi *= Z_prior[j]
            else:
                hat_fi *= 0
    return hat_fi



####### Kernel_based_pdf_estimation using the side spaces

def kernel_based_pdf_estimation_side_spaces(X, x=None, h=.2, verbose=0):
    """
    Estimate the pdf distribution of "X" at "x" using the set of observations X[i,:]. x has lenght k (the dimension of the problem), X has shape nxk (n observations of dimension k). 
    X can have missing values which should be filled with np.nan. A Kernel approximation is computed when the coordinates of the observations are know. 
    If a coordinate is unknown, the contribution of this term is replaced by a prior on the missingness distribution of that coordinate, computed from all the other samples. *The contribution is limited to the range of values, based on the rest of the dataset.*
    
    Example: 
    X = np.random.random((10,3))  # 10 Observations of a 3d problem
    X[0,1] = np.nan; X[4,2] = np.nan  # We don't know some entries. 
    h = .1  # bandwidth of the gaussian kernel
    x = [0.1, 0.1, 0.1]  # where we want to evaluate the pdf (in the 3d space)
    pdf_x = kernel_based_pdf_estimation_z_prior_limited_range(X,x=x,h=h,verbose=0)
    print('The prob at {} is {}.format(x,pdf_x))

    """

    hat_f, hat_f_0, hat_f_1, hat_f_2 = F_side_spaces(X=X, x=x, h=h)
    return hat_f, hat_f_0, hat_f_1, hat_f_2

#@jit(nopython=True, parallel=True)
def F_side_spaces(X=None, x=None, h=.2, verbose=0):
    """
    Computation of the pdf at x, using the prior on the missingness meachanism of each features. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    # number "training" samples 
    n = X.shape[0]      

    # init 
    hat_f = []
    hat_f_0 = []
    hat_f_1 = []  
    hat_f_2 = []  

    # Compute contribution of each samples to the estimation of the pdf at point x 
    for X_i in X:
        contribution_f, contribution_f_0, contribution_f_1, contribution_f_2 = f_xi_side_spaces(X_i, x, h)
        hat_f.append(contribution_f)
        hat_f_0.append(contribution_f_0)
        hat_f_1.append(contribution_f_1)
        hat_f_2.append(contribution_f_2)

    hat_f = np.mean(np.array(hat_f))   
    hat_f_0 = np.mean(np.array(hat_f_0))   
    hat_f_1 = np.mean(np.array(hat_f_1))      
    hat_f_2 = np.mean(np.array(hat_f_2))  

    return hat_f, hat_f_0, hat_f_1, hat_f_2

#@jit(nopython=True, parallel=True)
def f_xi_side_spaces(X_i, x, bandwidth):
    """
    Contribution of the X_i sample to the estimation of the pdf of X at x. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    k = X_i.shape[0]  # dimension of the space of samples. 
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel
    # Define the switch function between coordinates. It is needed as if X1 is missing, we need to build the distribution of x_2 knowing xi is missing, so the other line...
    # The switch function in general should send 0 to 1 and 1 to 0 in 2D. In 3D, 0 should be sent to [1,2], 1 to [0,2] and 2 to [0,1].
    switch = lambda j: -j+1

    # Since we are using a isomorph kernel, each axis can be handeled independently. 
    hat_fi = 1

    hat_fi_0 = 0
    hat_fi_1 = 0
    hat_fi_2 = 0

    # Handle the case where both are missing
    if np.isnan(X_i[0]) and np.isnan(X_i[1]):
        hat_fi_0 = 1 

    # Handle the case where both known
    elif not np.isnan(X_i[0]) and not np.isnan(X_i[1]):
        for j in range(k):
            # We can compute the contribution of the jth coordinate using the standard term
            hat_fi *= 1/bandwidth * K( (x[j]-X_i[j])/bandwidth )

    # Handle the case where X1 is missing
    elif np.isnan(X_i[0]) and not np.isnan(X_i[1]):

        hat_fi_2 = 1/bandwidth * K( (x[1]-X_i[1]) /bandwidth )


    else:

        hat_fi_1 = 1/bandwidth * K( (x[0]-X_i[0])/bandwidth )

    return hat_fi, hat_fi_0, hat_fi_1, hat_fi_2


####### Kernel_based_pdf_estimation using the side spaces but doing imputation for the joint probability distributio

def kernel_based_pdf_estimation_side_spaces_imputation(X, x=None, h=.2, verbose=0):
    """
    Estimate the pdf distribution of "X" at "x" using the set of observations X[i,:]. x has lenght k (the dimension of the problem), X has shape nxk (n observations of dimension k). 
    X can have missing values which should be filled with np.nan. A Kernel approximation is computed when the coordinates of the observations are know. 
    If a coordinate is unknown, the contribution of this term is replaced by a prior on the missingness distribution of that coordinate, computed from all the other samples. *The contribution is limited to the range of values, based on the rest of the dataset.*
    
    Example: 
    X = np.random.random((10,3))  # 10 Observations of a 3d problem
    X[0,1] = np.nan; X[4,2] = np.nan  # We don't know some entries. 
    h = .1  # bandwidth of the gaussian kernel
    x = [0.1, 0.1, 0.1]  # where we want to evaluate the pdf (in the 3d space)
    pdf_x = kernel_based_pdf_estimation_z_prior_limited_range(X,x=x,h=h,verbose=0)
    print('The prob at {} is {}.format(x,pdf_x))

    """
    n = X.shape[0]  # number "training" samples   
    
    # Define the set of samples for which all the data is available. 
    # (This set is used to compute priors)
    m = [not np.isnan(np.sum(X[i,:])) for i in range(X.shape[0])]
    X_prior = X[m,:]

    hat_f, hat_f_0, hat_f_1, hat_f_2 = F_side_spaces_imputation(X=X, X_prior=X_prior, x=x, h=h)
    return hat_f, hat_f_0, hat_f_1, hat_f_2

#@jit(nopython=True, parallel=True)
def F_side_spaces_imputation(X=None, X_prior=None, x=None, h=.2, verbose=0):
    """
    Computation of the pdf at x, using the prior on the missingness meachanism of each features. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """

    # init 
    hat_f = []
    hat_f_0 = []
    hat_f_1 = []  
    hat_f_2 = []  

    # Compute contribution of each samples to the estimation of the pdf at point x 
    for X_i in X:
        contribution_f, contribution_f_0, contribution_f_1, contribution_f_2 = f_xi_side_spaces_imputation(X_i, X_prior, x, h)
        hat_f.append(contribution_f)
        hat_f_0.append(contribution_f_0)
        hat_f_1.append(contribution_f_1)
        hat_f_2.append(contribution_f_2)

    hat_f = np.mean(np.array(hat_f))   
    hat_f_0 = np.mean(np.array(hat_f_0))   
    hat_f_1 = np.mean(np.array(hat_f_1))      
    hat_f_2 = np.mean(np.array(hat_f_2))  

    return hat_f, hat_f_0, hat_f_1, hat_f_2

@jit(nopython=True, parallel=True)
def f_xi_side_spaces_imputation(X_i, X_prior=None, x, bandwidth):
    """
    Contribution of the X_i sample to the estimation of the pdf of X at x. 
    Z_priors contains the empirical probability that feature j/k is missing, which are used as prior when the contribution of sample with partially missing data is calculated.
    """
    k = X_prior.shape[1]  # dimension of the space of samples. 
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel
    
    def W(X_1,X_2):
        """
        Weight between two samples X_1 and X_2 to measure the proximity of the hyperplane (in the case of missing values.)
        W(X_1,X_2) = e^( -1/2 1/h**2 sum_k(x_1k-x_2k)**2 )  for k s.t. x_1k and x_2k isn't nan. 
        """
        def dist(X_1,X_2):
            ks = [i for i in range(len(X_1)) if not np.isnan(X_1[i]) and not np.isnan(X_2[i])]
            if not ks:  # if ks is empty the distance can't be computed
                return np.nan
            
            d = 0
            for k in ks:
                x_1 = X_1[k]
                x_2 = X_2[k]
                d += (x_1-x_2)**2
            return np.sqrt(d)
        
        d = dist(X_1, X_2)
        if np.isnan(d):
            # Is the vectors don't share at least one common coordinate 
            return 0
        
        W = K( d/h )
        return W

    # Since we are using a isomorph kernel, each axis can be handeled independently. 
    hat_fi = 1

    hat_fi_0 = 0
    hat_fi_1 = 0
    hat_fi_2 = 0

    # Handle the case where both are missing
    if np.isnan(X_i[0]) and np.isnan(X_i[1]):
        hat_fi_0 = 1 

    # Handle the case where both known
    elif not np.isnan(X_i[0]) and not np.isnan(X_i[1]):
        for j in range(k):
            # We can compute the contribution of the jth coordinate using the standard term
            hat_fi *= 1/bandwidth * K( (x[j]-X_i[j])/bandwidth )

    # Handle the case where X1 is missing
    elif np.isnan(X_i[0]) and not np.isnan(X_i[1]):

        hat_fi_2 = 1/bandwidth * K( (x[1]-X_i[1]) /bandwidth )


    else:

        hat_fi_1 = 1/bandwidth * K( (x[0]-X_i[0])/bandwidth )

    return hat_fi, hat_fi_0, hat_fi_1, hat_fi_2
