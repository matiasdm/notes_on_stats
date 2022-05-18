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
from const import EPSILON
#from numba.typed import List TODO: work on fighting against the deprecation of list in Numba, cf: https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types

def kernel_based_empirical_risk(X, h):
    """
    Kernel Risk estimation of the pdf of X. We use a Gaussian kernel, X is the data, and h the bandwidth. See [1] pag 316.
    Arguments:
        X {[type]} -- data
        h {[type]} -- bandwidth
    """
    # Define shortcuts for K K2 and K_ast
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel N(0,1)
    K2 = lambda u: 1/np.sqrt(2*np.pi * 2) * np.exp(-u**2 / (2 * 2))  # Define the kernel with sigma^2=2
    K_ast = lambda u: K2(u) - 2*K(u)    
    n = len(X)
    J = 2/(n*h) * K(0)  #  Left term [1] pag. 316, Eq (20.25)
    for X_i in X:
        for X_j in X:
            J += 1 / (h * n**2) * K_ast( (X_i-X_j) / h )
    return J


def compute_kernel_opt_bandwidth(X, x, h_range=None, verbose=0):
    """
    Compute gaussian kernel pdf estimator optimal bandwidth. 
    """
    if h_range is None:
        h_range = np.linspace(0.05,1.5,20)
    
    # Compute the estimation of the cross-validation estimation risk
    J = [kernel_based_empirical_risk(X,hh) for hh in h_range]
    
    # Get the bandwidth that minimizes the empirical error. 
    h_opt = h_range[np.argmin(J)]  # get opt bandwidth
    
    if verbose>0:  # show results. 
        plt.plot(h_range, J); plt.xlabel('h'); plt.ylabel('$\hat{J(h)}$')
        plt.scatter(h_opt, min(J), 150, color='orange')
         
    return h_opt


def kernel_estimator(X,x=None,h=1):
    """
    Kernel estimation of the pdf of X. We use a Gaussian kernel, X is the data, x the values where
    f(x) is estimated, and h the bandwidth. See [1] pag 312.
    """
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel
    n = len(X)  # number of samples    
    hat_f = np.zeros_like(x)  # init.
    for j, x_j in enumerate(x):
        for X_i in X:
            hat_f[j] += 1/(n*h) * K( (x_j-X_i) / h )
    return hat_f


def compute_kernel_estimation_conf(X, hat_f=None, x=None, h=None, alpha=0.05):
    """Compute the kernel estimation confidence. See [1] pag 316 for details. 
    
    Arguments:
        X {1d array} -- The empirical data X_i
    
    Keyword Arguments:
        hat_f {1d array} -- The estimation of f, hat_f (default: {None})
        x {1d array} -- The points x in which f(x) is estimated (default: {None})
        h {float} -- the kernel bandwidth h (default: {None})
        alpha {float} -- the level of confidence for the interval (default: {0.05})
    
    Returns:
        l,u -- the lower and upper alpha,1-alpha confidence interval for hat_f.
    """
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)  # Define the kernel N(0,1)
    from scipy.stats import norm
    m = 3*h
    q = norm.ppf( ( 1 + (1-alpha)**(1/m) ) / (2) )  # ppf(x) = Phi^-1(x)  Phi = cdf_{N(0,1)}
    n = len(X)
    
    l = np.zeros_like(hat_f); u = np.zeros_like(hat_f)  # init.
    
    for j,xx in enumerate(x):  # for each x coordinate compute l(x) and u(x)
        Y_i = np.array([ 1/h * K((xx-XX)/h) for XX in X]) 
        bar_Y = np.mean(Y_i)
        s_square = 1 / (n-1) * np.sum( (Y_i-bar_Y)**2 )
        se = np.sqrt(s_square) / np.sqrt(n)
        
        l[j] = hat_f[j] - q*se
        u[j] = hat_f[j] + q*se
    return l, u


def compute_kernel_estimator_and_conf_interval(X, x=None, xmin=None, xmax=None, h=None, 
                                               alpha=0.05, verbose=0):
    """
    Compute the kernel based pdf estimation and the associated (alpha, 1-alpha) confidence interval. (See [1] chapter 20)
    Arguments
        X {[list or 1d array]} -- the data. 
    Keyword Arguments:
        x {[1d array]} -- where the pdf is estimated (x coords for "f(x)")
        xmin {[type]} -- [description] (default: {xmin})
        xmax {[type]} -- [description] (default: {xmax})
        alpha {[type]} -- [description] (default: {alpha})
        num_bins {[type]} -- [description] (default: {num_bins})
    """
    if xmin is None:
        xmin = min(X)
    if xmax is None:
        xmax = max(X)
    if x is None: 
        x = np.linspace(xmin,xmax,100)
    if h is None: 
        print('Estimating optimal kernel bandwidth... ')
        h = compute_kernel_opt_bandwidth(X,x)
        print('Optimal h: {:3.2f}'.format(h))
    
    # Estimate the kernel density estimator 
    hat_f = kernel_estimator(X,x=x,h=h)
    # Estimate the confidence interval associated to it
    l, u = compute_kernel_estimation_conf(X, hat_f=hat_f, x=x, h=h, alpha=alpha)

    if verbose>0:  # show the results
        plt.bar(x, hat_f, width=x[1]-x[0], color='orange', alpha=.2)    
        plt.plot(x, l, '--', linewidth=2, color='orange')
        plt.plot(x, u, '--', linewidth=2, color='orange')
        plt.title('PDF estimation with 95% CI')
    return hat_f, l, u


def compute_histogram_and_conf_interval(X, xmin=None, xmax=None, alpha=0.05, 
                                        num_bins=100, verbose=0):
    """
    Compute the histogram and the associated (alpha, 1-alpha) confidence interval. (See [1] pag 311)
    Arguments
        X {[list or 1d array]} -- the data. 
    Keyword Arguments:
        xmin {[type]} -- [description] (default: {xmin})
        xmax {[type]} -- [description] (default: {xmax})
        alpha {[type]} -- [description] (default: {alpha})
        num_bins {[type]} -- [description] (default: {num_bins})
    """
    if xmin is None:
        xmin = min(X)
    if xmax is None:
        xmax = max(X)
        
    # ------------------- #
    # See [1] theorem 20.10
    def compute_hist_conf_constant(alpha, m, n):   
        def z_u(u):  # Compute the upper u quantile of N(0,1)
            from scipy.stats import norm
            z = norm.ppf(1-u)  # z_u = Phi^-1(1-u)  with Phi = cdf_{N(0,1)}
            return z
        c = z_u(alpha/(2*m))/2 * np.sqrt(m/n)
        return c
    lf = lambda hat_f,c: (max( np.sqrt(hat_f)-c, 0 ))**2
    uf = lambda hat_f,c: (np.sqrt(hat_f)+c)**2
    # ------------------- #

    h = (xmax-xmin)/ num_bins;  # bin size
    n = len(X)  # number of samples 
    c = compute_hist_conf_constant(alpha, num_bins, n)  # coef for the interval confidence. 
    # Compute hist and conf. intervals. 
    count, edges = np.histogram(X, range=[xmin, xmax], bins=num_bins)  
    prob_bin = count/n
    hat_f = prob_bin/h  
    lower_bound = [lf(f,c) for f in hat_f]
    upper_bound = [uf(f,c) for f in hat_f]
    
    if verbose>0:  # plot results
        plt.bar(edges[:-1]+h/2, hat_f, width=h, color='orange', alpha=.2)    
        plt.plot(edges[:-1]+h/2, lower_bound, '--', color='orange', linewidth = 2)  
        plt.plot(edges[:-1]+h/2, upper_bound, '--', color='orange', linewidth = 2)  
        plt.xlim([xmin, xmax])
    return hat_f, lower_bound, upper_bound


def bin_size_risk_estimator(X, num_bins_range=[1,100], xmin=None, xmax=None, verbose=0):
    """
    Compute the cross-validation estimator risk to find the optimal number of bins for the histogram. (Check [1] pag. 310.)
    Arguments:
        X {list of 1D array} -- the data   
    Keyword Arguments:
        num_bins_range {[int,int]} -- Range for the number of bins (default: {[1,100]})
    Returns:
        J [list] -- List of risks.
        opt_num_bins -- number of bins that minimizes J.  
    """
    
    def hat_J(hat_p, n): 
        """
        Compute the cross-validation estimator of the risk. 
        hat_p: is N_i/n, where N_i is the number of datapoints in the ith bin
        n: is the number of datapoints
        """
        m = len(hat_p)  # the number of p_i is the number of bins. 
        h = 1/m  # When theorem 2.1 is obtained, the data is assumed to be mapped to the range [0,1]
        # therefore, when m bins are selected, the bin width is h=1/m
        J = 2 / ((n-1)*h) - (n+1) / (n-1) / h  * np.sum(np.square(hat_p))
        # WARNING:in [1] the definition is incorrect in Eq. (20.14). $h$ is missing in the second term, I checked with the original reference** where is correct Eq. (2.8) 
        # **Mats Rudemo. Empirical Choice of Histograms and Kernel Density Estimators. 1981
        return J
    
    # If xmin and xmax were not provided, get the min and max of the data. 
    if xmin is None: 
        xmin = min(X)
    if xmax is None: 
        xmax = max(X)
    n = len(X)  # number of samples 
    
    num_bins_set = np.arange(num_bins_range[0],num_bins_range[1])
    hat_J_n = []

    for (i,num_bins) in enumerate(num_bins_set):
        # Count empirical points per-bin 
        count, edges = np.histogram(X, range=[xmin, xmax], bins=num_bins)  
        # Convert count to an estimation of the prob(x in B)
        prob_bin = count/n; 
        # Compute the approximated Risk
        hat_J_i = hat_J(prob_bin, n)
        hat_J_n.append(hat_J_i)
    
    # Get the optimal value 
    # Smooth the signal to make the estimation more reliable
    from scipy.ndimage import gaussian_filter1d 
    smoothing_sigma = 1.5
    hat_J_n_filt = gaussian_filter1d(hat_J_n, smoothing_sigma)  # smooth to remove numerical noise
    
    opt_num_bins_idx = np.argmin(hat_J_n_filt)  # Find the index that minimized the expresion
    opt_num_bins = num_bins_set[opt_num_bins_idx]  # Get the optimal number of bins
    
    if verbose>0:  # Show some of the variables and outputs
        plt.xlim([min(num_bins_set), max(num_bins_set)]); plt.title('$\hat{J}(m)$')
        plt.legend(['raw', 'filtered', 'optimum']); plt.xlabel('Number of bins')
        plt.ylabel('Risk estimation')

        plt.plot(num_bins_set, hat_J_n, linewidth=2, color='steelblue')  # Show the raw risk estimation
        plt.plot(num_bins_set, hat_J_n_filt, '--', linewidth=2, color='steelblue')  # smoothed version
        plt.scatter(opt_num_bins, hat_J_n_filt[opt_num_bins_idx],150, color='orange')  # opt.
    
    return hat_J, opt_num_bins 


def estimate_cdf(X, num_bins=100, alpha=0.05, xmin=None, xmax=None, verbose=0):
    """
    Compute the empirical distribution F_i (estimation of the prob X < b_i). It also provides the alpha, 1-alpha confidence intervals. 
    ref: Wasserman, "All of statistics"
    Arguments:
        X {list or 1D array} -- list with the data
        num_bins {int} -- number of bins
        alpha {float} -- confidence interval 
    Returns:
        p_i -- histogram
        u_i -- upper edge of the confidence interval 
        l_i -- lower edge of the confidence interval
    """
    # Define shortcuts: 
    cdf_epsilon = lambda n, alpha: np.sqrt( (1/(2*n)) * np.log(2/alpha) )
    U = lambda hatF, epsilon: min(hatF + epsilon, 1)
    L = lambda hatF, epsilon: max(hatF - epsilon, 0)

    # If xmin and xmax were not provided, get the min and max of the data. 
    if xmin is None: 
        xmin = min(X)
    if xmax is None: 
        xmax = max(X)
    n = len(X)  # number of samples 
    
    # Count empirical points per-bin 
    count, edges = np.histogram(X, range=[xmin, xmax], bins=num_bins)  
    # Convert count to an estimation of the prob(x in B)
    prob_bin = count/n
    # Compute the cumulative probability
    hatF = prob_bin.cumsum()
    
    # Compute the confidence interval 
    epsilon = cdf_epsilon(n, alpha)    
    lower_bound = [L(hatF_i, epsilon) for hatF_i in hatF]  
    upper_bound = [U(hatF_i, epsilon) for hatF_i in hatF]  
    
    if verbose>0:
        h = (xmax-xmin)/num_bins
        # Plot the estimation
        plt.bar(edges[:-1]+h/2, hatF, width=h, color='orange', alpha=.2)
        plt.plot(edges[:-1]+h/2, lower_bound, '--', color='orange', linewidth = 2)  
        plt.plot(edges[:-1]+h/2, upper_bound, '--', color='orange', linewidth = 2)  
    
    return hatF, lower_bound, upper_bound

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


##############################################################################
############## Kernel_based_pdf_estimation imputing the missing values #######
##############################################################################


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
def f_xi(X_prior,X_i,x,h):
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


from numba import jit

##############################################################################
############## Kernel_based_pdf_estimation_with_missing_priors ###############
##############################################################################

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


##############################################################################
####### Kernel_based_pdf_estimation_with_missing_priors_limited_range#########
##############################################################################

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


##############################################################################
####### Kernel_based_pdf_estimation using the side spaces            #########
##############################################################################


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
    hat_fi_missing = [1, 1]
    hat_fi_0 = 0

    coords_missing = np.isnan(X_i)  # unknown coordinates of X_i

    # Handle the case where both are missing

    if coords_missing.sum() == k:
        hat_fi_0 = 1 
        
    else:
        for j in range(k):
            if not coords_missing[j]: # we know the j-th coordinate of X_i 
                # We can compute the contribution of the jth coordinate using the standard term
                hat_fi *= 1/bandwidth * K( (x[j]-X_i[j])/bandwidth )
                hat_fi_missing[switch(j)] = 0

            else:  # we don't know the j-th coordinate, 
                # We use prior on the missingness mechanism associated to this coordinates.
                hat_fi_missing[switch(j)] *= 1/bandwidth * K( (x[switch(j)]-X_i[switch(j)])/bandwidth ) # To check

    return hat_fi, hat_fi_0, hat_fi_missing[0], hat_fi_missing[1]

def kernel_based_pdf_estimation_xz(X, h=.2, resolution=50, cmap='Blues', verbose=0):
    
    from utils import estimate_pdf

    # Estimation of f(X_1,X_2|Z_1=1, Z_2=1), f(X_2|Z_1=0,Z_2=1) and f(X_1|Z_1=1,Z_2=0)
    hat_f, hat_f_0, hat_f_1, hat_f_2 = estimate_pdf(data=X, method='side_spaces', resolution=resolution, bandwidth=h) 
    
    # Normalization of the distributions
    hat_f /= (hat_f.sum()+EPSILON);hat_f_1 /= (hat_f_1.sum()+EPSILON);hat_f_2 /= (hat_f_2.sum()+EPSILON)

    # Computation of the marginals
    hat_f_2_marginal = hat_f.sum(axis=1);hat_f_1_marginal = hat_f.sum(axis=0)

    # Z_prior reflects P(Z_1=1, Z_2=1, ... Z_k=1)
    Z_prior = np.array([np.mean(~np.isnan(X[:,i])) for i in range(X.shape[1])])

    # Estimation of f(Z_1=0|X_2)
    hat_f_z1 = hat_f_2_marginal * Z_prior[0]/(hat_f_2_marginal * Z_prior[0]  + hat_f_2*(1-Z_prior[0]))
    hat_f_z1 /= (hat_f_z1.sum()+EPSILON)

    # Estimation of f(Z_1=1|X_2)
    hat_f_z1_bar = hat_f_2 * (1-Z_prior[0])/(hat_f_2_marginal * Z_prior[0]  + hat_f_2*(1-Z_prior[0]))
    hat_f_z1_bar /= (hat_f_z1_bar.sum()+EPSILON)

    # Estimation of f(Z_2=0|X_1)
    hat_f_z2 = hat_f_1_marginal * Z_prior[1]/(hat_f_1_marginal * Z_prior[1]  + hat_f_1*(1-Z_prior[0]))
    hat_f_z2 /= hat_f_z2.sum()

    # Estimation of f(Z_2=1|X_1)
    hat_f_z2_bar = hat_f_1 * (1-Z_prior[1])/(hat_f_1_marginal * Z_prior[1]  + hat_f_1*(1-Z_prior[0]))
    hat_f_z2_bar /= (hat_f_z2_bar.sum()+EPSILON)

    if verbose:

        fig, axes = plt.subplots(2, 5, figsize=(20, 8));axes = axes.flatten()
        axes[0].imshow(hat_f_2[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[0].set_title("A)\nf(X_2|Z_1=0)")
        axes[1].imshow(hat_f_2_marginal[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[1].set_title("B)\nf(X_2|Z_2=1)")
        axes[2].imshow(hat_f, cmap=cmap, origin='lower');axes[2].set_title("C)\nf(X_1, X_2|Z_1=1, Z_2=1)")
        axes[3].imshow(hat_f_1_marginal[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[3].set_title("D)\nf(X_1|Z_1=1)")
        axes[4].imshow(hat_f_1[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[4].set_title("E)\nf(X_1|Z_2=0)")

        axes[5].imshow(hat_f_z1_bar[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[5].set_title("f(Z_1=0|X_2)")
        axes[6].imshow(hat_f_z1[:,None].repeat(2, axis=1), cmap=cmap, origin='lower');axes[6].set_title("F)\nf(Z_1=1|X_2)")
        axes[8].imshow(hat_f_z2[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[8].set_title("G)\nf(Z_2=1|X_1)")
        axes[9].imshow(hat_f_z2_bar[None, :].repeat(2, axis=0), cmap=cmap, origin='lower');axes[9].set_title("f(Z_2=0|X_1)")
            
        _ = [ax.axis('off') for ax in axes]; plt.tight_layout()

        axes[7].text(0.5,0.5, "P(Z_1=0, Z_2=0)={:.3f}%".format(100*hat_f_0), size=18, ha="center", transform=axes[7].transAxes)

        
    return hat_f, hat_f_0, hat_f_1, hat_f_2, hat_f_1_marginal, hat_f_2_marginal, hat_f_z1, hat_f_z2, hat_f_z1_bar, hat_f_z2_bar




def fit_predict(X, y, proportion_train, resolution, bandwidth, verbose=True):
    
    #################################################################
    # (1) Separate between training and test set 
    #################################################################

    from utils import split_dataset
    X_train_pos, X_train_neg, X_test, y_true = split_dataset(X, y, proportion_train)


    #################################################################
    # (2) Estimation of the different distributions
    #################################################################

    from stats import kernel_based_pdf_estimation_xz
    hat_f_pos, hat_f_1_pos, hat_f_2_pos, hat_f_z1_knowing_x2_pos, hat_f_z2_knowing_x1_pos, hat_f_1_marginal_pos, hat_f_2_marginal_pos = kernel_based_pdf_estimation_xz(X=X_train_pos, h=bandwidth, resolution=resolution, cmap='Blues', verbose=0)
    hat_f_neg, hat_f_1_neg, hat_f_2_neg, hat_f_z1_knowing_x2_neg, hat_f_z2_knowing_x1_neg, hat_f_1_marginal_neg, hat_f_2_marginal_neg = kernel_based_pdf_estimation_xz(X=X_train_neg, h=bandwidth, resolution=resolution, cmap='Greens',verbose=0)


    #################################################################
    # (3) Prediction using maximum likelihood estimation
    #################################################################

    _, step = np.linspace(-2.5,2.5,resolution, retstep=True)

    # Contains for each sample of the Test set, the corresponding x and y index coordinates, in the matrix of the 2D pdf... 
    coord_to_index = np.floor_divide(X_test+2.5, step)


    # Init. the array of prediction
    y_pred = np.zeros(shape=y_true.shape[0]); arr = []


    #----------- Treat the case of when both coordinates are known


    # Index of the samples in the test set where both first coordinates are known 
    X_indexes_both_known = np.argwhere((~np.isnan(coord_to_index)).sum(axis=1)==2).squeeze(); arr.extend(X_indexes_both_known)

    # Coordinates of indexes in the feature space of the samples in the test set where both first coordinates are known 
    hat_f_coordinates = coord_to_index[X_indexes_both_known].astype(int)
    inds_array = np.moveaxis(np.array(list(map(tuple, hat_f_coordinates))), -1, 0)

    # Compare likelihood to do the prediction
    predictions_both_known = (hat_f_pos[tuple(inds_array)] > hat_f_neg[tuple(inds_array)]).astype(int)

    # Assign predictions 
    y_pred[X_indexes_both_known] = predictions_both_known

    #----------- Treat the case of when only the first coordinate is known

    # Index of the samples in the test set where only the first coordinate is known 
    X_indexes_first_known = np.argwhere(~np.isnan(coord_to_index[:,0]) & np.isnan(coord_to_index[:,1])).squeeze(); arr.extend(X_indexes_first_known)

    # Coordinates of index in the feature space of the samples in the test set where only the first coordinate is known 
    hat_f_coordinates = coord_to_index[X_indexes_first_known][:,0].astype(int)

    # Compare likelihood to do the prediction
    predictions_first_known = (hat_f_1_pos[hat_f_coordinates] > hat_f_1_neg[hat_f_coordinates]).astype(int)

    # Assign predictions 
    y_pred[X_indexes_first_known] = predictions_first_known


    #----------- Treat the case of when only the second coordinate is known


    # Index of the samples in the test set where only the first coordinate is known 
    X_indexes_second_known = np.argwhere(np.isnan(coord_to_index[:,0]) & ~np.isnan(coord_to_index[:,1])).squeeze(); arr.extend(X_indexes_second_known)

    # Coordinates of index in the feature space of the samples in the test set where only the first coordinate is known 
    hat_f_coordinates = coord_to_index[X_indexes_second_known][:,1].astype(int)

    # Compare likelihood to do the prediction
    predictions_second_known = (hat_f_2_pos[hat_f_coordinates] > hat_f_2_neg[hat_f_coordinates]).astype(int)

    # Assign predictions 
    y_pred[X_indexes_second_known] = predictions_second_known

    print("Sanity check: num prediction: {} == {}: Num samples\n".format(len(arr), y_true.shape[0]))



    if verbose:
        from utils import performance
        df = performance(X_test, y_true, y_pred, verbose=False)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8));axes = axes.flatten()

        axes[0].imshow(hat_f_2_pos[:,None].repeat(2, axis=1), cmap='Blues', extent=[-.5, .5, -2.5, 2.5]);axes[0].set_title("A)\nf(X_2|Z_1=0)")
        axes[1].imshow(hat_f_1_marginal_pos[:,None].repeat(2, axis=1), cmap='Blues', extent=[-.5, .5, -2.5, 2.5]);axes[1].set_title("B)\nf(X_2|Z_2=1)")
        axes[2].imshow(hat_f_pos, cmap='Blues', extent=[-2.5, 2.5, -2.5, 2.5]);axes[2].set_title("C)\nf(X_1, X_2|Z_1=1, Z_2=1)")
        axes[3].imshow(hat_f_2_marginal_pos[None, :].repeat(2, axis=0), cmap='Blues', extent=[-2.5, 2.5, -.5, .5]);axes[3].set_title("D)\nf(X_1|Z_1=1)")
        axes[4].imshow(hat_f_1_pos[None, :].repeat(2, axis=0), cmap='Blues', extent=[-2.5, 2.5, -.5, .5]);axes[4].set_title("E)\nf(X_1|Z_2=0)")
        #axes[5].imshow(hat_f_z1_knowing_x2[:,None].repeat(2, axis=1));axes[5].set_title("f(Z_1=0|X_2)")
        axes[6].imshow(hat_f_z1_knowing_x2_pos[:,None].repeat(2, axis=1), cmap='Blues', extent=[ -.5, .5,-2.5, 2.5]);axes[6].set_title("F)\nf(Z_1=1|X_2)")
        axes[8].imshow(hat_f_z2_knowing_x1_pos[None, :].repeat(2, axis=0), cmap='Blues', extent=[-2.5, 2.5, -.5, .5]);axes[8].set_title("G)\nf(Z_2=1|X_1)")
        #axes[9].imshow(hat_f_z2_knowing_x1[None, :].repeat(2, axis=0));axes[9].set_title("f(Z_2=0|X_1)")

        # plot on the A) plot the sample having only X2 
        axes[0].scatter([0]*len(df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1")), 
                        df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1")['X2'], 
                        color='b', label="TP (n={})".format(len(df.query(" `Z1`==0 and  `Z2`==1 and `True Positive`==1"))))
        axes[0].scatter([0]*len(df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")), 
                        df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")['X2'], 
                    color='g', label="FP (n={})".format(len(df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1"))))
        axes[0].scatter([0]*len(df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")),  
                        df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")['X2'], 
                        color='r', label="FN (n={})".format(len(df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1"))))


        axes[2].scatter(df.query(" `Z1`==1 and  `Z2`==1 and `True Positive`==1")['X1'], 
                        df.query(" `Z1`==1 and  `Z2`==1 and `True Positive`==1")['X2'], 
                        color='b', label="TP (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==1 and `True Positive`==1"))))
        axes[2].scatter(df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X1'], 
                        df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X2'], 
                    color='g', label="FP (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1"))))
        axes[2].scatter(df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X1'],  
                        df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X2'], 
                        color='r', label="FN (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1"))))

        axes[4].scatter(df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1")['X1'],
                        [0]*len(df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1")),  
                        color='b', label="TP (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==0 and `True Positive`==1"))))
        axes[4].scatter(df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")['X1'], 
                        [0]*len(df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")),
                        color='g', label="FP (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1"))))
        axes[4].scatter(df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")['X1'], 
                        [0]*len(df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")),  
                        color='r', label="FN (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1"))))

        _ = [ax.legend(prop={'size':10}, loc='lower right') for i,ax in enumerate(axes) if i in [0, 2, 4]]; [ax.axis('off') for ax in axes]; plt.tight_layout()


        fig, axes = plt.subplots(2, 5, figsize=(20, 8));axes = axes.flatten()

        axes[0].imshow(hat_f_2_neg[:,None].repeat(2, axis=1), cmap='Greens', extent=[-.5, .5, -2.5, 2.5]);axes[0].set_title("A)\nf(X_2|Z_1=0)")
        axes[1].imshow(hat_f_1_marginal_neg[:,None].repeat(2, axis=1), cmap='Greens', extent=[-.5, .5, -2.5, 2.5]);axes[1].set_title("B)\nf(X_2|Z_2=1)")
        axes[2].imshow(hat_f_neg, cmap='Greens', extent=[-2.5, 2.5, -2.5, 2.5]);axes[2].set_title("C)\nf(X_1, X_2|Z_1=1, Z_2=1)")
        axes[3].imshow(hat_f_2_marginal_neg[None, :].repeat(2, axis=0), cmap='Greens', extent=[-2.5, 2.5, -.5, .5]);axes[3].set_title("D)\nf(X_1|Z_1=1)")
        axes[4].imshow(hat_f_1_neg[None, :].repeat(2, axis=0), cmap='Greens', extent=[-2.5, 2.5, -.5, .5]);axes[4].set_title("E)\nf(X_1|Z_2=0)")
        #axes[5].imshow(hat_f_z1_knowing_x2[:,None].repeat(2, axis=1));axes[5].set_title("f(Z_1=0|X_2)")
        axes[6].imshow(hat_f_z1_knowing_x2_neg[:,None].repeat(2, axis=1), cmap='Greens', extent=[ -.5, .5,-2.5, 2.5]);axes[6].set_title("F)\nf(Z_1=1|X_2)")
        axes[8].imshow(hat_f_z2_knowing_x1_neg[None, :].repeat(2, axis=0), cmap='Greens', extent=[-2.5, 2.5, -.5, .5]);axes[8].set_title("G)\nf(Z_2=1|X_1)")
        #axes[9].imshow(hat_f_z2_knowing_x1[None, :].repeat(2, axis=0));axes[9].set_title("f(Z_2=0|X_1)")


        # plot on the A) plot the sample having only X2 
        axes[0].scatter([0]*len(df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1")), 
                        df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1")['X2'], 
                        color='g', label="TN (n={})".format(len(df.query(" `Z1`==0 and  `Z2`==1 and `True Negative`==1"))))
        axes[0].scatter([0]*len(df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")), 
                        df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1")['X2'], 
                    color='b', label="FN (n={})".format(len(df.query(" `Z1`==0 and  `Z2`==1 and `False Negative`==1"))))
        axes[0].scatter([0]*len(df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")),  
                        df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1")['X2'], 
                        color='r', label="FP (n={})".format(len(df.query(" `Z1`==0 and  `Z2`==1 and `False Positive`==1"))))


        axes[2].scatter(df.query(" `Z1`==1 and  `Z2`==1 and `True Negative`==1")['X1'], 
                        df.query(" `Z1`==1 and  `Z2`==1 and `True Negative`==1")['X2'], 
                        color='g', label="TN (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==1 and `True Negative`==1"))))
        axes[2].scatter(df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X1'], 
                        df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1")['X2'], 
                    color='b', label="FN (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==1 and `False Negative`==1"))))
        axes[2].scatter(df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X1'],  
                        df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1")['X2'], 
                        color='r', label="FP (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==1 and `False Positive`==1"))))

        axes[4].scatter(df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1")['X1'],
                        [0]*len(df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1")),  
                        color='g', label="TN (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==0 and `True Negative`==1"))))
        axes[4].scatter(df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")['X1'], 
                        [0]*len(df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1")),
                        color='b', label="FN (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==0 and `False Negative`==1"))))
        axes[4].scatter(df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")['X1'], 
                        [0]*len(df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1")),  
                        color='r', label="FP (n={})".format(len(df.query(" `Z1`==1 and  `Z2`==0 and `False Positive`==1"))))

        _ = [ax.legend(prop={'size':10}, loc='lower right') for i,ax in enumerate(axes) if i in [0, 2, 4]]; [ax.axis('off') for ax in axes]; plt.tight_layout()

    
    return X_test, y_true, y_pred

if __name__=='__main__':
    # Testing ...
    print('Testing stats.py ...')
    
    # Define some toy data and test the estimation of pdf kernel estimation for missing data
    X = np.array([[1.,1.],[-1,1],[np.nan,-1],[.5,np.nan]])  # the collected data
    x = [0,0.5]  # where the pdf is estimated
    h = .2 # kernel bandwidth
    
    hat_f = kernel_based_pdf_estimation(X,x=x,h=h)