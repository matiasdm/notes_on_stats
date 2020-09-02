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


def feature_values_positive_to_negative_ratio(Xp=None, Xn=None, verbose=0, 
                                              x_range=None, num_bins=50):
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
        
        plt.figure(figsize=[20,5]); 
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