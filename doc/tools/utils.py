
import numpy as np 
import matplotlib.pyplot as plt


def create_dataset(name, num_samples=10, ratio_of_missing_values=.5):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    n = num_samples + 2000  # we generate num_samples for testing and 2k as ground truth 
    if name=='moons':
        data = datasets.make_moons(n_samples=n, noise=.05)
    if name=='circles':
        data = datasets.make_circles(n_samples=n, factor=.5, noise=.05)
    if name=='blobs':
        data = datasets.make_blobs(n_samples=n, random_state=8)
    
    X_all, _ = data  # keep the 2D samples ignore the labels
    # normalize dataset for easier parameter selection
    X_all = StandardScaler().fit_transform(X_all)
    
    X = X_all[:num_samples,:] 
    Xgt = X_all[num_samples:,:]
    
    # Simulate missing samples
    for i in range(X.shape[0]):  # randomtly remove features
        if np.random.random() < ratio_of_missing_values:
            if np.random.random() < .5:  # remove samples from x or y with 
                # equal probability
                X[i,0] = np.nan
            else:
                X[i,1] = np.nan

    return X, Xgt


def estimate_pdf(data=None, method='our', resolution=20, bandwidth=None):
    xygrid = np.meshgrid(np.linspace(-2.5,2.5,resolution),np.linspace(-2.5,2.5,resolution))
    H,W = xygrid[0].shape
    hat_f = np.zeros_like(xygrid[0])  # init. the pdf estimation

    if method=='our':
        # See documentation
        from stats import kernel_based_pdf_estimation
        h = bandwidth
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(data,x=[x,y],h=h)
                
    if method=='naive':
        # Ignore missing values
        from stats import kernel_based_pdf_estimation
        h = bandwidth
        imp_data = data[~np.isnan(data[:,0]),:]
        imp_data = imp_data[~np.isnan(imp_data[:,1]),:]        
                
    if method=='mean':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_data = imp.fit_transform(data)
      
    if method=='median':
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_data = imp.fit_transform(data)
      
    if method=='knn':
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer()
        imp_data = knn_imputer.fit_transform(data)
        
    if method=='mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(n_nearest_features=None, imputation_order='ascending')
        imp_data = imp.fit_transform(data)
       
    if method in ['mice', 'knn', 'median', 'mean', 'naive']:
        from stats import kernel_based_pdf_estimation
        h = bandwidth
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation(imp_data,x=[x,y],h=h)

    return hat_f


def compare_imputation_methods(dataset='None', kernel_bandwidth=1, num_samples=100, percent_missing=70):
    
    h = kernel_bandwidth
    
    # (1) Create toy and ground truth data
    X, Xgt = create_dataset(dataset, 
                            num_samples=num_samples, 
                            ratio_of_missing_values=percent_missing/100)
    print('{} samples created'.format(X.shape[0]))
    plt.figure(figsize=[10,10]); plt.subplot(3,3,1); plt.scatter(X[:,0],X[:,1]); 
    plt.title('Toy data'); plt.xlim(-2.5, 2.5); plt.ylim(2.5, -2.5); 
    plt.xticks(()); plt.yticks(()); plt.axis('equal'); plt.axis('off')

    # Ground truth
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(Xgt)
    resolution = 20
    xygrid = np.meshgrid(np.linspace(-2.5,2.5,resolution),np.linspace(-2.5,2.5,resolution))
    H,W = xygrid[0].shape
    hat_f = np.zeros_like(xygrid[0])  # init. the pdf estimation
    for i in range(H):
        for j in range(W):
            x = xygrid[0][i,j]
            y = xygrid[1][i,j]
            hat_f[i,j] = np.exp(kde.score_samples([[x,y]]))
    plt.subplot(3,3,2); plt.imshow(hat_f); plt.axis('off'); plt.title('Ground truth')
    hat_fgt = hat_f
    hat_fgt /= hat_fgt.sum()
            
    methods = ['naive', 'mean', 'median', 'knn', 'mice', 'our']
    for i,method in enumerate(methods):
        hat_f = estimate_pdf(data=X, method=method, bandwidth=h)  
        hat_f /= hat_f.sum()
        plt.subplot(3,3,i+3); plt.imshow(hat_f); plt.axis('off');
        l2diff = np.mean( (hat_fgt-hat_f)**2 ); 
        plt.title('{} error {:2.1f}'.format(method,1e6*l2diff))
    
    return

