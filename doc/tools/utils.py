
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


def fi(x=12, y=12):
    return plt.figure(figsize=(x, y))



def create_dataset(name, num_samples=10, ratio_of_missing_values=.5, imbalance_ratio=.5, provide_labels=True, only_one_missing=True, verbose=True):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle 
    
    n = num_samples + 2000  # we generate num_samples for testing and 2k as ground truth 

    ################################
    # Generate the positive examples
    ################################
    if name=='moons':
        data = datasets.make_moons(n_samples=int(2*imbalance_ratio*n), noise=.05)
    elif name=='circles':
        data = datasets.make_circles(n_samples=int(2*imbalance_ratio*n), factor=.5, noise=.05)
    else:
        raise ValueError("Please use 'moons' or 'circles' datasets.") 
        
    X_all, labels = data  # keep the 2D samples 

    # normalize dataset for easier parameter selection
    X_all = StandardScaler().fit_transform(X_all)

    # Select the positive examples
    X_all = X_all[np.argwhere(labels==1).squeeze()]

    # Separate ground truth and training data
    X_pos = X_all[:int(num_samples*imbalance_ratio),:] 
    Xgt_pos = X_all[int(num_samples*imbalance_ratio):,:]
    labels_pos, labelsgt_pos = 1*np.ones((X_pos.shape[0], 1)), 1*np.ones((Xgt_pos.shape[0], 1))

    ################################
    # Generate the negative examples
    ################################
    if name=='moons':
        data = datasets.make_moons(n_samples=int(2*(1-imbalance_ratio)*n), noise=.05)
    elif name=='circles':
        data = datasets.make_circles(n_samples=int(2*(1-imbalance_ratio)*n), factor=.5, noise=.05)
    else:
        raise ValueError("Please use 'moons' or 'circles' datasets.") 
    
    X_all, labels = data  # keep the 2D samples 

    # normalize dataset for easier parameter selection
    X_all = StandardScaler().fit_transform(X_all)

    # Select the negative examples
    X_all = X_all[np.argwhere(labels==0).squeeze()]

    # Separate ground truth and training data
    X_neg = X_all[:int(num_samples*(1-imbalance_ratio)),:] 
    Xgt_neg = X_all[int(num_samples*(1-imbalance_ratio)):,:]
    labels_neg, labelsgt_neg = np.zeros((X_neg.shape[0], 1)), np.zeros((Xgt_neg.shape[0], 1))

    # Combine the positive and negative samples
    X, labels = np.concatenate([X_neg, X_pos], axis=0), np.concatenate([labels_neg, labels_pos], axis=0)
    Xgt, labelsgt = np.concatenate([Xgt_neg, Xgt_pos], axis=0), np.concatenate([labelsgt_neg, labelsgt_pos], axis=0)

    # Shuffle the data 
    X, labels = shuffle(X, labels, random_state=0)
    Xgt, labelsgt = shuffle(Xgt, labelsgt, random_state=0)
    
    
    if only_one_missing:
        # Simulate missing samples
        for i in range(X.shape[0]):  # randomtly remove features
            if np.random.random() < ratio_of_missing_values:
                if np.random.random() < .5:  # remove samples from x or y with 
                    # equal probability
                    X[i,0] = np.nan
                else:
                    X[i,1] = np.nan
    else:
        # Simulate missing samples
        for i in range(X.shape[0]):  # randomtly remove features
            if np.random.random() < ratio_of_missing_values:
                X[i,0] = np.nan

            if np.random.random() < ratio_of_missing_values:
                X[i,1] = np.nan

    if verbose:
        import seaborn as sns
        color = plt.get_cmap('tab20')(np.arange(0,2)); cmap = sns.color_palette(color)
        colors, colorsgt = [cmap[0] if l==1 else cmap[1] for l in labels], [cmap[0] if l==1 else cmap[1] for l in labelsgt]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax1.scatter(Xgt[:,0], Xgt[:,1], c=colorsgt);ax1.axis('off');ax1.set_title("Ground Truth\n{}% imbalance ratio\n".format(int(imbalance_ratio*100)), weight='bold')
        ax2.scatter(X[:,0], X[:,1], c=colors);ax2.axis('off');ax2.set_title("Created samples\n{}% imbalance ratio\n{} % missing data".format( int(imbalance_ratio*100),int(ratio_of_missing_values*100)), weight='bold')

    if provide_labels: 
        return X, Xgt, labels.squeeze(), labelsgt.squeeze()
    else:
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
                
    if method=='missing':
        # See documentation
        from stats import kernel_based_pdf_estimation_z_prior
        h = bandwidth
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation_z_prior(data,x=[x,y],h=h)  

    if method=='missing_limited_range':
        # See documentation
        from stats import kernel_based_pdf_estimation_z_prior_limited_range
        h = bandwidth

        # Compute the space mask to be sure not to add contribution on expty space, based on the resolution of the space
        m = [not np.isnan(np.sum(data[i,:])) for i in range(data.shape[0])]
        X_prior = data[m,:]
        hist2d, _, _ = np.histogram2d(X_prior[:,0], X_prior[:,1], bins=[np.linspace(-2.5,2.5,resolution), np.linspace(-2.5,2.5,resolution)])
        hist2d_up = np.concatenate([np.concatenate([hist2d, np.zeros((1, W-1))], axis=0), np.zeros((H, 1))], axis=1)
        mask_space = hist2d_up>0
        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                hat_f[i,j] = kernel_based_pdf_estimation_z_prior_limited_range(data,x=[x,y], put_weight=mask_space[i,j], h=h)  

    if method=='side_spaces':
        # See documentation
        from stats import kernel_based_pdf_estimation_side_spaces
        h = bandwidth
        hat_f_0 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_1 = np.zeros_like(xygrid[0])  # init. the pdf estimation
        hat_f_2 = np.zeros_like(xygrid[0])  # init. the pdf estimation

        for i in range(H):
            for j in range(W):
                x = xygrid[0][i,j]
                y = xygrid[1][i,j]
                # Computing contribution on coordinates i, j of hat_f, and coordinate i of hat_f_1 and coordinate j of hat_f_2
                hat_f[i,j], hat_f_0[i,j], hat_f_1[i,j], hat_f_2[i,j] =  kernel_based_pdf_estimation_side_spaces(X=data, x=[x, y], h=h)
                
        # Average the contribution of all i's and j's coordinate
        hat_f_0 = np.mean(hat_f_0)

        # Average the contribution of all j's coordinate on this horizontal line
        hat_f_1 = np.mean(hat_f_1, axis=0)
        
        # Average the contribution of all i's coordinate to form the vertical line
        hat_f_2 = np.mean(hat_f_2, axis=1)   

        return hat_f, hat_f_0, hat_f_1, hat_f_2    
                
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


def compare_imputation_methods(dataset='None', kernel_bandwidth=.2, num_samples=100, ratio_of_missing_values=.7, imbalance_ratio=.5, resolution=20, methods=None):
    
    h = kernel_bandwidth
    
    # (1) Create toy and ground truth data
    X, Xgt, _, _ = create_dataset(name=dataset, 
                                      num_samples=num_samples, 
                                      ratio_of_missing_values=ratio_of_missing_values, 
                                      imbalance_ratio=imbalance_ratio,
                                      provide_labels=True, 
                                      verbose=True)
    
    print('{} samples created'.format(X.shape[0]))
    plt.figure(figsize=[10,10]); plt.subplot(3,3,1); plt.scatter(X[:,0],X[:,1]); 
    plt.title('Toy data'); plt.xlim(-2.5, 2.5); plt.ylim(2.5, -2.5); 
    plt.xticks(()); plt.yticks(()); plt.axis('equal'); plt.axis('off')

    # Ground truth
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(Xgt)
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
            
    if methods is None:
        methods = ['naive', 'mean', 'median', 'knn', 'mice', 'our', 'missing']
    for i,method in enumerate(methods):
        hat_f = estimate_pdf(data=X, method=method, resolution=resolution, bandwidth=h)  
        hat_f /= hat_f.sum()
        plt.subplot(3,3,i+3); plt.imshow(hat_f); plt.axis('off');
        l2diff = np.mean( (hat_fgt-hat_f)**2 ); 
        plt.title('{} error {:2.5f}'.format(method,1e6*l2diff))
    
    return


def split_dataset(X, y, proportion_train):

    # TODO rename y_true et y_ytest
    X_train, X_test = X[:int(proportion_train*X.shape[0])], X[int(proportion_train*X.shape[0]):]
    y_train, y_test = y[:int(proportion_train*X.shape[0])], y[int(proportion_train*X.shape[0]):]
    X_train_pos, X_train_neg = X_train[(y_train==1).squeeze()], X_train[(y_train==0).squeeze()]
    
    return X_train_pos, X_train_neg, X_test, y_test.squeeze()

def my_classification_report(y_true, y_pred, ax=None, verbose=False):
    """
    Print several performance metrics that are common in the context of screening and fraud detection.
    """    

    """
    First compute the TP, FP, TN and FN from which most metrics derive
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    """
    Compute metrics of interest  
    """    

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = (tp + tn) / (tp + tn + fp +  fn)
    f1 = 2*tp / (2*tp + fp + fn)
    mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    tpr =  tp / (tp+fn)
    tnr = tn / (tn+fp)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    fnr = fn / (tp+fn)


    performances_metrics = {'Accuracy' : round(acc, 3),
                       'F1 score (2 PPVxTPR/(PPV+TPR))': round(f1, 3),
                       'Matthews correlation coefficient (MCC)': round(mcc, 3),
                       'Sensitivity, recall, hit rate, or true positive rate (TPR)': round(tpr, 3),
                       'Specificity, selectivity or true negative rate (TNR)': round(tnr, 3),
                       'Precision or positive predictive value (PPV)': round(ppv, 3),
                       'Negative predictive value (NPV)': round(npv, 3),
                       'Miss rate or false negative rate (FNR)': round(fnr, 3),
                       'False discovery rate (FDR=1-PPV)': round(1-ppv, 3),
                       'False omission rate (FOR=1-NPV)': round(1-npv, 3)}

    if verbose:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax if ax is not None else None)
        disp.im_.colorbar.remove()        
        print('Sample: {} positive and {} negative samples (#p/#n={:3.0f}%)'.format(tp+fn, tn+fp, 100*(tp+fn)/(tn+fp)))
        for item, value in performances_metrics.items():
            print("  {0:70}\t {1}".format(item, value))
        
    return ax, pd.DataFrame(performances_metrics, index=['0'])

def performance(X_test, y_true, y_pred, verbose=True):
    
    # Creation of a df for the results
    predictions_df = pd.DataFrame({'X1':X_test[:,0], 
                      'X2':X_test[:,1], 
                      'Z1':[1 if not np.isnan(x) else 0 for x in X_test[:,0]],
                      'Z2': [1 if not np.isnan(x) else 0 for x in X_test[:,1]],
                      'Y': y_true, 
                      'Prediction': y_pred, 
                      'True Positive': [1 if y_true==1 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      'True Negative': [1 if y_true==0 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      'False Positive': [1 if y_true==0 and y_pred==1 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      'False Negative': [1 if y_true==1 and y_pred==0 else 0 for (y_true, y_pred) in zip(y_true, y_pred)], 
                      })
    
    from utils import my_classification_report
    _, performances_df = my_classification_report(y_true, y_pred, verbose=verbose)

    return predictions_df, performances_df
