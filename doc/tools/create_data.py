"""
Some tools for data creation
----
matias.di.martino.uy@gmail.com,    Durham 2020. 
"""



import numpy as np

def create_headturn_toy_example(num_points=1e3, prop_positive=.05):
    """
    Define some toy data to illustrate some examples. 
    
    Keyword Arguments:
        num_points {int} -- number of points (default: {1e3})
        prop_positive {float} -- proportion of positive samples 
    
    Returns:
        X, Y: Data and labels
    """

    nump = int(num_points*prop_positive)  # Number of positive samples
    numn = num_points - nump  # Number of negative samples

    X = []
    Y = []

    # Define the negative class.
    nn1 = int(.7*numn)
    nn2 = int(numn - nn1)
    Xn1 = [min(max(.5*np.random.randn() + 1,0),5) for i in range(nn1)]
    Xn2 = [5*np.random.rand() for i in range(nn2)]
    for x in Xn1 + Xn2:
        X.append(x)
        Y.append(0)

    # Define the positive class.
    np1 = int(.7*nump)
    np2 = int(nump - np1)
    Xp1 = [min(1*max(np.random.randn() + 3,0),5) for i in range(np1)]
    Xp2 = [1.5+3*np.random.rand() for i in range(np2)]
    for x in Xp1 + Xp2:
        X.append(x)
        Y.append(1)
    
    X = np.array(X)
    Y = np.array(Y)
    # Sanity check       
    # df = pd.DataFrame({'X':X,'Y':Y})
    # sns.distplot(df.query('Y==0')['X'],bins=100, kde=True)
    # sns.distplot(df.query('Y==1')['X'],bins=100, kde=True)
    # plt.xlim([0,5])
    
    
    return X, Y