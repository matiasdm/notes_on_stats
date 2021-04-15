(sec:pdf_estimation)=
# Estimation of PDF and CDF distributions  

Continue from here updating this notebook. 


# import pre-installed packages and init. 
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys 
from myst_nb import glue

# add tools path and import our own tools
sys.path.insert(0, '../tools')
%load_ext autoreload
%autoreload 2
mpl.rcParams.update({'font.size': 14})

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

We will consider the illustrative example introduced in Section {ref}`sec:introduction_to_bayesian_inference`, please check that section for details. In short, the illustrative example consist on measuring a quantity X and predicting a label Y; in this toy example, X represent a feature we measure, and Y a binary variable associated to the diagnosis of ASD. 

# Let us create some toy data:
from create_data import create_headturn_toy_example
X_u, Y_u = create_headturn_toy_example(num_points=1e5, prop_positive=0.05)
X, Y = create_headturn_toy_example(num_points=1e3, prop_positive=0.2)

In Section {ref}`sec:introduction_to_bayesian_inference`, we discussed how to use the marginal probabilities $P(X|Y)$ to assess the diagnosis power of a feature "X" and, in particular, how to correctly measure performance in the context of imbalanced classes. During that discussion, we assumed $P(X|Y)$ was known, or more precisely, we assumed that a histogram of empirical observations was a _good approximation_ of the probability density function (PDF). This might or not be true; we need to analyze when a histogram is a good approximation of a PDF, and provide a formal numerical assessment for "_good approximation_." 

In the present section, we address most of the problems stated above. We provide quantitative indications of when a histogram is a good approximation of a PDF, and we show how to compute the optimal number of bins (or histogram resolution). We show how to calculate confidence intervals associated with empirical histograms. Also, we discuss the "Bootstrap" method, which is arguably one of the most robust and versatile techniques to estimate confidence intervals. Finally, we discuss the problem of estimating the parameters of a distribution.

(sec:cdf_definition)=
## CDF Definition

Let $X_1,...,X_n \sim F$ be an IID sample with distribution $F$. IID states for "independent identically distributed," i.e., all the samples are independent realizations of the same phenomenon. For example, consider the illustrative case introduced in Section {ref}`sec:introduction_to_bayesian_inference`. We measure for $n$ children in the ASD group their head-turn delay after a name call. All these events will be independent of each other, and all can be seen as a realization of a distribution $F$. The CDF $F(x)$ is formally defined as $P(X\leq x)$, in this example, the probability that child responds with a delay less than or equal to $x$. $F$ is unknown in practice, but in this toy example, we are simulating it. Since we have two different groups, we have two distributions $F_{ASD}$ and $F_{non-ASD}$. In this section, we are not focusing on distinguishing them but rather on how well we can approximate each. Hence, for the rest of the section, let us focus on a single one, for example, $F\stackrel{def}{=}F_{ASD}$. 

The **empirical distribution function** $\hat{F}$ is defined as:

$$
\hat{F}(x) = \frac{1}{n} \sum_{i=1}^{n} I(X_i \leq x)
$$ (eq:CDF_definition)

where $I(X_i \leq x) = 1$ if $X_i\leq x$, $0$ otherwise.

For example, {numref}`fig:CDF_empirical_approximation` shows for our toy example, the ground truth, and the empirical CDF estimated for 10, 20, and 200 observed samples.

# Let us create some toy data:
from create_data import create_headturn_toy_example
X_u, Y_u = create_headturn_toy_example(num_points=1e5, prop_positive=0.05)
X, Y = create_headturn_toy_example(num_points=1e3, prop_positive=0.2)

from stats import feature_values_positive_to_negative_ratio
Xp = X[Y==1]; Xn = X[Y==0]
df = pd.DataFrame({'X':X_u,'Y':Y_u}); 


fig = plt.figure(figsize=[15,5])
# In addition plot the empirical distribution for different number of datapoints. 
num_bins = 200; xmin = 0; xmax=5; n = len(Xp)
h = (xmax-xmin)/ num_bins;  # bin size

ns = [10,50,200]
for (i,n) in enumerate(ns):
    plt.subplot(1,3,i+1)
    # Plot the ground truth comulative distribution (formally is an approximation with a lot of empirical points)
    sns.kdeplot(df.query('Y==1')['X'], cumulative=True, linewidth=2, color='orange');

    x = Xp[:n]  # Sample n points
    # Count empirical points per-bin 
    count, edges = np.histogram(x, range=[xmin, xmax], bins=num_bins)  
    # Convert count to an estimation of the prob(x in B)
    prob_bin = count/n
    # Convert from prob in interval to pdf (int(pdf)_B = proba_B)
    pdf = prob_bin/h  
    # Compute the comulative probability
    cdf = prob_bin.cumsum()
    plt.bar(edges[:-1]+h/2, cdf, width=h, color='orange', alpha=.2)
    plt.xlim([0,5]); plt.xlabel('Head-turn delay :: t'); plt.ylabel('$P(X<t)$')
    plt.legend([]); plt.title('number of points: {}'.format(n));
    
glue("CDF_empirical_approximation", fig, display=False)

```{glue:figure} CDF_empirical_approximation
:figwidth: 80%
:name: "fig:CDF_empirical_approximation"

Empirical approximation of the CDF, as we increase the number of samples. 
```

(sec:pdf_definition)=
## PDF definition
Similar to the CDF, the probability density distribution (PDF) is defined as:

$$
f(x) \stackrel{def}{=} \lim_{\delta\rightarrow 0} \frac{P(x\leq X < x+\delta)}{\delta} = \lim_{\delta\rightarrow 0} \frac{F(x+\delta)-F(x)}{\delta} = F'(x).
$$ (eq:pdf_definition)

We will skip all the mathematical technicalities and assume all the limits introduced above exist (i.e., the CDFs considered are differentiable everywhere). We will also consider in all the practical applications we are working with, the CDFs are "smooth." While $F(x)$ has a very intuitive meaning: "the probability that $X$ is lower than $x$," the interpretation of $f(x)$ is more difficult. $f(x)$ can be understood as the derivative of $F$, but was no probabilist meaning unless we integrate it in a interval, i.e., $P(x\in A) = \int_A p(x)\,dx$. 

```{admonition} Working with the PDF requiers additional hypothesis about regularity. 

As we will discuss, **estimating the error and confidence intervals associated with the approximation of $F$ is substantially simpler than $f$.** For example, look at the approximations shown in {numref}`fig:CDF_empirical_approximation`, $\hat{F}$ looks quite similar to $F$ (when we have a decent amount of data). However, $\hat{F} '(x)$ can be very different from $F'(x)$, and this is why working with $f$ is more challenging. As we will see, to approximate $f$, we will have to add the hypothesis that "F" is smooth. 

```

(sec:CDF_confidence_intervals)=
## CDF confidence intervals

As we discussed in Section {ref}`sec:cdf_definition`, estimating the CDF is quite straight forward. The **empirical distribution function** $\hat{F}$ is defined as:

$$
\hat{F}(x) = \frac{1}{n} \sum_{i=1}^{n} I(X_i \leq x)
$$ (eq:empirical_cdf)

where $I(X_i \leq x) = 1$ if $X_i\leq x$, $0$ otherwise.

It can be proved (Theorem 7.3 {cite}`wasserman2013all`) that at any fixed value of $x$,
- $\mathbf{E}(\hat{F}(x)) = F(x)$,
- $\displaystyle\mathbf{V}(\hat{F}(x)) = \frac{F(x)(1-F(x))}{n}$.  

These properties are general (no assumptions are made about $F$). Despite showing that the estimation makes sense (i.e., it will give as the right value as the number of samples increases), the previous expressions are of little practical advantage. The variance of $\hat{F}$, $V(\hat{F})$, could be used to compute confidence intervals; however, notice that to calculate it, we need $F$ (which is what we were trying to estimate in the first place). 

In order to compute confidence intervals associated to $\hat{F}$, we can exploit Dvoretzky-Kiefer-Wolfowitz inequality. Let $X_1, ..., X_n \sim F$, then for any $\epsilon>0$, 

$$
P\left(\sup_{x}|F(x)-\hat{F}(x)|>\epsilon\right)\leq 2e^{-2n\epsilon^2}.
$$ (eq:DKW_inequality)

From DKW inequality, we can construct a confidence interval for $\hat{F}$ as follows.

**Empirical CDF estimation confidence interval:**
Define $L(x) = \max\{\hat{F}-\epsilon,0\}$ and $U(x) = \min\{\hat{F}+\epsilon, 1\}$, where 

$$
\epsilon = \sqrt{\frac{1}{2n}\log\left(\frac{2}{\alpha}\right)}.
$$ (eq:conf_cdf)

It follows from DKW that (for any $F$), 

$$
P\left(L(x) \leq F(x) \leq U(x) \ \ \mbox{for all } x \right) \geq 1 - \alpha.
$$ (eq:conf_interval_cdf)


# This are the importan pieces of code:  (check "estimate_cdf" in stats.py)
alpha = 0.05  # Set for example, a 5% confidence interval
cdf_epsilon = lambda n, alpha: np.sqrt( (1/(2*n)) * np.log(2/alpha) )
U = lambda hatF, epsilon: min(hatF + epsilon, 1)
L = lambda hatF, epsilon: max(hatF - epsilon, 0)

from stats import estimate_cdf
fig = plt.figure(figsize=[15,5])
# In addition plot the empirical distribution for different number of datapoints. 
num_bins = 200; xmin = 0; xmax=5; n = len(Xp)

ns = [10,50,200]
for (i,n) in enumerate(ns):
    plt.subplot(1,3,i+1)
    # Plot the ground truth comulative distribution (formally is an approximation with a lot of empirical points)
    sns.kdeplot(df.query('Y==1')['X'], cumulative=True, linewidth=2, color='orange');

    x = Xp[:n]  # Sample n points
    hatF, l, u = estimate_cdf(x, num_bins=num_bins, alpha=0.05, xmin=xmin, xmax=xmax, verbose=1)
    plt.xlim([0,5]); plt.xlabel('Head-turn delay :: t'); plt.ylabel('$P(X<t)$')
    plt.legend([]); plt.title('number of points: {}'.format(n));

glue("CDF_empirical_approximation_with_conf", fig, display=False)

```{glue:figure} CDF_empirical_approximation_with_conf
:figwidth: 80%
:name: "fig:CDF_empirical_approximation_with_conf"

Empirical approximation of the CDF and their associated 95% conf. inteval.  
```

{numref}`fig:CDF_empirical_approximation_with_conf` shows the estimation $\hat{F}$ of the distribution $F$, for datasets of different size. In addition the 95% confident interval associated to each of these estimations is illustrated. 

(sec: pdf_confidence_intervals)=
## PDF confidence intervals

We discussed that estimations of $F$ could be obtained without any explicit assumptions about $F$. On the other hand, estimating the density probability distribution $f$ requires some hypothesis about the regularity of the PDF. 

The simplest density estimator is a histogram. As before, let $X_1, ..., X_n$ be IID with density $f$ ($f=F'$), let assume the $X_i$ is restricted to the interval $[x_{min}, x_{max}]$, and let partition this interval into $m$ bins. The bin size $h$ can be computed as $h= (x_{max}-x_{min})/m$, the more bins (larger $m$) the smaller the bins size and vice-versa. 

{numref}`fig:pdf_examples` illustrates the histograms computed from $50$ empirical ASD samples of our toy head-turn delay example. We show the resulting histograms for three numbers of bins. Also (solid line), the ground truth PDF is plotted. 

# Impact of the bin size on the estimation error. 
fig = plt.figure(figsize=[15,5])
# In addition plot the empirical distribution for different number of datapoints. 
xmin = 0; xmax=5; n = 200
x = Xp[:n]  # Sample n points

num_bins_set = [2,9,90]
for (i,num_bins) in enumerate(num_bins_set):
    h = (xmax-xmin)/ num_bins;  # bin size

    plt.subplot(1,3,i+1)
    # Plot the ground truth comulative distribution (formally is an approximation with a lot of empirical points)
    sns.kdeplot(df.query('Y==1')['X'], cumulative=False, linewidth=2, color='orange');

    # Count empirical points per-bin 
    count, edges = np.histogram(x, range=[xmin, xmax], bins=num_bins)  
    # Convert count to an estimation of the prob(x in B)
    prob_bin = count/n
    # Convert from prob in interval to pdf (int(pdf)_B = proba_B)
    pdf = prob_bin/h  
    
    plt.bar(edges[:-1]+h/2, pdf, width=h, color='orange', alpha=.2)    
    if i == 0:
        plt.xlabel('Head-turn delay :: t'); plt.ylabel('$\hat{f}(x)$');
    plt.legend([]); plt.title('Number of bins: {}'.format(num_bins)); plt.xlim([0,5]);

glue("pdf_examples", fig, display=False)

```{glue:figure} pdf_examples
:figwidth: 80%
:name: "fig:pdf_examples"

Approximating the pdf with a discrete histogram. We illustrate the impact of the number of bins on the pdf estimation. 
```

To estimate $\hat{f}$, we first define the set of bins $\mathcal{B} = \{B_1, ..., B_m\}$. In our toy example, the bins are simply the intervals $B_i = (x_{min} + h(i-1), x_{min} + hi]$. The probability of $X$ to be in $B_i$ can be estimated as 

$$
\hat{p}_i = \frac{1}{n} \sum_{i=1}^n I(X_i \in B_i),
$$ 

and the pdf estimator $\hat{f}$ by

$$
\hat{f}(x) = \left\{\begin{array}{l}
\hat{p}_1/h  \ \ x\in B_1 \\ 
\hat{p}_2/h  \ \ x\in B_2 \\ 
... \\
\hat{p}_m/h  \ \ x\in B_m \\ 
\end{array}\right. .
$$ (eq:pdf_discrete_approximation)

Before discussing confidence intervals associated with these estimations, let us focus on the estimation error. As we can see, e.g., comparing the left and right examples in {numref}`fig:pdf_examples`, estimating $\hat{f}$ requires setting the bin size which has a great impact. A small number of bins (first example in {numref}`fig:pdf_examples`) would fail to accurately approximate $f$ do to the lack of spatial resolution. On the other extreme, (third example in {numref}`fig:pdf_examples`), too many bins would produce quantities prone to error (since only a few points lie in each segment). 

The problem discussed above can be formalized as follows. The integrated squared error (ISE) for the estimation $\hat{f}$ can be defined as:

$$
L\left(f, \hat{f}\right) = \int \left( f(x)-\hat{f}(x) \right)^2 dx. 
$$ (eq:ISE_pdf)

The risk or mean integrated squared error (MISE) is, 

$$
R\left(f, \hat{f}\right) = \mathbf{E}\left(L\left(f, \hat{f}\right)\right).
$$ (eq:MISE_pdf)

**Lemma 2.1 ([1] pag. 304)** The risk can be written as

$$
R\left(f, \hat{f}\right) = \int{b^2(x) dx} + \int{v(x) dx}
$$ (eq:risk_tradeoff)

where $b(x) = \mathbf{E}\left(\hat{f}(x)-f(x)\right)$ is the bias of $\hat{f}$ (at $x$), and $v(x) = \mathbf{V}\left(\hat{f}(x)\right)$ is the variance of the estimated value at $x$, $\hat{f}(x)$. 

```{admonition} Risk trade-off
:class: tip
**RISK = BIAS$^2$ + VARIANCE**
```

If the bins width is too large, a large bias cause large errors, on the other hand, if the bins are too narrow, the variance of the estimator will be large leading also to a poor approximation. **The optimal bin width is set by balancing these two factors such that the risk is minimized.** 

### Setting the optimal number of histograms

Setting the optimal number of bins $m^*$ by minimizing the risk function defined above is impractical, since it computation requires the knowledge of the ground truth density $f$ (a chicken and egg problem). Instead, we can solve an approximation of the error loss function as defined below.  

$$
L(f,\hat{f}) = \int\left(\hat{f}(x)-f(x)\right)^2 dx = \int \hat{f}^2(x) dx - 2\int \hat{f}(x)f(x) dx + \int f^2(x) dx, 
$$ (eq:loss_approximation)

the last term doesn't depend on $m$ so minimizing the risk is equivalent to minimizing the expected value of 

$$
J(f, \hat{f}) = \int \hat{f}^2(x) dx - 2\int \hat{f}(x)f(x) dx.
$$

We shall refer now to $\mathbf{E}(J)$ as the risk (thought formally it differs from it by a constant factor). The cross-validation estimator for the risk is 

$$
\hat{J}(\hat{f}) = \int \left( \hat{f}(x) \right)^2 dx - \frac 2 n \sum_{i=1}^{n} \hat{f}_{(-i)}(X_i)
$$

where $\hat{f}_{(-i)}$ is the histogram estimator obtained after removing the $i^{th}$ observation. 

```{admonition} Theorem 2.1 

The following identity holds ({cite}`wasserman2013all` pag. 310):

$$
\hat{J}(\hat{f}) = \frac{2}{(n-1)h} - \frac{n+1}{(n-1)h}\sum_{i=1}^{m} \hat{p}_i^2. 
$$ (eq:risk_estimation)

```

Equation {eq}`eq:risk_estimation` provides a practical alternative to compute the error associated with a bin size, and can be used to tune it as we illustrate below ({numref}`fig:pdf_empirical_risk_estimation`). 


```{caution}
In {cite}`wasserman2013all` the definition is incorrect in Eq. (20.14). $h$ is missing in the second term, see the original reference (Mats Rudemo 1981, Empirical Choice of Histograms and Kernel Denisty Estimators; Eq. (2.8)).
```

# To optimize the number of bins, we can use this expresion: (check bin_size_risk_estimator for more details).
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
    return J

fig = plt.figure(figsize=[10,5])

xmin = 0; xmax=5; n = 200
x = Xp[:n]  # Sample n points

from stats import bin_size_risk_estimator
plt.subplot(1,2,1)
hat_J, opt_num_bins = bin_size_risk_estimator(x, num_bins_range=[1,50], xmin=xmin, xmax=xmax, verbose=1)

plt.subplot(1,2,2)
sns.kdeplot(df.query('Y==1')['X'], cumulative=False, linewidth=2, color='orange');  # plot grount truth
num_bins = opt_num_bins; h = (xmax-xmin)/ num_bins;  # bin size
count, edges = np.histogram(x, range=[xmin, xmax], bins=num_bins)  
prob_bin = count/n; pdf = prob_bin/h; plt.bar(edges[:-1]+h/2, pdf, width=h, color='orange', alpha=.2)    
plt.xlim([0,5]); plt.xlabel('Head-turn delay :: t'); 
plt.legend([]); plt.title('hat(f) and f,  number of bins: {}'.format(num_bins));
glue("pdf_empirical_risk_estimation", fig, display=False)

```{glue:figure} pdf_empirical_risk_estimation
:figwidth: 80%
:name: "fig:pdf_empirical_risk_estimation"

The left side illustrates the ground truth risk (dashed) and the empirical risk estimation (solid) as we change the number of bins (i.e., the bin width). The right plot illustrates the ground truth pdf distribution and the histogram estimation for the optimal number of bins. 
```

### Confidence intervals associated to a pdf estimation

Once a histogram estimation of a pdf is obtained (after setting the number of bins and the bin size), we can estimate confidence intervals associated to it. Suppose $\hat{f}$ is a histogram with $m$ bins and binwidth $h=1/m$, approximating the ground truth density $f$. We define $\bar{f}=p_i/h$ where $p_i = \int_{B_i} f(x)dx$. As before, $B_i$ represents the interval associated to the $i^{th}$ bin. $\bar{f}$ represents the ground truth "histogram version" of the pdf $f$. 

```{admonition} Definition
A pair of functions $(l(x), u(x))$ is a $1-\alpha$ confidence band if, 

$$
\mathbf{P}\left( l(x) \leq \bar{f}(x) \leq u(x) \mbox{ for all } x \right) \geq 1-\alpha
$$ (eq:def_confidence_band)

```

```{admonition} Theorem

Theorem 2.2 {cite}`wasserman2013all` Pag. 311: Let $m(n)$ be the number of bins in the histogram $\hat{f}$. Assume $m(n)\rightarrow 0$ and $m(n)\frac{\log(n)}{n}\rightarrow 0$ as $n\rightarrow \infty$. Define 

$$
l(x) = \left(\max\left\{ \sqrt{\hat{f}(x)} -c, 0\right\} \right)^2
$$

$$
u(x) = \left(\sqrt{\hat{f}(x)} + c \right)^2
$$

where 

$$
c = \frac{z_{\alpha/(2m)}}{2}\sqrt{\frac{m}{n}}.
$$ (eq:c_pdf_interval)

Then, $(l(x),u(x))$ is an approximate $1-\alpha$ confidence band. 
```

# (see compute_histogram_and_conf_interval)
def compute_hist_conf_constant(alpha, m, n):   
    def z_u(u):  # Compute the upper u quantile of N(0,1)
        from scipy.stats import norm
        z = norm.ppf(1-u)  # z_u = Phi^-1(1-u)  with Phi = cdf_{N(0,1)}
        return z
    c = z_u(alpha/(2*m))/2 * np.sqrt(m/n)
    return c

lf = lambda hat_f,c: (max( np.sqrt(hat_f)-c, 0 ))**2
uf = lambda hat_f,c: (np.sqrt(hat_f)+c)**2

# Impact of the bin size on the estimation error. 
from stats import compute_histogram_and_conf_interval
fig = plt.figure(figsize=[15,5])
# In addition plot the empirical distribution for different number of datapoints. 
xmin = 0; xmax=5; num_bins = 10; alpha = 0.05 # Conf interval
 
ns = [20, 100, 200]
for (i,n) in enumerate(ns):
    x = Xp[:n]  # Sample n points
    
    plt.subplot(1,3,i+1)
    sns.kdeplot(df.query('Y==1')['X'], cumulative=False, linewidth=2, color='orange');

    # Count empirical points per-bin 
    hat_f, l, u = compute_histogram_and_conf_interval(x, xmin=xmin, xmax=xmax, alpha=alpha, 
                                                      num_bins=num_bins, verbose=1)
    if i == 0:
        plt.xlabel('Head-turn delay :: t'); plt.ylabel('$\hat{f}(x)$');
    else:
        plt.xlabel(' ')
    plt.legend([]); plt.title('Number empirical samples: {}'.format(n)); plt.xlim([0,5]);
glue("pdf_histogram_confidence_intervals", fig, display=False)

```{glue:figure} pdf_histogram_confidence_intervals
:figwidth: 80%
:name: "fig:pdf_histogram_confidence_intervals"

Ground truth distribution (solid), and the estimated histograms with their associated 5-95% confidence intervals for different number of samples. 
```

{numref}`fig:pdf_histogram_confidence_intervals` shows the ground truth $f$, and the estimated $\hat{f}$ from 20, 100, and 200 empirical samples. (In all the cases we We used the optimal bin size estimated above.) The $5\%-95\%$ confidence interval associated with each estimation is displayed. 

As we can see, the intervals get narrower as the number of samples increases. Even for 200 samples, the uncertainty bounds estimated is quite large; tighter estimations can be obtained using kernel based methods which we will address next. 

### Kernel-based density estimation 

Kernel density estimators are smoother and converge faster to the true distributions, i.e., they are more accurate for the same amount of samples providing tighter confidence intervals. As above $X_1, ..., X_n$ are IID samples from $f$. Given a kernel $K$ and a positive number $h$ (called _the bandwidth_), the kernel density estimator is defined 

$$
\hat{f}(x) = \frac{1}{n} \sum_{i=1}^{n} \frac 1 h K\left(\frac{x-X_i}{h}\right).
$$ (eq:kernel_estimation)

To construct a kernel density estimator we need to choose a kernel $K$ and a bandwidth $h$. It can be shown that the selection of $h$ is important while the choice of $K$ isn't crucial {cite}`wasserman2013all`. Here, we will use the Gaussian kernel: 

$$
\displaystyle K(u) = \frac{1}{\sqrt{2\pi}} e^{\frac{-u^2}{2}}. 
$$ (eq:gaussian_kernel)

{numref}`fig:pdf_kernel` shows the kernel density estimator for the head-turn delay example introduced above. 

# Example of a kernel estimation implementation 
def kernel_estimator(X,x=None,h=1):
    K = lambda u: 1/np.sqrt(2*np.pi) * np.exp(-u**2 / 2)
    n = len(X)  # number of samples    
    hat_f = np.zeros_like(x)  # init.
    for j, x_j in enumerate(x):
        for X_i in X:
            hat_f[j] += 1/(n*h) * K( (x_j-X_i) / h )
    return hat_f

from stats import kernel_estimator
# Impact of the bin size on the estimation error. 
fig = plt.figure(figsize=[15,5])
# In addition plot the empirical distribution for different number of datapoints. 
xmin = 0; xmax=5; n = 200; X = Xp[:n] 
h0 = (xmax-xmin)/10  # Bandwidth
N = 30  # x resolution for the estimation of the kernel. 
x = np.linspace(xmin,xmax,N)  # Define the points in which the density is estimated. 

h_set = [h0/10,h0,2*h0]  # Define the set of bandwidths.
for (i,h) in enumerate(h_set):
    plt.subplot(1,3,i+1)
    # Plot the ground truth comulative distribution (formally is an approximation with a lot of empirical points)
    sns.kdeplot(df.query('Y==1')['X'], cumulative=False, linewidth=2, color='orange');
    # Count empirical points per-bin 
    hat_f = kernel_estimator(X,x=x,h=h)
    plt.bar(x, hat_f, width=x[1]-x[0], color='orange', alpha=.2)    
    if i == 0:
        plt.xlabel('Head-turn delay :: t'); plt.ylabel('$\hat{f}(x)$');
    plt.legend([]); plt.title('Bandwidth: {}'.format(h)); plt.xlim([0,5]);
glue("pdf_kernel", fig, display=False)

```{glue:figure} pdf_kernel
:figwidth: 80%
:name: "fig:pdf_kernel"

Kernel estimation for 200 samples, for different bandwidth sizes. 
```

As we illustrate in {numref}`fig:pdf_kernel`, the bandwidth $h$ plays a similar role to the number of bins in the histogram estimation. The smaller the bandwidth, the larger the error because of variance, the larger the bandwidth, the larger the error because of bias. As for the number of bins, we need to estimate the optimal bandwidth by minimizing the approximation error. Recall that:

> RISK = BIAS^2 + VARIANCE


The cross-validation estimation of the risk $\hat{J}(h)$ can be approximated using Theorem 2.3.

```{admonition} Theorem
Theorem 2.3 {cite}`wasserman2013all` Pag. 316: For any $h>0$, $\mathbf{E}[\hat{J}(h)] = \mathbf{E}[J(h)]$, and 

$$
\hat{J}(h) \approx \frac{1}{h n^2}\sum_{i}\sum_{j} K^*\left(\frac{X_i-X_j}{h}\right) + \frac{2}{nh}K(0)
$$

where $K^*(u) = K^{(2)}(u)-2 K(u)$ and $K^{(2)}(z) = \int K(z-y)K(y)dy$. In particular, if $K$ is $N(0,1)$, then $K^{(2)}$ is the $N(0,2)$ density. 
```

{numref}`fig:pdf_kernel_estimation_optimal_bandwidth` shows the estimated risk (left) and the estimated density for the optimal bandwidth. 

def kernel_based_empirical_risk(X, h):
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

from stats import kernel_estimator
# Impact of the bin size on the estimation error. 
fig = plt.figure(figsize=[10,5])
# In addition plot the empirical distribution for different number of datapoints. 
xmin = 0; xmax=5; n = 200; X = Xp[:n] ; N = 30  # x resolution for the estimation of the kernel. 
x = np.linspace(xmin,xmax,N)  # Define the points in which the density is estimated. 

from stats import kernel_based_empirical_risk
hs = np.linspace(0.05,1.5,20)
J = [kernel_based_empirical_risk(X,hh) for hh in hs]
plt.subplot(1,2,1); plt.plot(hs,J); plt.xlabel('h'); plt.ylabel('$\hat{J(h)}$')
h_opt = hs[np.argmin(J)]  # get opt bandwidth
plt.scatter(h_opt, min(J), 150, color='orange')
plt.subplot(1,2,2); sns.kdeplot(df.query('Y==1')['X'], cumulative=False, linewidth=2, color='orange');
hat_f = kernel_estimator(X,x=x,h=h_opt)
plt.bar(x, hat_f, width=x[1]-x[0], color='orange', alpha=.2)    
plt.xlabel('Head-turn delay :: t'); plt.xlim([0,5]);
plt.legend([]); plt.title('PDF estimation for the optimal bandwidth: {:2.1f}'.format(h_opt)); 
glue("pdf_kernel_estimation_optimal_bandwidth", fig, display=False)

```{glue:figure} pdf_kernel_estimation_optimal_bandwidth
:figwidth: 80%
:name: "fig:pdf_kernel_estimation_optimal_bandwidth"

Kernel bandwidth optimization (left), and the kernel estimation for the optimal bandwidth)
```

### Confidence associated with a kernel estimation

As for the estimated histograms, confidence intervals can be computed for the kernel density estimation. The confidence band is defined for a smoothed version $\bar{f}$ of the ground truth distribution $f$, defined as

$$
\bar{f}(x)=\int{\frac 1 h K \left( \frac{x-u}{h} \right) f(u)\, du } \ = \mathbf{E}(\hat{f}(x).
$$ (eq:bar_f_conf_interval)

When the normal kernel is used, a $1-\alpha$ confidence interval $(l(x),u(x))$ is given by

$$
l(x) = \hat{f}(x)- q\, \mbox{se}(x), u(x) = \hat{f}(x) + q\, \mbox{se}(x)
$$

where 

$$
\mbox{se}(x) = \frac{s(x)}{\sqrt{n}}, \ \ s^2(x) = \frac{1}{n-1} \sum_i (Y_i(x) - \bar{Y}(x))^2, \ \ \left(\bar{Y}(x) = \frac 1 n \sum_i Y_i(x)\right),\ Y_i(x) = \frac{1}{h} K\left(\frac{x-X_i}{h}\right),
$$

$$
q = \Phi^{-1}\left(\frac{1 + (1-\alpha)^{1/m}}{2}\right), \mbox{ and } m=3h.
$$

{numref}`fig:kernel_estimation_with_CI` illustrates the $95\%$ confidence intervals for the kernel density estimation presented in {numref}`fig:pdf_kernel_estimation_optimal_bandwidth`. Compare this result with the ones obtained for the histogram estimator {numref}`fig:pdf_histogram_confidence_intervals`. 

# Example of implementation (see stats compute_kernel_estimator_and_conf_intervals for more details).
def compute_kernel_estimation_conf(X, hat_f=None, x=None, h=None, alpha=0.05):
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

from stats import kernel_estimator
# Impact of the bin size on the estimation error. 
fig = plt.figure(figsize=[10,5])
# In addition plot the empirical distribution for different number of datapoints. 
xmin = 0; xmax=5; n = 200; X = Xp[:n] ; N = 30  # x resolution for the estimation of the kernel. 
x = np.linspace(xmin,xmax,N)  # Define the points in which the density is estimated. 
alpha=0.05  # set conf. interval

from stats import compute_kernel_estimator_and_conf_interval
h = 0.3; 
plt.subplot(1,2,1); sns.kdeplot(df.query('Y==1')['X'], cumulative=False, linewidth=2, color='orange');
hat_f, l, u = compute_kernel_estimator_and_conf_interval(X,x=x,xmin=xmin,xmax=xmax,h=h,
                                                         alpha=alpha,verbose=1)
plt.xlabel('Head-turn delay :: t'); plt.xlim([0,5]); plt.legend([]); 
plt.subplot(1,2,2);  
sns.kdeplot(df.query('Y==1')['X'], cumulative=False, linewidth=2, color='orange');
hat_f, l, u = compute_histogram_and_conf_interval(X, xmin=xmin, xmax=xmax, alpha=alpha, 
                                                  num_bins=num_bins, verbose=1)
plt.title('Histogram estimation with 95% CI'); plt.legend([]); 
glue("kernel_estimation_with_CI", fig, display=False)

```{glue:figure} kernel_estimation_with_CI
:figwidth: 80%
:name: "fig:kernel_estimation_with_CI"

Kernel PDF estimation and its 95% CI (left), and the Histogram PDF estimation with its associated CI (right). In both cases the optima bandwidth/bin size was selected. 200 empirical samples were considered for both. 
```

As discussed above, kernel estimations provide tighter approximations for the same amount of data, and have, in general, a faster convergence to the ground truth distribution. 

(sec:parameter_estimation)= 
## Parameter estimation

In the previous sections, we focused on the estimation of distributions on a model agnostic fashion. Meaning, no hypothesis about the nature of the distribution was made. In contrast, if we know the distribution model, for example, that the distribution is Gaussian, we only need to estimate two number to characterize the process: the mean and standard deviation. In other scenarios, even if we don't know the model of the distribution, we might still only care about certain properties of the distribution, such as its mean. The approximation of these quantities is described as the theory of parameter estimation. 

```{admonition} Theorem
Theorem 2.4: The Central Limit Theorem (CLT)** Let $X_1, ..., X_n$ be IID with mean $\mu$ and variance $\sigma^2$. Let $\bar{X}_n = n^{-1} \sum_{i=1}^n X_i$. Then

$$
\frac{\bar{X}_n-\mu}{\sigma / \sqrt{n}} \stackrel{n\rightarrow \infty}{\rightsquigarrow} N\left(0, 1\right).
$$
```

In Theorem 2.4 $X_n \rightsquigarrow F$ means that $X_n$ converges to $F$ _in distribution_, i.e., $\lim_{n\rightarrow \infty} F_n(x) = F(x)$ for all $x$ for which $F$ is continuous. (A formal and more exhaustive explanation is provided in {cite}`wasserman2013all` {cite}`kay1993fundamentals`). This theorem is central to the objectives of this section. First, it tells us that a simple average of the samples is a good approximation of the mean of the distribution of $X_i$. As the number of samples increases, it tells us that the error tends to 0. It also gives an idea of the error (the variance) which, as we shall see, can be used to estimate confidence intervals. The most remarkable part is that it tells us that $\bar{X_n}$ has a normal probability distribution, with no hypothesis about $F$! (the distribution of $X_i$). 

```{admonition} Theorem
Theorem 2.5 ({cite}`wasserman2013all` pag. 78) Assume the same conditions as in Theorem 2.4. And define, 

$$
S_n^2 = \frac{1}{n-1} \sum_{i=1}^{n}(X_i - \bar{X}_n)^2.
$$

Then, 

$$
\frac{\bar{X}_n-\mu}{S_n / \sqrt{n}} \stackrel{n\rightarrow \infty}{\rightsquigarrow} N\left(0, 1\right).
$$
```

Theorem 2.5 essentially tell us that we can still apply the CLT using the estimated variance $S_n$ instead of the ground truth variance $\sigma$. This is very useful since $\sigma$ is unknown in practice. Using Theorem 2.5, computing a $\alpha$ confidence interval for $\bar{X}_n$ is straightforward: $\bar{X}_n \pm z_{\alpha/2} \frac{S_n}{\sqrt{n}}$. (For $\alpha=0.05$, $z_{\alpha/2} = 1.96 \approx 2$, is common then to use the approximated rule of $\pm 2 \frac{S_n}{\sqrt{n}}$.) Below we provide computations examples for the toy data used across this section. 

# As before, for compactness we refer to the positive class the class of ASD kids, 
# and negative class the clas of non-ASD kids. 
num_p = len(Xp); num_n = len(Xn)  # number of samples in each group

bar_Xp = np.mean(Xp); bar_Xn = np.mean(Xn);  # Empirical mean of each sample

# Shortcut for the emp. estimation of the std (numpy uses 1/n instead 1/n-1)
var = lambda X: 1/(len(X)-1) * np.sum(np.array( [(X_i-np.mean(X))**2 for X_i in X] ))


Sp = np.sqrt(var(Xp)); Sn = np.sqrt(var(Xn));  # Empirical estimation of the square root of the variance S
# Compute the 5% conf. interval (we use the approx z_(5%/2) ~ 2)
conf_p = 2*Sp/np.sqrt(num_p); conf_n = 2*Sn/np.sqrt(num_n)

print('The estimated mean for the head-tourn delay (mean and 95% CI).')
print('ASD group     :: {:2.2f} +- {:2.2f} seconds'.format(bar_Xp, conf_p))
print('non-ASD group :: {:2.2f} +- {:2.2f} seconds'.format(bar_Xn, conf_n))

What if we want to model the difference between the two means? We can use that the difference between to (independent) normally distributed variables is also normal. If $X_1\,\sim\,N(\mu_1,\sigma_1^2)$, $X_2\,\sim\,N(\mu_2,\sigma_2^2)$, then $Y = X_1-X_2\, \sim\, N(\mu_1-\mu_2,\sigma_1^2+\sigma_2^2)$.

mean_diff = bar_Xp - bar_Xn
S_diff = np.sqrt( (Sp/np.sqrt(num_p))**2 + (Sn/np.sqrt(num_n))**2 )
conf_diff = 2*S_diff
print('The ASD kids respond in average {:2.2f} +- {:2.2f} seconds slower than the non-ASD kids.'.format(mean_diff, conf_diff))

### A general framework

The ideas discussed in Sections {ref}`sec:cdf_definition`, {ref}`sec:pdf_definition` and {ref}`sec:parameter_estimation` can be unified in a more general framework. Let $X_1, ..., X_n$ be IID samples from some distribution $F$. $\theta$ denotes a parameter we want to estimate, e.g., the ground truth height of the histogram for some bin $B_i$, or the mean of $F$ $\mu$. $\hat{\theta}$ denotes our estimation of $\theta$ from the observed data. $\hat{\theta}$ can be expressed as a function of the random variables $X_i$: $\hat{\theta}=g(X_1, ..., X_n)$. As this is a function of random variables, $\hat{\theta}$ is itself a random variable. 

For example, on the experiment discussed above, $\theta=\mu$, and we defined our estimator $\hat{\theta}=\bar{X_n}$. The CLT tells us that the distribution of the random variable $\hat{\theta}$ is normal (regardless of what is the distribution $F$ that generated the samples $X_i$). 

The bias of an estimator is defined by,

$$
\mbox{bias}(\hat{\theta}) = \mathbf{E}(\hat{\theta})-\theta,
$$

and the standard error $\mbox{se} = \sqrt{\mathcal{V}(\hat{\theta})}$.

The mean squared error is defined as 

$$ 
MSE = \mathbf{E}(\hat{\theta}-\theta)^2.
$$

```{admonition} Theorem
Theorem 2.6 ({cite}`wasserman2013all` pag. 91) The MSE can be written as

$$
MSE = \mbox{bias}^2(\hat{\theta}) + \mathbf{V}(\hat{\theta}).
$$

```

```{admonition} Definition
An estimator is asymptotically normal if 

$$
\frac{\hat{\theta}-\theta}{\mbox{se}} \stackrel{n\rightarrow \infty}{\rightsquigarrow} N\left(0, 1\right).
$$

```

For example, in the experiment illustrated above, we estimated the mean by the empirical mean, we had an asymptotically normal estimator with $\mbox{se}=\sigma/\sqrt{n}$. Obtaining confidence intervals for asymptotically normal estimators is straight forward, an $\alpha$ confidence interval is $\hat{\theta}\pm z_{\alpha/2}\mbox{se}$. (Remember that for $\alpha=5\%$, $z_{\alpha/2}=1.96 \approx 2$.  

### Bootstrap method

The bootstrap is a method for estimating standard errors and computing confidence intervals. Let $X_1, ..., X_n\sim F$ be a set of random variables and $\hat{\theta} = g(X_1,...,X_n)$ any function of the data. The goal is to estimate the variance $\mathbf{V}_F(\hat{\theta})$. We write $\mathbf{V}_F(\hat{\theta})$ to emphasize that the variance usually depends on the unknown distribution $F$. Bootstrap has two main step: (i) estimate $\mathbf{V}_F(\hat{\theta})$ with $\mathbf{V}_{\hat{F}}(\hat{\theta})$, and (ii) approximate $\mathbf{V}_{\hat{F}}(\hat{\theta})$.

#### Estimating the variance 

The first step consists of estimating the variance of $\hat{\theta}$ assuming $F=\hat{F}$ where $\hat{F}$ is the empirical distribution of the data. Since $\hat{F}$ puts $1/n$ probability in each sample, simulating new samples $X^*_1, ..., X^*_n\sim \hat{F}$ is equivalent of sampling with replacement $n$ samples from $\{X_1, ..., X_n\}$. The method can be summarized as follows. 
1. Draw $X_1^*, ..., X_n^*\sim \hat{F}$.
2. Compute $\hat{\theta}^* = g(X_1^*, ..., X_n^*)$.
3. Repeat steps 1 and 2, $B$ times, to get $\hat{\theta}^*_1, ..., \hat{\theta}^*_B$.
4. Let 

$$
v_{boot} = \frac 1 B \sum_{b = 1}^{B} \left( \hat{\theta}^*_b - \bar{\theta}^* \right)^2,
$$

where $\bar{\theta}^* = \frac 1 B \sum_b \hat{\theta}^*_b$.


Remember we are doing two approximations:

$$
\mathbf{V}_F(\hat{\theta}) \stackrel{\mbox{not so small}}{\approx} \mathbf{V}_{\hat{F}}(\hat{\theta}) \stackrel{\mbox{small}}{\approx} v_{boot}. 
$$

#### Bootstrap confidence intervals

There are several methods to construct bootstrap confidence intervals. We discuss two: _The Normal Interval_, and _the pivotal interval_. 

```{admonition} Method 1: The normal interval.

This method is straight forward but has a limitation: it assumes the distribution of $\hat{\theta}$ is approximately normal (if we estimate the mean $\hat{X}$, for example, this is true thanks to the CLT). Then, the $1-\alpha$ confidence interval is simply $\hat{\theta} \pm z_{\alpha / 2}\sqrt{v_{boot}}$. 

```

```{admonition} Method 2: Pivotal Interval. 

We define the error $R=\hat{\theta}-\theta$, and $H(r)$ the CDF of $R$, i.e., $H(r)=P(R\leq r)$. Of course the ground truth value of $\theta$ is unknown so we can't compute directly $R$ or $H$. Instead, we will approximate $H$ using the bootstrap estimations of $\hat{\theta}^*_b$. We define

$$
\hat{H}(r) \stackrel{def}{=} \frac{1}{B} \sum_{b=1}^{B} I(R^*_b \leq r)
$$

where $R^*_b \stackrel{def}{=} \hat{\theta}^*_b - \hat{\theta}$. It follows that an approximate $1-\alpha$ confidence interval is $(\hat{a},\hat{b})$ where, 

$$
\hat{a} = \hat{\theta} - \hat{H}^{-1}\left(1-\frac{\alpha}{2}\right) = \hat{\theta} - (\hat{\theta}^*_{1-\alpha/2} - \hat{\theta}) = 2\hat{\theta} - \hat{\theta}^*_{1-\alpha/2}.
$$

$$
\hat{b} = \hat{\theta} - \hat{H}^{-1}\left(\frac{\alpha}{2}\right) = \hat{\theta} - (\hat{\theta}^*_{\alpha/2} - \hat{\theta}) = 2\hat{\theta} - \hat{\theta}^*_{\alpha/2}.
$$

$\hat{\theta}^*_{\beta}$ is the $\beta$ quantile of the sample $\{\hat{\theta}^*_{1}, ..., \hat{\theta}^*_{B}\}$. For example, $\hat{\theta}^*_{0.05}$ is the sample $\hat{\theta}^*_{i}$ for which $95\%$ of $\hat{\theta}^*_{j}$ are larger.  

```

Below we illustrate the implementation of this methods on our toy numerical example. 

# Example: estimate the confidence interval for the mean of the head-turn delay on the ASD group 
# using bootstrap and the pivotal interval method. 
X = Xp ; n = len(X); bar_X = np.mean(X) 
# Recall our previous confidence interval estimation using the CLT. 
# Empirical variance (un-biased estimator using 1/n-1). 
var = lambda X: 1/(len(X)-1) * np.sum(np.array( [(X_i-np.mean(X))**2 for X_i in X] ))
se = np.sqrt(var(X)/n)  # Using CLT 

# Now using bootstrap
def bootstrap(X,B):
    bar_X_b = []
    for b in range(B):
        X_b = np.random.choice(X, n, replace=True)  # Get n samples from {X_1,...,X_n} with replacement. 
        bar_X_b.append(np.mean(X_b))  # Estimate the mean from the sample
    # Compute the mean of the estimated values
    v_boot = np.var(bar_X_b)
    return v_boot

fig = plt.figure()
Bs = np.linspace(1,1000,250); v_boots = [bootstrap(X,int(B)) for B in Bs];
plt.plot(Bs, v_boots, '-', color='orange'); 
plt.plot(Bs, [se**2 for B in Bs], '--', color='steelblue');
plt.legend(['bootstrap estimation', 'variance from CLT']); 
plt.xlabel('Number of bootstrap iterations'); plt.ylabel('Variance of the estimator');
glue("bootstrap_variance_estimation", fig, display=False)

```{glue:figure} bootstrap_variance_estimation
:figwidth: 80%
:name: "fig:bootstrap_variance_estimation"

Bootstrap variance estimation. 
```

# Compare 95% confidence intervals from bootstrap and from the CLT. 
alpha = 0.05
B = 200; bar_X_b = []
for b in range(B):
    X_b = np.random.choice(X, n, replace=True)  # Get n samples from {X_1,...,X_n} with replacement. 
    bar_X_b.append(np.mean(X_b))  # Estimate the mean from the sample

bar_X_u = np.quantile(bar_X_b, 1-alpha/2)
bar_X_l = np.quantile(bar_X_b, alpha/2)
hat_a = 2*bar_X - bar_X_u
hat_b = 2*bar_X - bar_X_l
print('Mean 95% conf interval from bootstrap :: ({:2.2f}, {:2.2f}) seconds'.format(hat_a, hat_b))
print('Mean 95% conf interval from CLT :: ({:2.2f}, {:2.2f}) seconds'.format(bar_X-2*se, bar_X+2*se))

(sec: pdf_estimation_with_missing_values)=
## PDF estimation with missing values

We compare different techniques to estimate the pdf using multidimensional kernel methods, in particular, in the context of samples with missing data. 

### Adapting the kernel method
As discussed before, the pdf estimation can be expressed as a liner combination of the contribution of each training samples, weighted by a kernel function. For example, we can estimate the pdf at $u$, given a set of observations $X_i$, $i=1\,...\,n$ as,

$$
\hat{f}(u) = \frac 1 n \sum_i f_{X_i}(u),
$$(eq:kernel_estimation_mdim)

we denote $f_{X_i}$ the contribution of the $i^{th}$ observation to the estimation. For a canonical kernel estimations, using a Gaussian kernel, $f_{X_i}$ is defined as, 

$$
\displaystyle f_{X_i}(u) = \frac{1}{\sqrt{(2\pi)^k|\Sigma|}}e^{-\frac 1 2 (u-X_i)^t \Sigma^{-1} (u-X_i)}
$$(eq:k-dim_kernel)
 
Equations {eq}`eq:kernel_estimation_mdim` and {eq}`eq:k-dim_kernel` generalize Eq. {eq}`eq:kernel_estimation` for the case of k-dimensional problem. We set consider a symmetric kernel (i.e. $\Sigma=h^2 Id$), with an fixed bandwidth $h$ for all axis. Under this consideration, {eq}`eq:k-dim_kernel` can be simplified as, 

$$
\displaystyle f_{X_i}(u) = \frac{1}{h^k\sqrt{(2\pi)^k}}e^{-\frac{1}{2 h^2} (u-X_i)^t(u-X_i)}.
$$(eq:k-dim_kernel2)

Using standard properties of the exponential functions, Eq. {eq}`eq:k-dim_kernel2` can be expressed as, 

$$
f_{X_i}(u) = \prod_j \frac{1}{h\sqrt{2\pi}} e^{-\frac{1}{2 h^2} (u_j-X_{ij})^2}.
$$

Notice that the first subindex (i) indicates which observation is considered, while the second index (j) refers to the coordinate number in the k-dimensional feature space. This expression is practical, since decouples the axis during the computation of the joint probability. Observe that the decomposition comes from the symmetry of the kernel and not from assuming any independence hypothesis between the features. 

Putting everything together, Eq. {eq}`eq:kernel_estimation_mdim` can be expressed as, 

$$
\hat{f}(u) = \frac 1 n \sum_i \prod_j \frac{1}{h\sqrt{2\pi}} e^{-\frac{1}{2 h^2} (u_j-X_{ij})^2}.
$$(eq:kernel_estimation_mdim2)

The computation of Eq. {eq}`eq:kernel_estimation_mdim2` requires the knowledge of all the elements for each single observation, in that sense, can not be computed for partial observations. However, the factorization into individual observations and coordinates provides a suitable representation to handle missing data as we discuss before. 

We generalize {eq}`eq:kernel_estimation_mdim2` as

$$
\hat{f}(u) = \frac 1 n \sum_i \prod_j f_{ij}(u).
$$(eq:kernel_estimation_mdim3)

$f_{ij}(u)$ represents the contribution of the $j^{th}$ coordinate of the $i^{th}$ observation at the location $u$. 

$$
f_{ij}(u) \stackrel{def}{=} \frac{1}{h\sqrt{2\pi}} e^{-\frac{1}{2 h^2} (u_j-X_{ij})^2}
$$(eq:fij_known_j)

if the $j^{th}$ coordinate of the $i^{th}$ observation is known, otherwise, 

$$
f_{ij}(u) \stackrel{def}{=} \frac{1}{n'} \frac{\sum_{i'\in I} w_{i'} f_{i'j}(u)}{\sum_{i'\in I} w_{i'}}
$$(eq:fij_unknown_j)

with $I$ the set of indices of observations for which all the coordinates are known, $w_{i'} = \frac{1}{h\sqrt{2\pi}}e^{-\frac{1}{2h} d_{X_{i}}(X_{i'}, X_{i})^2}$, $d_{X_{i}}(X_{i'}, X_{i})$ represents the euclidian distance between the k-dimensional vectors $X_{i'}$ and $X_i$ constrained to the coordinates for which the observation $X_i$ is defined.  

Intuitively, this generalized kernel approximation, computes the standard contribution for the $j^{th}$ component of the $i^{th}$ observation if it is known. If the $j^{th}$ coordinate of $X_i$ is missing, that term is replaced by a prior computed from the data for which all the components are known. This prior, is weighted for each sample, projecting each complete observation $X_{i'}$ into the hyperplane defined by the missing coordinates of $X_i$.

# The approach described above is implemented in 
from stats import kernel_based_pdf_estimation
help(kernel_based_pdf_estimation)

### Other state of the art approaches

The most common approach to handle missing data, consists of defining an imputation criteria, then generate a set observations for which all the coordinates of all the observations are available, and finally, use standard pdf estimation algorithms. We compare the approximation method described before with classical imputation methods: mean, median, MICE, and Knn. Check sklearn documentation and the references listed below for more information about these algorithms. 

https://www.statsmodels.org/stable/user-guide.html#background
https://machinelearningmastery.com/iterative-imputation-for-missing-values-in-machine-learning/

from utils import compare_imputation_methods

dataset_names = ['moons','circles','blobs']
for dataset in dataset_names:
    compare_imputation_methods(dataset=dataset, 
                               num_samples=150, 
                               percent_missing=70, 
                               kernel_bandwidth=.2)

### What if the prediction sample has missing values?

We focus previously on dealing with missing values on the data available to estimate the pdf ($X_i$). Sometimes, we want to estimate the probability of an observed sample with values $u$, but some of the coordinates could be missing. Think for example, that you have the PDF to estimate the probability of the salary of a person based on their educations, years of experience, and age; but for some reason, the age is missing. 

As in {eq}`eq:kernel_estimation_mdim2` we use the $j$ subindex to denote the coordinate, $u=\{u_j\}$ with $j=1,...,k$ (as before, we are working on a $k$-dimensional feature space). Let us define the set $N = [j_1, ..., j_p]$ the set of coordinates for which $u$ is known. When we know all the coordinates, we are essentially estimating the density distribution associated with a point in the k-dimensional feature space. If we don't know one axis, we can think as estimating the density along a line (associated to all the point that share the known axis). If we don't know two coordinates we are estimating the density associated with a 2D plane and so on. In general, the density associated with the hyperplane of unknown coordinates can be estimated as:

$$
\hat{f}_N(u) = \int_{u_j j\notin N} \hat{f}(u) =  \int_{u_j j\notin N} \frac 1 n \sum_i \prod_j f_{ij}(u).
$$(eq:kernel_estimation_missing_values)

Using the definition $f_{ij}(u)$ introduces in {eq}`eq:fij_unknown_j`, and standard properties of calculus, we have, 
 
$$
\hat{f}_N(u) = \int_{u_j j\notin N} \frac 1 n \sum_i \prod_j f_{ij}(u) = \frac 1 n \sum_i \prod_{j \in N} f_{ij}(u) \prod_{j \notin N} \int_{-\infty}^{+\infty} \frac{1}{h\sqrt{2\pi}} e^{-\frac{1}{2 h^2} (u_j-X_{ij})^2} du_{j} = \frac 1 n \sum_i \prod_{j \in N} f_{ij}(u). $$(eq:kernel_estimation_missing_values2)

Equation {eq}`eq:kernel_estimation_missing_values2` proves that restricting the prediction to a subset of features, is as simple as computing the pdf restricting the training data to only those known features. As we show, the mathetical interpretation is that we will be estimating the density in a hyperplance of the feature space rather than a single point. 

