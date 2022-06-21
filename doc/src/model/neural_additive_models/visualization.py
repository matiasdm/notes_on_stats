import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, auc


def plot_roc_curves(results, ax = None):

    if ax is None:
        fig, ax =  plt.subplots(1, 1, figsize=(5, 5))
    
    for _, res in results.groupby('replicate'):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_pred'])      
        roc_auc = auc(fpr, tpr)    
        ax.plot(fpr, tpr, '-', color='orange', lw=0.5)

    fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, '-', color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc,)
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    return ax
   


def plot_prediction_plot(results, pred_col, resp_col, axes=None, size = (7, 5), fname = None):

    plt.figure(figsize=size)
    x = results[pred_col]
    y = results[resp_col]
    plt.scatter(x, y, color='darkorange', lw=1.0)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='navy', lw=1.5, linestyle='--')
    plt.grid()
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show() 


        

def plot_shape_functions(results, features,  axes=None, ncols=4, start_axes_plotting=None):

    n = len(features)
    nrows = n // ncols

    if axes is None:
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize); axes = axes.flatten()


    for i, feature in enumerate(features):

        if start_axes_plotting is not None:
            i+=start_axes_plotting

        results.sort_values(feature, inplace = True)
        for _, res in results.groupby('replicate'):
            x = (res[[feature, feature + '_partial']]
                .drop_duplicates(subset = feature)
                .set_index(feature))

            x.plot.line(ax = axes[i], 
                        color = 'orange', 
                        lw = 0.25)

        x = results.pivot_table(index = feature, 
                                columns = 'replicate', 
                                values = feature + '_partial')

        x = (x
            .interpolate()
            .mean(axis = 1)
            .rename(feature + '_partial')
            .sort_index())

        x.plot.line(ax = axes[i], 
                    color = 'orange', 
                    lw = 1.5)

        # Plot frequencies        
        twin = axes[i].twinx()
        x = results[feature]
        if x.dtype == object:
            x.value_counts().plot.bar(
                width = 1, 
                alpha = .15, 
                ax = twin,
            )    

            labs = [l.get_text() for l in twin.get_xticklabels()]
            axes[i].set_xticklabels(labs, rotation=45, ha='right')

        else:
            x.plot.hist(alpha = .15, 
                        bins=20,
                        ax = twin)
        twin.set_ylabel('frequnecy')


        axes[i].grid(True)
        axes[i].set_ylabel('Shape functions')
        axes[i].xaxis.label.set_visible(False)
        axes[i].set_title(feature.replace('_', ' '))
        axes[i].get_legend().remove()
    
    return axes


