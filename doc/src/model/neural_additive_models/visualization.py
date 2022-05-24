import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, auc


def plot_roc_curves(results, pred_col, resp_col, size = (7, 5), fname = None):
    plt.clf()
    plt.style.use('classic')
    plt.figure(figsize=size)
    
    for _, res in results.groupby('replicate'):
        fpr, tpr, _ = roc_curve(res[resp_col], res[pred_col])      
        roc_auc = auc(fpr, tpr)    
        plt.plot(fpr, tpr, '-', color='orange', lw=0.5)

    fpr, tpr, _ = roc_curve(results[resp_col], results[pred_col])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, '-', color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc,)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()    


def plot_prediction_plot(results, pred_col, resp_col, size = (7, 5), fname = None):
    plt.clf()
    plt.style.use('classic')
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


        

def plot_shape_functions(results, features, ncols = 4, size = (25, 8), fname = None):

    n = len(features)
    nrows = n // ncols

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize); axes = axes.flatten()
    for i, feature in enumerate(features):
        r = i // ncols
        c = i % ncols

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
                        ax = twin)
        twin.set_ylabel('frequnecy')


        axes[i].grid(True)
        axes[i].set_ylabel('Shape functions')
        axes[i].xaxis.label.set_visible(False)
        axes[i].set_title(feature.replace('_', ' '))
        axes[i].get_legend().remove()

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
    return


