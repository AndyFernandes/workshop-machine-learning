import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Matplolib defaults
figsize = (8,6)
fontsize = 20
markersize = 12
linewidth = 2
dpi = 300
plt.rcParams.update({'figure.autolayout': True})

def normalize_data(x, ignore_first=False):
    
    x_normalized = x.copy()
    
    if ignore_first:
        x_mean = np.mean(x_normalized[:,1:], axis=0)        
        x_normalized[:,1:] -= x_mean
        x_std = np.std(x_normalized[:,1:], axis=0)
        x_normalized[:,1:] /= x_std
        
    else:
        x_mean = np.mean(x_normalized, axis=0)        
        x_normalized -= x_mean
        x_std = np.std(x_normalized, axis=0)
        x_normalized /= x_std
    
    return {'data': x_normalized, 'mean': x_mean, 'std': x_std}

def plot_loss_path(loss, title=None, save=False, file_name='img.png'):
    
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    plt.plot(range(1,len(loss)+1), loss, '-k')
    plt.xlabel('Iterações', fontsize=fontsize)
    plt.ylabel('Função custo', fontsize=fontsize)
    #plt.tight_layout()
    
    if title is not None:
        plt.title(title, fontsize=fontsize)        
    
    if save:
        plt.savefig(file_name, dpi=dpi)
        
    plt.show()
    
def plot_regression_line(x, y, x_new, w, title=None, save=False,
                         file_name='img.png', xlab="", ylab=""):
    
    pred = x_new @ w 
    
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    if x.shape[1] > 1:
        plt.plot(x[:,1], y, 'ob', markersize=markersize)
    else:
        plt.plot(x, y, 'ob', markersize=markersize)
    plt.plot(x_new[:,1], pred, '-r', linewidth=linewidth)
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    aux_y = np.concatenate((y,pred))
    plt.ylim(np.min(aux_y) - 0.1*np.abs(np.min(aux_y)), np.max(aux_y) + 0.1*np.abs(np.max(aux_y)))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    #plt.tight_layout()
    if title is not None:
        plt.title(title, fontsize=fontsize)
        
    if save:
        plt.savefig(file_name, dpi=dpi)
        
    plt.show()  
    
def make_meshgrid(x, y, steps=300):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps),
                         np.linspace(y_min, y_max, steps))
    return xx, yy

def plot_contours(ax, w, clf, xx, yy, colors=['red', 'blue']):
    labels = clf(np.c_[xx.ravel(), yy.ravel()], w)
    labels = labels.reshape(xx.shape)
    out = ax.contourf(xx, yy, labels, levels=len(np.unique(labels))-1, colors=colors, alpha=0.5)
    #out = ax.contourf(xx, yy, labels, cmap=plt.cm.brg, alpha=0.5)
    return out

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(title)
    else:
        print(title)

    print(cm)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=figsize)    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    
    return ax