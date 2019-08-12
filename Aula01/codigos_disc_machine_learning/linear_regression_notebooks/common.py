import numpy as np
import matplotlib.pyplot as plt

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
    
def make_meshgrid(x, y, h=0.01):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, w, clf, xx, yy, colors=['red', 'blue']):
    labels = clf(np.c_[xx.ravel(), yy.ravel()], w)
    labels = labels.reshape(xx.shape)
    out = ax.contourf(xx, yy, labels, levels=len(np.unique(labels))-1, colors=colors, alpha=0.5)
    #out = ax.contourf(xx, yy, labels, cmap=plt.cm.brg, alpha=0.5)
    return out