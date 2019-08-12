import numpy as np
import linear_regression_models as lin

def logistic_function(w, x):
    return 1/(1 + np.exp(- x @ w))

def logistic_loss(y, pred):
    return np.mean( - y * np.log(pred) - (1 - y) * np.log(1 - pred) ) 

def softmax_function(W, x):
    out = np.array([ np.exp(x @ W[:,k]) for k in range(W.shape[1]) ])
    if len(x.shape) > 1:
        out = out.transpose()
        return out / np.sum(out, axis=1)[:,None]
    else:
        return out / np.sum(out)

def softmax_loss(y, pred):
    res = np.array( [ y[:,k] * np.log(pred[:,k]) for k in range(y.shape[1]) ] ).transpose()
    return - np.mean(np.sum(res, axis=1))
        
def gd(x, y, num_epochs=100, alpha=0.001, w_initial=None, poly_order=1, build_regressors=True, compute_loss=True):
        
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x, poly_order=poly_order)    
    else:
        x_matrix = x       
        
    if len(y.shape) == 1:   
        y = y[:,None]
    
    if w_initial is None:
        w = np.zeros((x_matrix.shape[1], y.shape[1]))
    else:
        w = w_initial
        
    output_function = logistic_function
    loss = logistic_loss
    if w.shape[1] > 1:
        output_function = softmax_function
        loss = softmax_loss
    
    loss_history = []
    for epoch in range(num_epochs):
        error = y - output_function(w, x_matrix)
        for k in range(w.shape[1]):
            w[:,k] += alpha * np.mean(error[:,k:k+1] * x_matrix, axis=0) 
        if compute_loss:
            loss_history.append(loss(y, output_function(w, x_matrix)))
      
    if compute_loss:
        return {'w': w, 'loss_history': loss_history}
    else:
        return {'w': w}
    
def lms(x, y, num_epochs=100, alpha=0.001, w_initial=None, poly_order=1, build_regressors=True, compute_loss=True):
    
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x, poly_order=poly_order)    
    else:
        x_matrix = x     
        
    if len(y.shape) == 1:   
        y = y[:,None]
    
    if w_initial is None:
        w = np.zeros((x_matrix.shape[1], y.shape[1]))
    else:
        w = w_initial
    
    output_function = logistic_function
    loss = logistic_loss
    if w.shape[1] > 1:
        output_function = softmax_function
        loss = softmax_loss
    
    loss_history = []
    for epoch in range(num_epochs):
        random_permutation = np.random.permutation(y.shape[0])
        for xi, yi in zip(x_matrix[random_permutation], y[random_permutation]):            
            error = yi - output_function(w, xi)
            for k in range(w.shape[1]):
                w[:,k] += alpha * error[k] * xi
        if compute_loss:
            loss_history.append(loss(y, output_function(w, x_matrix)))
    
    if compute_loss:
        return {'w': w, 'loss_history': loss_history}
    else:
        return {'w': w}

def predict(w, x, y=None, poly_order=1, build_regressors=True):
        
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x, poly_order=poly_order)    
    else:
        x_matrix = x    
        
    output_function = logistic_function
    loss = logistic_loss
    if w.shape[1] > 1:
        output_function = softmax_function
        loss = softmax_loss
    
    if y is None:
        return output_function(w, x_matrix)
    else:
        pred = output_function(w, x_matrix)
        return {'pred': pred, 'loss': loss(y, output_function(w, x_matrix))}