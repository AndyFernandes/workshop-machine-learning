import numpy as np

def build_poly_regressors(x, poly_order=1):
    
    poly_regressors = np.hstack((np.ones((x.shape[0], 1)), x))
    
    if poly_order > 1:
        for i in range(2,poly_order+1):        
            poly_regressors = np.hstack((poly_regressors, x**i))
            
    return poly_regressors

def gd(x, y, num_epochs=100, alpha=0.001, w_initial=None, poly_order=1, lamb=None, build_regressors=True):
        
    if build_regressors:
        x_matrix = build_poly_regressors(x, poly_order=poly_order)    
    else:
        x_matrix = x       
    
    if w_initial is None:
        w = np.zeros(x_matrix.shape[1])
    else:
        w = w_initial
    
    mse_history = []
    for epoch in range(num_epochs):
        error = y - x_matrix @ w
        if lamb is None:
            w += alpha * (np.mean(error[:, None] * x_matrix, axis=0))
            mse_history.append(np.mean((y - x_matrix @ w)**2))
        else:
            w += alpha * np.mean(error[:, None] * x_matrix, axis=0)
            mse_history.append(np.mean((y - x_matrix @ w)**2) + lamb * np.sum(w**2))
            
    return {'w': w, 'mse_history': mse_history}
    
def lms(x, y, num_epochs=100, alpha=0.001, w_initial=None, poly_order=1, lamb=None, build_regressors=True):
    
    if build_regressors:
        x_matrix = build_poly_regressors(x, poly_order=poly_order)    
    else:
        x_matrix = x        
    
    if w_initial is None:
        w = np.zeros(x_matrix.shape[1])
    else:
        w = w_initial.copy()
    
    mse_history = []
    for epoch in range(num_epochs):
        random_permutation = np.random.permutation(y.shape[0])
        for xi, yi in zip(x_matrix[random_permutation], y[random_permutation]):
            error = yi - w @ xi            
            if lamb is None:                
                w += alpha * error * xi
                mse_history.append(np.mean((y - x_matrix @ w)**2))
            else:
                w += alpha * (error * xi - lamb * w)
                mse_history.append(np.mean((y - x_matrix @ w)**2) + lamb * np.sum(w**2))
        
    return {'w': w, 'mse_history': mse_history}
    
def ols(x, y, poly_order=1, lamb=None, build_regressors=True):
    
    if build_regressors:
        x_matrix = build_poly_regressors(x, poly_order=poly_order)    
    else:
        x_matrix = x       
    
    if lamb is None:
        w = np.linalg.pinv(x_matrix) @ y
    else:
        reg_matrix = lamb*np.eye(x_matrix.shape[1])
        reg_matrix[0,0] = 0
        w = np.linalg.inv(x_matrix.transpose() @ x_matrix + reg_matrix) @ x_matrix.transpose() @ y
            
    mse = np.mean((y - x_matrix @ w)**2)
    
    return {'w': w, 'mse': mse}

def predict(w, x, y=None, poly_order=1, build_regressors=True):
        
    if build_regressors:
        x_matrix = build_poly_regressors(x, poly_order=poly_order)    
    else:
        x_matrix = x       
    
    if y is None:
        return x_matrix @ w
    else:
        pred = x_matrix @ w
        return {'pred': pred, 'mse': np.mean((y - pred)**2)}
    