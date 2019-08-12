import numpy as np
import linear_regression_models as lin

def squared_euclidean_distance(xi, xj):
    return np.sum((xi - xj)**2)

def batch_squared_euclidean_distance(x1, x2):
    return -2 * x1 @ x2.T + np.sum(x1**2, axis=1)[:,None] + np.sum(x2**2, axis=1)

def manhattan_distance(xi, xj):
    return np.sum(np.abs(xi - xj))

def batch_manhattan_distance(x1, x2):
    return np.abs(x1[:,None] - x2).sum(-1)

def squared_mahalanobis_distance(xi, xj, cov_matrix, inv_cov_matrix=None):
    if inv_cov_matrix is None:
        inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * np.mean(np.diag(cov_matrix)) * 10 ** -6)
        
    return np.sum((xi - xj).T @ inv_cov_matrix @ (xi - xj))

def batch_squared_mahalanobis_distance(x1, x2, cov_matrix):
    inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * np.mean(np.diag(cov_matrix)) * 10 ** -6)    
        
    return np.sum((x1 @ inv_cov_matrix) * x1, axis=1)[:,None] + \
           np.sum((x2 @ inv_cov_matrix) * x2, axis=1) - 2 * (x1 @ inv_cov_matrix @ x2.T)

def create_distance_matrix(x_new, x, distance_metric):
    
    distance_metric = distance_metric.lower()
    
    if distance_metric == 'mahalanobis':
        mu = np.mean(x, axis=0)
        cov_matrix = (x - mu).T @ (x - mu) / x.shape[0]
        inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * np.mean(np.diag(cov_matrix)) * 10 ** -6)
    
    dist_matrix = np.zeros((x_new.shape[0], x.shape[0]))
    for i in range(x_new.shape[0]):
        for j in range(x.shape[0]):
            if distance_metric == 'manhattan':
                dist_matrix[i,j] = manhattan_distance(x_new[i,:], x[j,:])
            elif distance_metric == 'mahalanobis':
                dist_matrix[i,j] = squared_mahalanobis_distance(x_new[i,:], x[j,:],
                                                                cov_matrix=None, inv_cov_matrix=inv_cov_matrix)
            else:
                dist_matrix[i,j] = squared_euclidean_distance(x_new[i,:], x[j,:])
    
    return dist_matrix

def create_batch_distance_matrix(x_new, x, distance_metric):
    
    distance_metric = distance_metric.lower()
    
    if distance_metric == 'manhattan':
        dist_matrix = batch_manhattan_distance(x_new, x)
        
    elif distance_metric == 'mahalanobis':
        mu = np.mean(x, axis=0)
        cov_matrix = (x - mu).T @ (x - mu) / x.shape[0]        
        dist_matrix = batch_squared_mahalanobis_distance(x_new, x, cov_matrix=cov_matrix)
        
    else:
        dist_matrix = batch_squared_euclidean_distance(x_new, x)
    
    return dist_matrix

def predict_class(x, y, x_new, k=1, distance_metric='euclidean'):
    
    classes = np.unique(y)
    
    dist_matrix = create_batch_distance_matrix(x_new, x, distance_metric)
    
    knn = np.argsort(dist_matrix)[:,0:k]

    return classes[np.argmax(np.apply_along_axis(np.bincount, 1, y[knn], minlength=classes.shape[0]), axis=1)]

def predict_regression(x, y, x_new, k=1, weights='uniform', distance_metric='euclidean'):
    
    dist_matrix = create_batch_distance_matrix(x_new, x, distance_metric)
    
    knn = np.argsort(dist_matrix)[:,0:k]
    
    if weights.lower() == 'distance':
        k_weights = 1 / dist_matrix[np.arange(dist_matrix.shape[0])[:,None],knn]
        return np.sum(y[knn] * k_weights, axis=1) / np.sum(k_weights, axis=1)
    
    else:
        return np.mean(y[knn], axis=1)