import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
        
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    std_replaced = np.where(std == 0, 1.0, std)
    standardized_data = (data - mean) / std_replaced
    
    cov_matrix = np.cov(standardized_data, rowvar=False)
    
    cov_matrix = np.atleast_2d(cov_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    top_k_eigenvectors = eigenvectors[:, :k]
    
    
    for i in range(top_k_eigenvectors.shape[1]):
        vec = top_k_eigenvectors[:, i]
        for val in vec:
            if abs(val) > 1e-10:
                if val < 0:
                    top_k_eigenvectors[:, i] = -vec
                break
                
    return np.round(top_k_eigenvectors, 4)