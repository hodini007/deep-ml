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
    # 1. Standardize the dataset: (X - mean) / std along each feature (axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero if a feature has constant values
    std_replaced = np.where(std == 0, 1.0, std)
    standardized_data = (data - mean) / std_replaced
    
    # 2. Compute the covariance matrix
    cov_matrix = np.cov(standardized_data, rowvar=False)
    
    # In case cov_matrix is a 0D scalar (when n_features == 1)
    cov_matrix = np.atleast_2d(cov_matrix)
    
    # 3. Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # 5. Extract top k components
    top_k_eigenvectors = eigenvectors[:, :k]
    
    # 6. Apply sign convention:
    # First element with |val| > 1e-10 must be positive
    for i in range(top_k_eigenvectors.shape[1]):
        vec = top_k_eigenvectors[:, i]
        for val in vec:
            if abs(val) > 1e-10:
                if val < 0:
                    top_k_eigenvectors[:, i] = -vec
                break
                
    # 7. Round to 4 decimal places
    return np.round(top_k_eigenvectors, 4)