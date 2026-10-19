import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    B = A.T @ A
    
    # 2. Extract components to compute the rotation angle theta
    b11, b12 = B[0, 0], B[0, 1]
    b22 = B[1, 1]
    
    # 3. Calculate theta to diagonalize B
    if np.abs(b12) < 1e-15:
        theta = 0.0
    else:
        theta = 0.5 * np.arctan2(2.0 * b12, b11 - b22)
        
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # 4. Construct the right singular vector matrix V
    V = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    Vt = V.T
    
    # 5. Compute the columns of U * Sigma
    # Multiplying A by V performs a one-sided transformation
    A_prime = A @ V
    
    # 6. Compute singular values (norms of the columns of A_prime)
    s0 = np.linalg.norm(A_prime[:, 0])
    s1 = np.linalg.norm(A_prime[:, 1])
    
    # 7. Normalize the columns of A_prime to get U
    # Guard against division by zero for singular/rank-deficient matrices
    u0 = A_prime[:, 0] / s0 if s0 > 1e-15 else np.array([1.0, 0.0])
    u1 = A_prime[:, 1] / s1 if s1 > 1e-15 else np.array([0.0, 1.0])
    U = np.column_stack((u0, u1))
    
    S = np.array([s0, s1])
    
    # 8. Ensure singular values are sorted in descending order
    if S[0] < S[1]:
        S = S[::-1]
        U = U[:, ::-1]
        Vt = Vt[::-1, :]
        
    return U, S, Vt