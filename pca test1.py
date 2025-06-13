import numpy as np

# Original Data 
X = np.array([
    [2, 3, 4, 5, 6, 7, 8, 9],
    [3, 5, 7, 8, 8, 10, 13, 14]
])

# Step 1: Center the data
mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - mean
print("Centered Data:\n", X_centered)

# Step 2: Covariance matrix
cov_matrix = np.cov(X_centered)
print("Covariance Matrix:\n", cov_matrix)

# Step 3: Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Step 4: Sort eigenvectors by eigenvalues (descending)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("Sorted Eigenvalues:\n", eigenvalues)
print("Sorted Eigenvectors:\n", eigenvectors)

# Step 5: Project the data
PC1 = eigenvectors[:, 0].reshape(2, 1)
Y = PC1.T @ X_centered

print("Projected Data (1D):", Y)

# Step 6: Reconstruct the data from projection
X_reconstructed = PC1 @ Y + mean

print("Reconstructed Data (approx):\n", X_reconstructed)

