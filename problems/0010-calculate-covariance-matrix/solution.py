def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	import numpy as np
	ans=np.cov(vectors)
	return ans