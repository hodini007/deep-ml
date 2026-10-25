def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	import numpy as np
	P=np.linalg.inv(C) @ B 
	return P