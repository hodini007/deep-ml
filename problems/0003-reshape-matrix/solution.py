import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	size1=sum(len(sublist) for sublist in a)
	if size1 != (new_shape[0]*new_shape[1]):
		return []
	reshaped_matrix=np.reshape(a, new_shape)
	return reshaped_matrix