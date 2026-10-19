import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	a,t,s=np.array(A),np.array(T),np.array(S)
	def invertable(x):
		m,n=x.shape
		if m!=n or np.linalg.det(x)==0:
			return False
		else:
			return True 
	
	if not invertable(s) or not invertable(t):
		return -1
	
	transformed_matrix=np.linalg.inv((t))@a@s #for matrix multiplication we use @ , for elementwise multiplication we use *
	return transformed_matrix