from math import  sqrt
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	b=matrix[0][0]+matrix[1][1]
	c=(matrix[0][0]*matrix[1][1])-(matrix[0][1]*matrix[1][0])
	val1=((b)+sqrt(b**2-4*c))/2
	val2=((b)-sqrt(b**2-4*c))/2
	eigenvalues=[val1,val2]
	eigenvalues.sort(reverse=True)
	return eigenvalues