from continuous_sinkhorn import *

def mmd(n1, n2, gamma):
	## optimal f:
	mmd_value = 0
	m = n1.shape[1]
	n = n2.shape[1]
	for i in range(m):
		for j in range(m):
			mmd_value+= 1.0/(m**2) * gaussian_kernel(n1[:,i], n1[:,j], gamma)
	for i in range(n):
		for j in range(n):
			mmd_value+= 1.0/(n**2) * gaussian_kernel(n2[:,i], n2[:,j], gamma)
	for i in range(m):
		for j in range(n):
			mmd_value -= 2.0/(n*m) * gaussian_kernel(n1[:,i], n2[:,j], gamma)
	return mmd_value


def grad_mmd(n1, n2, gamma):
	grad1 = np.zeros(n1.shape)
	grad2 = np.zeros(n2.shape)
	m = n1.shape[1]
	n = n2.shape[1]

	## outer loop trough the columns of n1
	for index in range(m):
		## part of the gradient corresponding to the n1 dataset
		for i in range(m):
			if i != index:
				## Add gradient of 1/m^2*k(x_index, x_i) with respect to x_index
				#grad1[:, index] +=  1.0/(m**2)*gaussian_kernel(n1[:,index], n1[:,i], gamma)*(-1.0/(gamma**2))*2*(n1[:,index]- n1[:,i])		
				grad1[:, index] +=  -2.0/((gamma**2)*(m**2))*gaussian_kernel(n1[:,index], n1[:,i], gamma)*(n1[:,index]- n1[:,i])

			else:
				grad1[:, index] +=  0
		### part of the gradient corresponding to the n2 dataset for this point
		for j in range(n):
			#Chain rule explicit top / reduced bottom 
			#grad1[:, index] +=  -2.0/(m*n)*gaussian_kernel(n1[:,index], n2[:,j], gamma)*2*(-1.0/(gamma**2))*(n1[:,index]- n2[:,j])
			grad1[:, index] +=  4.0/((gamma**2)*m*n)*gaussian_kernel(n1[:,index], n2[:,j], gamma)*(n1[:,index]- n2[:,j])

	## outer loop trough the columns of n2
	for index in range(n):
		## part of the gradient corresponding to the n2 dataset
		for i in range(n):
			if i != index:
				grad2[:, index] +=  -2.0/((gamma**2)*(m**2))*gaussian_kernel(n2[:,index], n2[:,i], gamma)*(n2[:,index]- n2[:,i])

			else:
				grad2[:, index] +=  0
		### part of the gradient corresponding to the n1 dataset for this point
		for j in range(m):
			grad2[:, index] +=  4.0/((gamma**2)*m*n)*gaussian_kernel(n2[:,index], n1[:,j], gamma)*(n2[:,index]- n1[:,j])



	return grad1, grad2


n1 = np.array([1,4,5,6,7,8])
n1 = n1.reshape(1, len(n1))
n2 = np.array([1,4,5,6,7,8])#np.array([1,1,1,9])
n2 = n2.reshape(1, len(n2))

mmd_value = mmd(n1, n2, 0.1)

mmd_grad_values = grad_mmd(n1, n2, 1)
