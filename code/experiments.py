from continuous_sinkhorn import *

######### Gaussian experiments ###############

dim = 10
mu_1 = np.zeros(dim)
mu_2 = np.zeros(dim)

dataset_size = 1000
kernel_1 = gaussian_kernel
kernel_2 = gaussian_kernel
step_size = .1
smoothing = .1
rounds = 1000
cost_function = lp_cost

dataset_1 = np.random.multivariate_normal(mu_1, np.eye(dim), dataset_size)
dataset_2 = np.random.multivariate_normal(mu_2, np.eye(dim), dataset_size)

samples_x, samples_y, coefficients, lambda_x, lambda_y = get_test_functions_coefficients(dataset_1, dataset_2, kernel_1, kernel_2, step_size, smoothing, 
	rounds, cost_function, sampling_method = "sequential")

print(evaluate_sinkhorn_distance(dataset_1, dataset_2, lambda_x, lambda_y))