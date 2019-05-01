
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pickle
import random
import IPython


def generate_empirical_distribution(object_list, object_types):
	result = np.zeros(len(object_types))
	for obj in object_list:
		result[object_types.index(obj)] += 1.0
	return result / (1.0*len(object_list))

def dummy_cost(embedding1, embedding2):
    return 0

def dummy_big_cost(embedding1, embedding2):
    return 100


def dummy_test_function(embedding):
	return -1

### Utilities for continuous embeddings
def gaussian_kernel(x,y, sigma = 1):
	return np.exp(-np.dot(x-y, x-y)/(sigma**2))

def lp_cost(x, y, p=2):
	return np.linalg.norm(x-y)

def produce_test_function(datapoints, alphas, kernel, random_features = False):
	if random_features:
		raise ValueError("Option not implemented")
	else:
		def test_function(x):
			result = 0
			#print("alphas ", alphas)
			#print("datapoints ", datapoints)
			#print("embedding of datapoints ", [path_embedding(datapoints[i]) for i in range(len(datapoints))])
			for i in range(len(datapoints)):
				result += alphas[i]* kernel(datapoints[i], x)
			return result
		return test_function

def get_new_test_functions_coefficient(x_t, y_t, lambda_x, lambda_y, kernel_1, kernel_2, step_size, 
	smoothing, round_index, cost_function):
	
	coefficient = step_size*(1.0/np.sqrt(round_index))
	print(lambda_x(x_t), " lambda x_t" )
	print(lambda_y(y_t), " lambda y_t")
	print(cost_function(x_t, y_t), " cost function ")
	coefficient *= (1-np.exp( (lambda_x(x_t) + lambda_y(y_t) - cost_function(x_t, y_t))/smoothing ))
	#IPython.embed()
	return coefficient

def build_lambda(datapoints, coefficients, kernel):
	def lambda_star(x):
		value = 0
		for i in range(len(datapoints)):
			value += coefficients[i]*kernel(datapoints[i], x)
		return value
	return lambda_star

def get_test_functions_coefficients(dataset_1, dataset_2, kernel_1, kernel_2, step_size, smoothing, 
	rounds, cost_function, sampling_method = "random"):
	samples_x = []
	samples_y = []
	coefficients = []
	lambda_x = dummy_test_function
	lambda_y = dummy_test_function
	for round_index in range(1,rounds+1):
		if sampling_method == "sequential":
			x_t = dataset_1[round_index-1]
			y_t = dataset_2[round_index-1]
		elif sampling_method == "random":
			x_t = random.choice(dataset_1)
			y_t = random.choice(dataset_2)
		else:
			raise ValueError("Sampling method not available.")

		coefficient = get_new_test_functions_coefficient(x_t, y_t, lambda_x, lambda_y, kernel_1, kernel_2, 
					step_size, smoothing, round_index, cost_function)
		print(coefficient, " coefficient ")
		coefficients.append(coefficient)
		samples_x.append(x_t)
		samples_y.append(y_t)
		lambda_x = build_lambda(samples_x, coefficients, kernel_1)
		lambda_y = build_lambda(samples_y, coefficients, kernel_2)

	return samples_x, samples_y, coefficients, lambda_x, lambda_y

## This does not include the entropy term.
def evaluate_sinkhorn_distance(dataset_1, dataset_2, lambda_x, lambda_y):
	expectation_x = 0
	expectation_y = 0
	for datapoint_x in dataset_1:
		expectation_x += lambda_x(datapoint_x)
	expectation_x = 1.0/len(dataset_1)*expectation_x
	for datapoint_y in dataset_2:
		expectation_y += lambda_y(datapoint_y)
	expectation_y = 1.0/len(dataset_2)*expectation_y

	return expectation_x + expectation_y




def penalized_linear_regression( dataset_1, responses_1, dataset_2, responses_2, kernel_1, 
	kernel_2, cost_function,  ):
	dimension = len(dataset_1[0])
	## initialize theta at some random point
	theta = np.random.normal(0, 1, dimension)

	
	
