# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from continuous_sinkhorn import *
import matplotlib.pyplot as plt
######### Gaussian experiments ###############

dim = 1
mu_1 = np.ones(dim)
mu_2 = np.ones(dim)

dataset_size = 1000
kernel_1 = gaussian_kernel
kernel_2 = gaussian_kernel
step_size = .01 #good setting .01
smoothing = .1 #good setting .01
rounds = 5000
cost_function = lp_cost

dataset_1 = np.random.multivariate_normal(mu_1, np.eye(dim), dataset_size)
dataset_2 = np.random.multivariate_normal(mu_2, np.eye(dim), dataset_size)

samples_x, samples_y, coefficients, lambda_x, lambda_y = get_test_functions_coefficients(dataset_1, dataset_2, kernel_1, kernel_2, step_size, smoothing, 
	rounds, cost_function, sampling_method = "random")

print(evaluate_sinkhorn_distance(dataset_1, dataset_2, lambda_x, lambda_y))


x_axis = np.linspace(-5,5, 400)
plt.plot(x_axis, [lambda_x(x) for x in x_axis])
plt.plot(x_axis, [lambda_y(x) for x in x_axis])
plt.show()
#plt.savefig("./test_func_plot.png")



import IPython
IPython.embed()