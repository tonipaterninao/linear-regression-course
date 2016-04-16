#!/usr/bin/python
# -*- coding: utf-8 -*-

import graphlab as gl
import numpy as np
import math

def get_numpy_data(data_sframe, features, output):

	data_sframe['constant'] = 1 # add a constant column to an SFrame

	# prepend variable 'constant' to the features list
	features = ['constant'] + features

	# select the columns of data_SFrame given by the 'features' list into the SFrame 'features_sframe'
	features_matrix = gl.SFrame()

	for feature in features:
		features_matrix[feature] = data_sframe[feature]
	# this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
	features_matrix = features_matrix.to_numpy()

	# assign the column of data_sframe associated with the target to the variable 'output_array'
	output_array = data_sframe[output]

	# this will convert the SArray into a numpy array:
	output_array = output_array.to_numpy() # GraphLab Create>= 1.7!!
	return features_matrix, output_array

def predict_outcome(feature_matrix, weights):
	predictions = np.dot(feature_matrix, weights)
	return predictions

def feature_derivative(errors, feature):
	derivative = 2 * np.dot(feature, errors)
	return derivative

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
	converged = False
	weights = np.array(initial_weights)
	while not converged:
		# compute the predictions based on feature_matrix and weights:

		predictions = predict_outcome(feature_matrix, weights)

		# compute the errors as predictions - output:

		errors = predictions - output
		
		gradient_sum_squares = 0 # initialize the gradient
		
		# while not converged, update each weight individually:
		for i in range(len(weights)):
			# Recall that feature_matrix[:, i] is the feature column associated with weights[i]

			# compute the derivative for weight[i]:
			derivative = feature_derivative(errors[i], feature_matrix[:,i])
			type(derivative)
			# add the squared derivative to the gradient magnitude
			gradient_sum_squares = gradient_sum_squares + derivative ** 2
			
			# update the weight based on step size and derivative:
			w = weights[i] + (step_size * derivative)
			type(w)
			#weights[i] = w
			
		gradient_magnitude = np.sqrt(gradient_sum_squares)
		if gradient_magnitude.all() < tolerance:
			converged = True
	return(weights)

######################################################################################################

sales = gl.SFrame('kc_house_data.gl/')

train_data,test_data = sales.random_split(.8,seed=0)

simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

print "ANSWER TO Q10:\t" + str(simple_weights[1])

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

simple_prediction = predict_outcome(test_simple_feature_matrix, simple_weights)

print "ANSWER TO Q11:\t" + str(simple_prediction[0])

simple_RSS = ((test_output - simple_prediction) ** 2).sum()

## Running gradient descent on more than one feature

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

double_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

double_prediction = prediction_outcome(feature_matrix, double_weights)

print "ANSWER Q15:\t" + str(double_prediction[0])

print "ANSWER Q17:\t" + str(test_output[0])

double_RSS = ((test_output - double_prediction) ** 2).sum()

print "ANSWER Q19:\t" + "1: " + str(single_RSS) + "\t2: " + str(double_RSS)

