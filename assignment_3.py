#!/usr/bin/python
# -*- coding: utf-8 -*-

import graphlab as gl
import matplotlib.pyplot as plt
#

def polynomial_sframe(feature, degree):
	# assume that degree >= 1
	# initialize the SFrame:
	output = gl.SFrame()
	# and set poly_sframe['power_1'] equal to the passed feature
	
	output['power_1'] = feature

	# first check if degree > 1
	if degree > 1:
		# then loop over the remaining degrees:
		for power in range(2, degree+1):
			# first we'll give the column a name:
			name = 'power_' + str(power)
			# assign poly_sframe[name] to be feature^power
			output[name] = feature.apply(lambda x: x ** power)
	return output

def get_RSS(input_feature, output, model):
	
	prediction = model.predict(input_feature)
	RSS = ((output - prediction) ** 2).sum()
	
	return RSS

sales = gl.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living','price'])

# first degree model
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']

model1 = gl.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)

print("plotting the model (1st degree complexity)")

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
poly1_data['power_1'], model1.predict(poly1_data),'-')
plt.show()

# second degree model

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']

model2 = gl.linear_regression.create(poly2_data, target = 'price', features = ['power_1','power_2'], validation_set = None)

print("plotting the 2nd degree model")

plt.plot(poly2_data['power_1'], poly2_data['price'], '.',
poly2_data['power_1'], model2.predict(poly2_data),'-')
plt.show()

# third degree model

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
poly3_data['price'] = sales['price']
plt.show()

model3 = gl.linear_regression.create(poly3_data, target = 'price', features = ['power_1','power_2','power_3'], validation_set = None)

print("plotting the 3rd degree model")

plt.plot(poly3_data['power_1'], poly3_data['price'], '.',
poly3_data['power_1'], model3.predict(poly3_data), '-')

# 15th degree model

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
poly15_data['price'] = sales['price']
plt.show()

model15 = gl.linear_regression.create(poly15_data, target = 'price', 
features = ['power_1','power_2','power_3','power_4','power_5','power_6','power_7',
'power_8','power_9','power_10','power_11','power_12','power_13','power_14','power_15'], validation_set = None)

print("plotting the 15th degree model")

plt.plot(poly15_data['power_1'], poly15_data['price'], '.',
poly15_data['power_1'], model15.predict(poly15_data), '-')
plt.show()

## fit 15th degree model on 4 different sets

set1, set2 = sales.random_split(0.5, seed = 0)

set1, set3 = set1.random_split(0.5, seed = 0)

set2, set4 = set2.random_split(0.5, seed = 0)

print("fit the 15th degree model on the 4 subsets")

set_1 = polynomial_sframe(set1['sqft_living'], 15)
set_1['price'] = set1['price']
set_2 = polynomial_sframe(set2['sqft_living'], 15)
set_2['price'] = set2['price']
set_3 = polynomial_sframe(set3['sqft_living'], 15)
set_3['price'] = set3['price']
set_4 = polynomial_sframe(set4['sqft_living'], 15)
set_4['price'] = set4['price']

model15_1 = gl.linear_regression.create(set_1, target = 'price', 
features = ['power_1','power_2','power_3','power_4','power_5','power_6','power_7',
'power_8','power_9','power_10','power_11','power_12','power_13','power_14','power_15'], validation_set = None)

print("Coefficients of model15_1")
model15_1['coefficients'].print_rows(16)

print("plot for model15_1")

plt.plot(set_1['power_1'], set_1['price'], '.',
set_1['power_1'], model15_1.predict(set_1), '-')
plt.show()

model15_2 = gl.linear_regression.create(set_2, target = 'price', 
features = ['power_1','power_2','power_3','power_4','power_5','power_6','power_7',
'power_8','power_9','power_10','power_11','power_12','power_13','power_14','power_15'], validation_set = None)

print("Coefficients of model15_2")
model15_2['coefficients'].print_rows(16)

print("plot for model15_2")

plt.plot(set_2['power_1'], set_2['price'], '.',
set_2['power_1'], model15_2.predict(set_2), '-')
plt.show()

model15_3 = gl.linear_regression.create(set_3, target = 'price', 
features = ['power_1','power_2','power_3','power_4','power_5','power_6','power_7',
'power_8','power_9','power_10','power_11','power_12','power_13','power_14','power_15'], validation_set = None)

print("Coefficients of model15_3")
model15_3['coefficients'].print_rows(16)

print "plot for model15_3"

plt.plot(set_3['power_1'], set_3['price'], '.',
set_3['power_1'], model15_3.predict(set_3), '-')
plt.show()

model15_4 = gl.linear_regression.create(set_4, target = 'price', 
features = ['power_1','power_2','power_3','power_4','power_5','power_6','power_7',
'power_8','power_9','power_10','power_11','power_12','power_13','power_14','power_15'], validation_set = None)

print "Coefficients of model15_4"
model15_4['coefficients'].print_rows(16)

print "plot for model15_4"

plt.plot(set_4['power_1'], set_4['price'], '.',
set_4['power_1'], model15_4.predict(set_4), '-')
plt.show()

## finding the optimal polynomial degree

training_and_validation, test_data = sales.random_split(0.9, seed=1)
train_data, validation = training_and_validation.random_split(0.5, seed=1)

minRSS = 1e40
opt_degree = 0

for degree in range(1,16):
	print "Fitting " + str(degree) + "th degree model..."
	t_data = polynomial_sframe(train_data['sqft_living'], degree)
	t_data['price'] = train_data['price']
	feature_list = t_data.column_names()[:degree]
	current_mdl = gl.linear_regression.create(t_data, target = 'price', features = feature_list, validation_set = None, verbose = False)

	v_data = polynomial_sframe(validation['sqft_living'], degree)
	v_data['price'] = validation['price']

	print "RSS of " + str(degree) + "th degree model:"
	_eval_ = get_RSS(v_data[feature_list], v_data['price'], current_mdl)
	print _eval_
	print ""
	if _eval_ < minRSS:
		minRSS = _eval_
		opt_degree = degree
		print "optimal degree updated to " + str(degree)
		print ""
		
		tst_data = polynomial_sframe(test_data['sqft_living'], opt_degree)
		tst_data['price'] = test_data['price']
		
		print "evaluation of " + str(opt_degree) + "th degree on test data"
		test_RSS = get_RSS(tst_data[feature_list], tst_data['price'], current_mdl)
		print test_RSS
		print ""
