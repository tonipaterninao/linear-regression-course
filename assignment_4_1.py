#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

def polynomial_sframe(feature, degree):
	# assume that degree >= 1
	# initialize the SFrame:
	output = pd.DataFrame()
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

def k_fold_cross_validation(k, l2_penalty, data, output):
	n = len(data)
	l_model = linear_model.Ridge(alpha= l2_penalty, normalize=True)
	RSS_list = []

	for i in range(0,k-1):
		# slice the training data into training and validation
		start = (n*i)/k
		end = (n*(i+1)/k)-1
		v_set = data[start:end+1]
		v_out = output[start:end+1]
		t_set = data[0:start].append(data[end+1:n])
		t_out = output[0:start].append(output[end+1:n])
		# fit the model using the training data
		i_model = l_model.fit(t_set, t_out)
		# predict with current model
		prediction = i_model.predict(v_set)
		RSS = ((v_out - prediction) ** 2).sum()
		RSS_list.append(RSS)
	
	return sum(RSS_list)/len(RSS_list)

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living','price'])

l2_small_penalty = 1.5e-5

poly15_data = polynomial_sframe(sales['sqft_living'], 15) # use equivalent of `polynomial_sframe`
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])
print "Q1: coefficients of model fit with poly15_data"
print model.coef_

# dtype_dict same as above
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_1_pol = polynomial_sframe(set_1['sqft_living'], 15)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_2_pol = polynomial_sframe(set_2['sqft_living'], 15)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_3_pol = polynomial_sframe(set_3['sqft_living'], 15)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
set_4_pol = polynomial_sframe(set_4['sqft_living'], 15)

# new l2 penalty
l2_small_penalty=1e-9

model = linear_model.Ridge(alpha= l2_small_penalty, normalize=True)

print "Fitting models on all four sets (ANSWER TO Q2):"

# fit the new model on the four datasets
model_1 = model.fit(set_1_pol, set_1['price'])
print "Coefficients for set_1:"
print model_1.coef_

model_2 = model.fit(set_2_pol, set_2['price'])
print "Coefficients for set_2:"
print model_2.coef_

model_3 = model.fit(set_3_pol, set_3['price'])
print "Coefficients for set_3:"
print model_3.coef_

model_4 = model.fit(set_4_pol, set_4['price'])
print "Coefficients for set_4:"
print model_4.coef_

plt.plot(set_1_pol['power_1'], set_1['price'], '.',
set_1_pol['power_1'], model_1.predict(set_1_pol), '-')
#plt.show()

plt.plot(set_2_pol['power_1'], set_2['price'], '.',
set_2_pol['power_1'], model_2.predict(set_2_pol), '-')
#plt.show()

plt.plot(set_3_pol['power_1'], set_3['price'], '.',
set_3_pol['power_1'], model_3.predict(set_3_pol), '-')
#plt.show()

plt.plot(set_4_pol['power_1'], set_4['price'], '.',
set_4_pol['power_1'], model_4.predict(set_4_pol), '-')
#plt.show()

print "Fitting models with large penalty (Q3):"
l2_large_penalty=1.23e2
model = linear_model.Ridge(alpha= l2_large_penalty, normalize=True)

model_1 = model.fit(set_1_pol, set_1['price'])
print "Coefficients for set_1:"
print model_1.coef_

model_2 = model.fit(set_2_pol, set_2['price'])
print "Coefficients for set_2:"
print model_2.coef_

model_3 = model.fit(set_3_pol, set_3['price'])
print "Coefficients for set_3:"
print model_3.coef_

model_4 = model.fit(set_4_pol, set_4['price'])
print "Coefficients for set_4:"
print model_4.coef_

plt.plot(set_1_pol['power_1'], set_1['price'], '.',
set_1_pol['power_1'], model_1.predict(set_1_pol), '-')
#plt.show()

plt.plot(set_2_pol['power_1'], set_2['price'], '.',
set_2_pol['power_1'], model_2.predict(set_2_pol), '-')
#plt.show()

plt.plot(set_3_pol['power_1'], set_3['price'], '.',
set_3_pol['power_1'], model_3.predict(set_3_pol), '-')
#plt.show()

plt.plot(set_4_pol['power_1'], set_4['price'], '.',
set_4_pol['power_1'], model_4.predict(set_4_pol), '-')
#plt.show()

print "CROSS VALIDATION"
train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
	start = (n*i)/k
	end = (n*(i+1)/k)-1
	print i, (start, end)

data_pol = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
train_price = train_valid_shuffled['price']

min_val_error = 1e999
opt_l2 = 0

for l2 in np.logspace(3,9,num=13):
	print "l2_penalty : " + str(l2)
	avg_error = k_fold_cross_validation(k=10, l2_penalty=l2, data=data_pol, output=train_price)
	print "avg error : " + str(avg_error)
	if avg_error < min_val_error:
		min_val_error = avg_error
		opt_l2 = l2
	print "min_val_error : " + str(min_val_error)
	print "opt_l2 : " + str(opt_l2)

print "Q5: 10-fold cross-validation's optimal L2 penalty:"
print opt_l2

print "Using best L2 penalty on training set"

opt_model = linear_model.Ridge(alpha = opt_l2, normalize = True)
opt_model.fit(data_pol, train_price)

print "Predicting on test set..."
test_pol = polynomial_sframe(test['sqft_living'], 15)
prediction = opt_model.predict(test_pol)
RSS_test = ((test['price'] - prediction) ** 2).sum()
print "Q6: RSS on test data set using opt. L2 penalty:"
print RSS_test
