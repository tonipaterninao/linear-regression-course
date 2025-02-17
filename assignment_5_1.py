#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import log, sqrt
from sklearn import linear_model  # using scikit-learn

def get_RSS(input_feature, output, model):
	
	prediction = model.predict(input_feature)
	RSS = ((output - prediction) ** 2).sum()
	
	return RSS

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights

print "Q1: Chosen coefficients (lambda = 500)..."
for i in range(0,len(model_all.coef_)):
	if model_all.coef_[i] != 0:
		print all_features[i]
print ""

testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

min_RSS = 1e999
min_l1 = 0
for l1_penalty in np.logspace(1,7,num=13):
	model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
	model.fit(training[all_features], training['price'])
	RSS = get_RSS(validation[all_features],validation['price'],model)
	if RSS < min_RSS:
		print "min_RSS and min_l1 updated"
		min_RSS = RSS
		min_l1 = l1_penalty

print "Q2: 'best' L1 penalty:"
print min_l1, min_RSS
print ""

## Applying best Lasso penalty on test data
model = linear_model.Lasso(alpha=min_l1, normalize = True)
model.fit(training[all_features], training['price'])
best_l1_RSS = get_RSS(testing[all_features], testing['price'],model)
print "RSS on testing data:"
print best_l1_RSS

print "Non-zero coefficients with the best_l1"
print np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)

## Rule of thumb: 7 sparse features
max_nonzeros = 7
list_nonzeros = []
for l1_penalty in np.logspace(1, 4, num=20):
	model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
	model.fit(training[all_features], training['price'])
	list_nonzeros.append(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))

## finding too small l1 penalties
print list_nonzeros
l1_penalty_min = 0
l1_penalty_max = 1e999
more_nonzeros = map(lambda x: x > max_nonzeros, list_nonzeros)
print more_nonzeros
less_nonzeros = map(lambda x: x < max_nonzeros, list_nonzeros)
print less_nonzeros

for i in range(0,len(list_nonzeros)):
	if more_nonzeros[i]:
		if np.logspace(1, 4, num=20)[i] > l1_penalty_min:
			l1_penalty_min = np.logspace(1, 4, num=20)[i]
	elif less_nonzeros[i]:
		if np.logspace(1, 4, num=20)[i] < l1_penalty_max:
			l1_penalty_max = np.logspace(1, 4, num=20)[i]

print "Q3: Small range for l1 rule of thumb:"
print l1_penalty_min,l1_penalty_max
print ""
	
min_RSS = 1e999
min_l1 = 0
print "Exploring small range..."
for l1_penalty in np.linspace(l1_penalty_min, l1_penalty_max, num=20):
	model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
	model.fit(training[all_features], training['price'])
	sparsity = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
	RSS = get_RSS(validation[all_features],validation['price'],model)
	if RSS < min_RSS and sparsity == max_nonzeros:
		print "min_RSS and min_l1 updated"
		min_RSS = RSS
		min_l1 = l1_penalty

print "Q4: 'best' L1 penalty (with sparsity = 7)"
print min_l1, min_RSS
print ""

model = linear_model.Lasso(alpha=min_l1, normalize=True)
model.fit(training[all_features], training['price'])

print "Q5: Chosen coefficients (lambda = "+ str(min_l1) +")..."
for i in range(0,len(model.coef_)):
	if model.coef_[i] != 0:
		print all_features[i]
print ""




