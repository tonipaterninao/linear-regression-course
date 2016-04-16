#!/usr/bin/python

import graphlab as gl
from numpy import log

# import the data

sales = gl.SFrame('kc_house_data.gl/')

## Add transformed variables

sales['bedrooms_sq'] = sales['bedrooms'] * sales['bedrooms']
sales['bed_bath_rooms'] = sales['bedrooms'] * sales['bathrooms']
sales['log_sqft_living'] = log(sales['sqft_living'])
sales['lat_plus_long'] = sales['lat'] + sales['long']

train_data,test_data = sales.random_split(.8,seed=0)

## Answer to Q4

print 'ANSWER TO Q4'
print "Bedrooms squared:\t" + str(test_data['bedrooms_sq'].mean())
print "Bedrooms * bathrooms:\t" + str(test_data['bed_bath_rooms'].mean())
print "Logarithm sqft:\t" + str(test_data['log_sqft_living'].mean())
print "Latitude + longitude:\t" + str(test_data['lat_plus_long'].mean())
print ""

## Build the models

model_1 = gl.linear_regression.create(train_data, target = 'price',
								features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'],
								validation_set = None)
print ""
model_2 = gl.linear_regression.create(train_data, target = 'price',
								features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms'],
								validation_set = None)
print ""
model_3 = gl.linear_regression.create(train_data, target = 'price',
								features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms',
											'bedrooms_sq', 'log_sqft_living', 'lat_plus_long'],
								validation_set = None)
print ""
## Answer to Q5

print 'ANSWER TO Q5:\tmodel_1 coefficients'
coefs_1 = model_1['coefficients']
print coefs_1
print ""
print 'ANSWER TO Q6:\tmodel_2 coefficients'
coefs_2 = model_2['coefficients']
print coefs_2
print ""
## Compute the RSS for each model

pred_1 = model_1.predict(train_data)
eval1 = model_1.evaluate(train_data)
print 'RSS1'
print eval1
print ""

pred_2 = model_2.predict(train_data)
eval2 = model_2.evaluate(train_data)
print 'RSS2'
print eval2
print ""

pred_3 = model_3.predict(train_data)
eval3 = model_3.evaluate(train_data)
print 'RSS3'
print eval3
print ""

## Compute RSS for the testing data

print 'TESTING DATA'
print ""

print 'RSS1'
pred1t = model_1.predict(test_data)
eval1t = model_1.evaluate(test_data)
print eval1t
print""

print 'RSS2'
pred2t = model_2.predict(test_data)
eval2t = model_2.evaluate(test_data)
print eval2t
print ""

print 'RSS3'
pred3t = model_3.predict(test_data)
eval3t = model_3.evaluate(test_data)
print eval3t
print ""
