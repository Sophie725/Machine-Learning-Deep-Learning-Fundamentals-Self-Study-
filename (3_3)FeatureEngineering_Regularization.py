# Multiple Regression: Linear regression of multiple features
# Feature Engineering: Make a new feature employing the given features
## Prepare Data
import pandas as pd # pd = data analysis library
### dataframe = core data frame of pd: can deal with multiple array of data
### commonly using csv file to make it as pd df
df = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/perch_full.csv')
perch_full = df.to_numpy()
print(perch_full)

import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

### Divide perch_full and perch_weight into training and testing sets
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42)

## Transformer of Sklearn: Sklearn provides various classes to make features and preprocessing
### Model class of Sklearn: fit(), score(), predict()
### Transformer class of Sklearn: fit(), transform()
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias = False)
### fit() finds a new feature to be made
"""
poly.fit([[2, 3]]) #using 1 sample consisting of Feature2 and Feature3
### Transform() actually transforms the data, accompanying by fit()
print(poly.transform([[2, 3]]))
"""

### Apply to the real data
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

### Check how a number of different features are made
print(poly.get_feature_names_out())

### Transform the data using the transformer for training set: NO poly.fit(test_input)
test_poly = poly.transform(test_input)
print(test_poly.shape)
print(poly.get_feature_names_out())

## Train Multiple Regression Model
from sklearn.linear_model import LinearRegression
### Use previously made train_poly to train the model
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

### Add more features to test the regression
### May be inappropriate causing overfitting
poly = PolynomialFeatures(degree = 5, include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
### Here, testing 42 samples using 55 features is improper
print(lr.score(test_poly, test_target))

## Regularization
### Prevent the model to be overfit to the training set
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
print(train_scaled.shape)
print(test_scaled.shape)

## Ridge Regression: 1st method of preventing from overfitting
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

### Changing alpha hyperparameter for manipulating the extent of regularization
### The bigger the alpha, the stronger the regularization; less overfitting
### Hyperparameter = must be assigned by user beforehand
### Best way to find the appropriate alpha: drawing a graph of R^2
### The point where the score of testing and training sets are equal brings the best alpha value
import matplotlib.pyplot as plt
### Make a list to store the output as alpha value will be changed while finding the proper one
train_score = []
test_score = []

### Method to change alpha value from 0.001 to 100 by * 10 and to store in the lists
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # Create a Ridge model
    ridge = Ridge(alpha = alpha)
    # Train the model
    ridge.fit(train_scaled, train_target)
    # Store the training score and testing score
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

### Drawing graphs based on the given scores
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
### The closest score occurs when alpha = 10^(-1)

### Draw the graph based on the calculated alpha score
ridge = Ridge(alpha = 0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

## Lasso Regression: 2nd method of preventing from overfitting
### Similar to Ridge model
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha = alpha, max_iter = 10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

### For ridge or lasso model training, preprocessing is necessary. But for some other models, 
### preprocessing might not be necessary.

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

### np.sum() gives True as 1, False as 0: 40 features out of 55 features gives 1 
### Therefore, it uses 15 features out of 55 features as total
print(np.sum(lasso.coef_ == 0))