# Understand the GradientDescent algorithm. Train classifier model from the massive datasets.
# Stochastic Gradient Descent = finding the fastest way using one sample out of the whole datasets
# among training sets 
# used for Gradual learning; train model with new data without discarding the previously used data
# Epoch: use one series of the whole datasets in the Stochastic Gradient Descent; can repeat 
# multiple times

# (Standard) Gradient Descent: use one sample to perform Gradient Descent; repeat the step
# multiple times; multiple Epoch
# Minibatch Gradient Descent: use multiple samples to perform GradientDescent
# Batch Gradient Descent: use the whole sample for 1 time descent

# Loss function: the smaller the loss, the better: the starting point for the descent gradient
# ~ Cost function

# Logistic Loss function:
# for Binary classification, (+) class loss is -log, (-) class loss is 1-log << Called Logistic loss ftn, or binary cross-entropy loss ftn
# for Multiple classification << Called Cross-Entropy loss ftn

## SGDClassifier

### Create pd df from csv data file
import pandas as pd
fish = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']] # Use the five property data as input
fish_target = fish[['Species']].to_numpy()

from sklearn.model_selection import train_test_split # Split into testing and training sets
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

### Classifier class provided by Sklearn for Sochastic GradientDescent: SGDClassfier
### SGDClassifier assign 2 parameters when making an object
### Loss assign the type of loss function (here, loss = 'log'. this means choosing binary classification)
### max_iter = # of epoch (here, repetition of 10 times). Then print the accuracy score of
### testing and training sets
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss = 'log_loss', max_iter = 1000, random_state = 42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

### Adding more sc to train. Use partial_fit() to train continuously.
### Similar to fit(), but can continue by 1 epoch
sc.partial_fit(train_scaled, train_target) # Different than batch gradient descent.
print(sc.score(train_scaled, train_target)) # In the programming, the algorithm repeats one epoch
print(sc.score(test_scaled, test_target)) # by one data sample each time.

## Epoch and Overfitting/Underfitting
### When do partial_fit(), the score increases significantly. 
### Set the standard of how many times the training should be repeated.
### Early Stopping: Stop before the model transits from underfitting to overfitting

### To use partial_fit() only. Not fit().
### Convey the whole classes of the training sets into partial_fit().

### Make a list of 7 types of fish using np.unique().
### Prepare 2 new lists to document the score of training and testing sets after testing one epoch
import numpy as np
sc = SGDClassifier(loss = 'log_loss', random_state = 42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0, 300): # Train using 300 epoch
    sc.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

### The Early Stopping point here is when max_iter = 100
sc = SGDClassifier(loss = 'log_loss', max_iter = 100, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# Hinge Loss ~ Support Vector Machine: a loss ftn for another ML algorithm
# The default loss ftn for most ML is 'hinge'
### E.g.
sc = SGDClassifier(loss = 'hinge', max_iter = 100, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)
sc.fit(test_scaled, test_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))