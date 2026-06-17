## Prepare data
### Use pd to convert csv file into dataframe, them print out first 5 rows using head()
### Dataframe: 2D table data structure provided by pd; 
### easy to convert into np; well aligned with sklearn
import pandas as pd
fish = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fish.csv')
print(fish.head())

### unique(): to see unique types of fish present in the column 'Species'
print(pd.unique(fish['Species']))

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])

fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

### Preprocessing the data using the class StandardScaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

## Predict the probability using KneighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 3) # Set the # of Kneighbors as 3
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

## Predict probabilities using KN
from sklearn.neighbors import KNeighborsClassifier
### set the number of KNeighbor as 3
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# Multi-class classification: target data including more than 2 classes
### By default, align in the alphabetical order
print(kn.classes_)
print(kn.predict(test_scaled[:5]))

### Testing the first 5 samples
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 4))

### Testing the 4th sample
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

# Logistic Regression: Classifier model similar to linear regression
## Sigmoid function, or Logistic function: always put the value within 0 and 1
### Draw the graph using np. Calculate Sigmoid between -5 and 5 with the interval of 0.1
### Use np.exp() to calculate exponential function
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

## Binary Classification using Logistic Classification: Boolean Indexing
### Simple Example
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
### Real Application
### If the train_target is either Bream or Smelt, it will return True. Otherwise, return False
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
### Gather only the datas of bream and smelt
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

### Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5])) # Predict the species of the first 5 samples
print(lr.predict_proba(train_bream_smelt[:5])) # Probabilty of Negative Class(0) and Positive Class(1)

print(lr.classes_) # Bream < Negative Class, Smelt < Positive Class

print(lr.coef_, lr.intercept_)

### Calculate z value using LogisticRegression model
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

### Convert Decisions array value into probability
from scipy.special import expit
print(expit(decisions))

## Perform multiple classification using LogisticREgression
### LogisticRegression uses repetitive algorithm with its defalut repetition as 100 times using
### max_iter parameter; can increases the # of repetition as 1000 or more or less
### LogisticREgression regulate the coefficient as Ridge, which are so-called 'L2 regulation'
### Instead of alpha, the regulating hyperparameter = C (by default, C = 1)
### The smaller the C, the bigger the regulation (here, using C = 20 to relieve the regulation)
lr = LogisticRegression(C = 20, max_iter = 1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target)) # Neither overfitting or lessfitting

print(lr.predict(test_scaled[:5])) # Testing on the first 5 samples

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 3)) # Probabilities for the first 5 samples # 7 rows (compared to  2 rows for binary classification)
print(lr.classes_) # Show which row matches to which species
### The class with the highest probabilities would be the species class

### "Softmax": convert 7 z values into probabilites
print(lr.coef_.shape, lr.intercept_.shape)

### Use decision_function() to obtain z1~z7 values. Convert into probabilities using softmax()
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals = 2))

### Scipy provides softmax ftn.
from scipy.special import softmax
proba = softmax(decision, axis = 1)
print(np.round(proba, decimals = 3))

