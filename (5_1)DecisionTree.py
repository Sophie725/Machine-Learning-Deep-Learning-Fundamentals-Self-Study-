# Use Decision Tree algorithm to solve another Classification problem

## Classify using Logistic Regression
import pandas as pd
wine = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/wine.csv')

print(wine.head()) # To show the first 5 samples # if class = 0, red wine; if class = 1, white wine

### Useful 2 pd df methods
### (1) info(): verify any missed data types or data values on every row
print(wine.info())
### (2) describe(): give out a brief stats of every column e.g. Max, mini, avg
print(wine.describe())

### Convert pd df into np array. Classify training and testing sets.
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
### Because the data sets have too many samples lets use only 20% of the whole datasets as training sets.
### Otherwise, the default test_size = 0.25, which is 25% for training sets. (= 75% for testing sets)
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)

print(train_input.shape, test_input.shape)

### Preprocessing using Sklearn StandardScalar
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

### Train Logistic Regression model using train_scaled and test_scaled
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)

## Decision Tree: Easy to interpret
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

### Draw a decision tree: Creation of root node to leaf node
### Very complicated model is drawn
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

### Simplified version of model: limit the depth of the tree
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH']) # "filled=True": assign color for each class
plt.show()
### Gini Impurity: parent node/child node; information gain; another method = "criterion='entropy'"

### Tree splitting: define the Max depth of tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

### Draw a new tree
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

### Visualize with newly trained tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

### Main feature to distinguish different sets
print(dt.feature_importances_)