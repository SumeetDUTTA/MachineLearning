from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(int)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[2.6]]))
print(example)

# Using matplotlib to plot the visualization
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)
print(y_prob)
plt.plot(x_new, y_prob[:, 1], 'g-', label='Iris-Virginica')
plt.show()

# print(y)
# print(x)
