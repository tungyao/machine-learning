import numpy as np
from sklearn import linear_model
from numpy import recfromcsv,loadtxt
import matplotlib.pyplot as plt
origin_source = loadtxt('../data/simple_line.csv',delimiter=',',skiprows=1,dtype=np.int)
X_ = origin_source.tolist()
Y = []
X = []

for i in range(len(X_)):
    X.append([X_[i][0]])
    Y.append(X_[i][1])
model = linear_model.LinearRegression()
model.fit(X,Y)
a = model.predict(200)
print(a)
b = model.predict(X)
plt.scatter(X,Y)
plt.plot(X,b)
plt.show()