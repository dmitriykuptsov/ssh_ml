from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pandas as pd
import numpy as np
import sys

df = pd.read_csv("ml2.pcap.csv")

#df = df[["proto", "subproto", "bulk0", "bulk1", "bulk2", "bulk3", "client_bulksize_avg", "server_bulksize_avg"]]

train, test = train_test_split(df, test_size=0.5)

train_X = train.drop(["proto", "subproto"], axis=1)

train_Y = train["proto"]

#print(train_Y)

#for row in train_Y:
#    print(row)

test_X = test.drop(["proto", "subproto"], axis=1)
test_Y = test["proto"]

# Neural network classifier

ann = MLPClassifier(solver='sgd', max_iter=500, hidden_layer_sizes=(10, 5))
ann.fit(train_X, train_Y)
pred_y = ann.predict(test_X)
print("Artificial neural network accuracy:")
print(np.sum(pred_y == test_Y) / len(test_Y))


# Random forest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(train_X, train_Y)
pred_y = clf.predict(test_X)

print("Random forset accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))

# Naive Bayes classifier
model = GaussianNB()
model.fit(train_X, train_Y)
pred_y = model.predict(test_X)
print("Naive Bayes accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))

# SVM
clf = svm.SVC()
clf.fit(train_X, train_Y)
pred_y = model.predict(test_X)
print("SVM accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))

# SVM
clf = svm.SVC(kernel='rbf')
clf.fit(train_X, train_Y)
pred_y = model.predict(test_X)
print("SVM accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))
