from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="features file")
args = parser.parse_args()
df = pd.read_csv(args.file)

#df = df[["proto", "subproto", "bulk0", "bulk1", "bulk2", "bulk3", "client_bulksize_avg", "server_bulksize_avg"]]

train, test = train_test_split(df, test_size=0.3)
train_X = train.drop(["proto", "subproto"], axis=1)
train_Y = train["proto"]

test_X = test.drop(["proto", "subproto"], axis=1)
test_Y = test["proto"]

# Neural network classifier

ann = MLPClassifier(solver='sgd', hidden_layer_sizes=(10, 5))
ann.fit(train_X, train_Y)
pred_y = ann.predict(test_X)
print("Artificial neural network accuracy:")
print(np.sum(pred_y == test_Y) / len(test_Y))
print('Precision: %.3f' % precision_score(test_Y, pred_y, average='micro'))
print("Recall: %.3f" % recall_score(test_Y, pred_y, average='micro'))

# Random forest
clf = RandomForestClassifier(max_depth=5)
clf.fit(train_X, train_Y)
pred_y = clf.predict(test_X)
print("Random forset accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))
print('Precision: %.3f' % precision_score(test_Y, pred_y, average='micro'))
print("Recall: %.3f" % recall_score(test_Y, pred_y, average='micro'))

# Naive Bayes classifier
model = GaussianNB()
model.fit(train_X, train_Y)
pred_y = model.predict(test_X)
print("Naive Bayes accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))
print('Precision: %.3f' % precision_score(test_Y, pred_y, average='micro'))
print("Recall: %.3f" % recall_score(test_Y, pred_y, average='micro'))

# SVM
clf = svm.SVC()
clf.fit(train_X, train_Y)
pred_y = clf.predict(test_X)
print("SVM accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))
print('Precision: %.3f' % precision_score(test_Y, pred_y, average='micro'))
print("Recall: %.3f" % recall_score(test_Y, pred_y, average='micro'))
# SVM
clf = svm.SVC(kernel="rbf")
clf.fit(train_X, train_Y)
pred_y = clf.predict(test_X)
print("SVM accuracy:")
print(np.sum(pred_y == test_Y)/len(test_Y))
print('Precision: %.3f' % precision_score(test_Y, pred_y, average='micro'))
print("Recall: %.3f" % recall_score(test_Y, pred_y, average='micro'))
