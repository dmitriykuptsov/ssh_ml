from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import pandas as pd
import numpy as np
import sys

def parse_dataset(file):
	data = []
	c = 0
	with open(file) as f:
		lines = f.readlines();
		for line in lines:
			if c == 0:
				c += 1
				continue
			c += 1
			line = line.strip()
			cols = line.split(",")
			row = []
			"""
			if cols[1] == "NaN" or cols[1] == "Infinity":
				continue
			if cols[14] == "NaN" or cols[14] == "Infinity":
				continue
			if cols[15] == "NaN" or cols[15] == "Infinity":
				continue
			if cols[43] == "NaN" or cols[43] == "Infinity":
				continue
			if cols[44] == "NaN" or cols[44] == "Infinity":
				continue
			if cols[45] == "NaN" or cols[45] == "Infinity":
				continue
			if cols[46] == "NaN" or cols[46] == "Infinity":
				continue
			if cols[47] == "NaN" or cols[47] == "Infinity":
				continue
			if cols[70] == "NaN" or cols[70] == "Infinity":
				continue
			if cols[74] == "NaN" or cols[74] == "Infinity":
				continue
			"""
			#row.append(float(cols[5].strip()))
			row.append(float(cols[6].strip()))
			row.append(float(cols[7].strip()))
			row.append(float(cols[8].strip()))
			row.append(float(cols[9].strip()))
			#row.append(float(cols[10].strip()))
			#row.append(float(cols[11].strip()))
			#IPV4_SRC_ADDR 0,L4_SRC_PORT 1,IPV4_DST_ADDR 2,L4_DST_PORT 3,PROTOCOL 4,L7_PROTO 5,IN_BYTES 6,OUT_BYTES 7,IN_PKTS 8,OUT_PKTS 9,TCP_FLAGS 10,FLOW_DURATION_MILLISECONDS 11,Label,Attack
			if cols[13].strip() == "Benign":
				row.append(0)
			elif cols[13].strip() in ["DDOS attack-HOIC", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DDoS attacks-LOIC-HTTP", "DDOS attack-LOIC-UDP", "DoS attacks-GoldenEye", "DoS attacks-Slowloris"]:
				row.append(1)
			else:
				continue
			data.append(row)
	return data

def get_benign_samples(data, size):
	benign = []
	for row in data:
		if row[5] == 0:
			benign.append(row)
	#rows = np.random.choice(len(benign), size)
	rows = benign[:size]
	c = 0
	samples = []
	for i in rows:
		samples.append(benign[c])
		c += 1
	return samples

def get_ddos_samples(data, size):
	ddos = []
	c = 0
	for row in data:
		if row[5] == 1:
			ddos.append(row)
			if c > size:
				break
			c += 1
	return ddos

def remove_label(data):
	r_data = []
	for row in data:
		r_data.append(row[:-1])
	return r_data

def get_ys(data):
	y = []
	for row in data:
		if row[-1] == 0:
			y.append([1, 0])
		else:
			y.append([0, 1])
	return y

def array_to_dataframe(a):
	np_array = None
	for row in a:
		if np_array is None:
			np_array = np.array([row])
		else:
			np_array = np.append(np_array, np.array([row]), axis=0)
	return np_array

data = parse_dataset("NF-CSE-CIC-IDS2018.csv")

benign = get_benign_samples(data, 10)
ddos = get_ddos_samples(data, 10)

split_point_b = int(len(benign) * 0.5)
split_point_p = int(len(ddos) * 0.5)

train_benign = benign[:split_point_b]
train_ddos = ddos[:split_point_p]

test_benign = benign[split_point_b:]
test_ddos = ddos[split_point_p:]

train_y_b = get_ys(train_benign)
train_y_p = get_ys(train_ddos)

train_x_b = remove_label(train_benign)
train_x_p = remove_label(train_ddos)

test_y_b = get_ys(test_benign)
test_y_p = get_ys(test_ddos)

test_x_b = remove_label(test_benign)
test_x_p = remove_label(test_ddos)

train_y_b = array_to_dataframe(train_y_b)
train_y_p = array_to_dataframe(train_y_p)

test_y_b = array_to_dataframe(test_y_b)
test_y_p = array_to_dataframe(test_y_p)

train_x_b = array_to_dataframe(train_x_b)
train_x_p = array_to_dataframe(train_x_p)

test_x_b = array_to_dataframe(test_x_b)
test_x_p = array_to_dataframe(test_x_p)

train_x = np.concatenate((train_x_b, train_x_p))
train_y = np.concatenate((train_y_b, train_y_p))

test_x = np.concatenate((test_x_b, test_x_p))
test_y = np.concatenate((test_y_b, test_y_p))

clf = MLPClassifier(solver='sgd', max_iter=500, hidden_layer_sizes=(20, 4))

# Classifies correctly
#train_x = np.array([[1, 2], [2, 1], [1, 1.5], [1.5, 2], [4, 5], [6,4]])
#train_y = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])

print(train_x)
clf.fit(train_x, train_y)
for row in train_x:
	print(row)
#for y in train_y:
#	print(y)
pred = clf.predict_proba(train_x)
#for row in test_x:
#	print(row)
for p in pred:
	print(np.argmax(p))
#	print(p)
#print(train_x)
