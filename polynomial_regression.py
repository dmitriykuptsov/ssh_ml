from random import uniform
from math import sin
from numpy import arange
from numpy import dot
from matplotlib import pyplot as plt
import numpy as np
from math import *

def generate_sequence(n, step=1):
	for i in arange(0, n, step):
		yield sin(i)

gen = generate_sequence(10, 0.1)
seq = []
for elem in gen:
	elem += uniform(-3, 3)
	seq.append(elem)

plt.plot(arange(0, 10, 0.1), seq)
plt.show()

x = arange(0, 10, 0.1)
x = np.array(x)

bias = np.repeat(1, int(10/0.1))
x1 = np.vstack((x, x*x))
x2 = np.vstack((x1, x*x*x))
x3 = np.vstack((x2, x*x*x*x))
x4 = np.vstack((x3, bias))
x = np.transpose(x4)

y = seq

theta = [0.7, 1, 10, 2, 1]


def cost_function(theta, x, y):
	h = np.dot(x, theta)
	n = np.shape(h)[0]
	d = np.transpose(y) - h
	return np.dot(np.transpose(d), d)/(2*n)

print(cost_function(theta, x, y))

def d_cost_function(theta, x, y):
	h = np.dot(x, theta)
	n = np.shape(h)[0]
	d = np.transpose(y) - h
	return -1*np.dot(np.transpose(d), x)/n

print(d_cost_function(theta, x, y))

epsilon = 1000000
iter = 0

learning_rate = 0.00001

while abs(epsilon) > 0.0001:
	c1 = cost_function(theta, x, y)
	theta = theta - learning_rate * d_cost_function(theta, x, y)
	print(theta)
	c2 = cost_function(theta, x, y)
	epsilon = c1 - c2
	print(epsilon)
	iter += 1
	print("-----")


#print(theta)
#print(np.dot(np.transpose(theta), [[0, 1], [10, 1]]))

plt.plot(arange(0, 10, 0.1), seq)
#plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.dot(np.transpose(theta), [[0, 10], [1, 1]]))
plt.show()

