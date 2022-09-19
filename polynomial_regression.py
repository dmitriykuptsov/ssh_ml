from random import uniform
from math import sin
from numpy import arange
from numpy import dot
from matplotlib import pyplot as plt
import numpy as np
from math import *

def generate_sequence(n, step=1):
	for i in arange(-n, n, step):
		yield -0.3*i*i*i*i*i+0.5*i*i*i*i+0.5*i*i*i+1*i*i+0.6*i

gen = generate_sequence(5, 0.1)
seq = []
for elem in gen:
	elem += uniform(-50, 50)
	seq.append(elem)

x = arange(-5, 5, 0.1)
x = np.array(x)
bias = np.repeat(1, int(10/0.1))
x1 = np.vstack((x, x*x))
x2 = np.vstack((x1, x*x*x))
x3 = np.vstack((x2, x*x*x*x))
x4 = np.vstack((x3, x*x*x*x*x))
x5 = np.vstack((x4, x*x*x*x*x*x))
x6 = np.vstack((x5, x*x*x*x*x*x*x))
x7 = np.vstack((x6, bias))
x = np.transpose(x7)
y = seq
theta = [0, 0, 0, 0, 0, 0, 0, 1]

def cost_function(theta, x, y):
	h = np.dot(x, theta)
	n = np.shape(h)[0]
	d = np.transpose(y) - h
	return np.dot(np.transpose(d), d)/(2*n)

def d_cost_function(theta, x, y):
	h = np.dot(x, theta)
	n = np.shape(h)[0]
	d = np.transpose(y) - h
	return -1*np.dot(np.transpose(d), x)/n

print(d_cost_function(theta, x, y))

epsilon = inf
iter = 0

learning_rate = 0.0000000000001

while epsilon > 0.01:
	c1 = cost_function(theta, x, y)
	theta = theta - learning_rate * d_cost_function(theta, x, y)
	print(theta)
	c2 = cost_function(theta, x, y)
	epsilon = c1 - c2
	print(epsilon)
	iter += 1
	print("-----")


x = arange(-5, 5, 0.1)
x = np.array(x)
x0 = arange(-5, 5, 0.1)

bias = np.repeat(1, int(10/0.1))
x1 = np.vstack((x, x*x))
x2 = np.vstack((x1, x*x*x))
x3 = np.vstack((x2, x*x*x*x))
x4 = np.vstack((x3, x*x*x*x*x))
x5 = np.vstack((x4, x*x*x*x*x*x))
x6 = np.vstack((x5, x*x*x*x*x*x*x))
x7 = np.vstack((x6, bias))
x = x7

plt.plot(arange(-5, 5, 0.1), seq, 'r+')
plt.plot(x0, np.dot(np.transpose(theta), x))
plt.show()

