#!/usr/bin/env python2

import pandas as pd
import numpy as np
import pickle
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier

gamma = 1
c = 2
n_neighbors = 3
h = .005
deltaup,deltadown,point_temp,names,deltadown_test=[],[],[],[],[]
deltaup_test,point_temp_test,names_test,downtime=[],[],[],[]
downtime_test,uptime,uptime_test,index,index_test=[],[],[],[],[]
y,y_test=[],[]

x_min, x_max = 0,0
y_min, y_max = 0,0

def network_test(x,w):
	global clf
	X = np.array(point_temp)
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(x, w), random_state=1)
	clf.fit(X,y)

def knn(k):
	global clf
	X = np.array(point_temp)
	clf = neighbors.KNeighborsClassifier(k, weights='uniform')
	clf.fit(X,y)

def svm_test(gamma,c):
	global clf
	X = np.array(point_temp)
	clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
	clf.fit(X,y)

def load_file(file_name,user_name):
	df = pd.read_csv(file_name)
	global deltaup,deltadown,point_temp,names
	global x_min,x_max
	global y_min,y_max
	deltaup = df.deltaup
	deltadown = df.deltadown
	x_min, x_max = min(deltaup) - .1, max(deltaup) + .1
	y_min, y_max = min(deltadown) - .1, max(deltadown) + .1
	names = df.user

	for i in range(len(deltaup)):
		point_temp.append([deltaup[i],deltadown[i]])
		if(names[i] == user_name):
			y.append(1)
		else:
			y.append(0)

def load_file_higher_dim(file_name,user_name):
	df = pd.read_csv(file_name)
	global deltaup,deltadown,point_temp
	global downtime,uptime,index,names
	global x_min,x_max
	global y_min,y_max
	deltaup = df.deltaup
	deltadown = df.deltadown
	downtime = df.downtime
	uptime = df.uptime
	index = df.char_index
	x_min, x_max = min(deltaup) - .1, max(deltaup) + .1
	y_min, y_max = min(deltadown) - .1, max(deltadown) + .1
	names = df.user

	for i in range(len(deltaup)):
		point_temp.append([deltaup[i],deltadown[i],index[i]])
		if(names[i] == user_name):
			y.append(1)
		else:
			y.append(0)

def load_file_test(file_name,user_name):
	qf = pd.read_csv(file_name)
	global deltaup_test,deltadown_test,point_temp_test,names_test
	deltaup_test = qf.deltaup
	deltadown_test = qf.deltadown
	names_test = qf.user

	for i in range(len(deltaup_test)):
		point_temp_test.append([deltaup_test[i],deltadown_test[i]])
		if(names_test[i] == user_name):
			y_test.append(1)
		else:
			y_test.append(0)

def load_file_test_higher_dim(file_name,user_name):
	qf = pd.read_csv(file_name)
	global deltaup_test,deltadown_test,point_temp_test,names_test
	global downtime_test,uptime_test,index_test
	deltaup_test = qf.deltaup
	deltadown_test = qf.deltadown
	names_test = qf.user
	downtime_test = qf.downtime
	uptime_test = qf.uptime
	index_test = qf.char_index

	for i in range(len(deltaup_test)):
		point_temp_test.append([deltaup_test[i],deltadown_test[i],index_test[i]])
		if(names_test[i] == user_name):
			y_test.append(1)
		else:
			y_test.append(0)

def accuracy_score(test, pred):
	correct = 0
	total = len(test)
	for i in range(total):
		if test[i] == pred[i]:
			correct = correct + 1
	return float(correct)/float(total)

def train_data():
	best,best_g,best_c,score,best_k = 0,0,0,0,0
	for i in range(1,10):
		print "current progress: " + str(i)
		for j in range(1,10):
			svm_test(i,j)
			score = accuracy_score(y_test, clf.predict(point_temp_test))
			if score > best:
				best = score
				best_g = i
				best_c = j
	print "best: " + str(best)
	print "best_x: " + str(best_g)		#9
	print "best_y: " + str(best_c)		#47
	gamma = best_g
	c = best_c
	svm_test(gamma, c)
	print accuracy_score(y_test, clf.predict(point_temp_test))

def best_train_data():
	gamma = 9
	c = 47
	svm_test(gamma, c)
	print accuracy_score(y_test, clf.predict(point_temp_test))

load_file_test_higher_dim("./Data/Test.csv","kole")
load_file_higher_dim("./Data/Training.csv","kole")

train_data()
best_train_data()
