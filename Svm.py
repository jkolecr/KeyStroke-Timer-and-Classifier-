import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
gamma = 1
c = 2
n_neighbors = 3
h = .005
deltaup = []
deltadown = []
point_temp = []
names = []
deltadown_test = []
deltaup_test = []
point_temp_test = []
names_test = []
downtime = []
downtime_test =[]
uptime = []
uptime_test = []
index = []
index_test = []

#X = np.array([[0, .0677], [.2474, .089], [0.3980, .0991]])
y = []
y_test = []


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

def knn_and_plot():
	global deltaup 
	global deltadown
	global point_temp
	global clf
	X = np.array(point_temp)
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
	
	clf.fit(X,y)
	
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	plt.scatter(deltaup, deltadown, c=y, cmap=cmap_bold,edgecolor='k', s=20)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())

	plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, 'uniform'))

	plt.show()


def svm_and_plot():
	global deltaup 
	global deltadown
	global point_temp
	global clf
	X = np.array(point_temp)
	clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
	
	clf.fit(X,y)
	
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	plt.scatter(deltaup, deltadown, c=y, cmap=cmap_bold,edgecolor='k', s=20)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())

	plt.title("2 class svm (gamma = %i, c = '%i')" % (gamma, c))

	plt.show()


def load_file(file_name,user_name):
	df = pd.read_csv(file_name)
	global deltaup 
	global deltadown
	global point_temp
	global names
	global x_min, x_max
	global y_min, y_max
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
	global deltaup 
	global deltadown
	global point_temp
	global downtime, uptime, index
	global names
	global x_min, x_max
	global y_min, y_max
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
	global deltaup_test 
	global deltadown_test
	global point_temp_test
	global names_test
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
	global deltaup_test 
	global deltadown_test
	global point_temp_test
	global names_test
	global downtime_test, uptime_test, index_test
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
	return float(correct) / float(total)

load_file_test_higher_dim("real_test.csv","kole")
load_file_higher_dim("kole_Jinglebells.csv","kole")

best = 0
best_g = 0
best_c = 0
score = 0
best_k = 0
#for i in range(1,100):
#	print "current progress: " + str(i)
#	for j in range(1,100):
#		svm_test(i,j)
#		score = accuracy_score(y_test, clf.predict(point_temp_test))
#		if score > best:
#			best = score 
#			best_g = i
#			best_c = j
#print "best: " + str(best)
#print "best_x: " + str(best_g)		#9
#print "best_y: " + str(best_c)		#47
#gamma = best_g
#c = best_c

gamma = 9
c = 47
#for i in range(1,100):
#	knn(i)
#	score = accuracy_score(y_test, clf.predict(point_temp_test))
#	if score > best:
#			best = score
#			best_k = i

#n_neighbors = best_k
#print best_k
svm_test(gamma, c)		
print accuracy_score(y_test, clf.predict(point_temp_test))

