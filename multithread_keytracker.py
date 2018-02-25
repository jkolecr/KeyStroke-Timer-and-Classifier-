#!/usr/bin/env python2

from pynput import keyboard
import time
import sys
import csv
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


buff = ''
downbuff = []
upbuff = []
password = ';;92=distance=common=catch=45;;'
deltadown = []
deltaup = []
filename = 'real_test.csv'
start = time.time()
first = 0
timedownbuff = []
timeupbuff = []
user = 'kole'
k = 2
h = .02

def save_output():
	i = 0;
	global deltaup, deltadown, upbuff, downbuff, buff, timedownbuff, timeupbuff
	text = ''
	try:
		with open(filename, 'r') as csvfile:
			mycsv = csv.reader(csvfile)
			for row in mycsv:
				text = row[0]
	except:
		print 'creating file'
	with open(filename, 'a') as csvfile:
		fieldnames = ['char', 'deltaup','deltadown','downtime','uptime', 'char_index','user']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		print "col:" + str(text.split(',')[0])
		if str(text.split(',')[0]) == '':
			writer.writeheader()

		for down,up in zip(deltadown,deltaup):
			writer.writerow({'char': ''.join(buff[i]), 'deltaup': ''.join(str(up)), 'deltadown': ''.join(str(down)), 'uptime': ''.join(str(timeupbuff[i])), 'downtime': ''.join(str(timedownbuff[i])), 'char_index': str(i), 'user': user})
			i += 1
		print 'done writeing'

def calculate_delta():
	global deltaup
	global deltadown
	global upbuff
	global downbuff
	i = 0
	deltaup.append(0)
	while(i < len(downbuff)):
		deltadown.append(upbuff[i] - downbuff[i])
		i += 1
	i = 1
	while(i < len(downbuff)):
		deltaup.append(upbuff[i] - downbuff[i - 1])
		i += 1

def on_key_release(key):
	global password
	global upbuff
	global buff
	global downbuff
	global deltadown
	global deltaup
	global timeupbuff

	key = str(key).replace('u\'','',1).replace('\'','')
	if str(key) != 'Key.enter':
		upbuff.append(time.time())
		buff += key
		timeupbuff.append(time.time() - start)
	if buff == password:
		calculate_delta()
		save_output()
#		print 'deltadown:'
#		print ''.join(str(deltadown))
#		print 'deltaup'
#		print ''.join(str(deltaup))
#		print 'downbuff'
#		print ''.join(str(downbuff))
#		print 'upbuff'
#		print ''.join(str(upbuff))
		sys.exit(0)

def on_key_press(key):
	global buff
	global upbuff
	global start_time
	global downbuff
	global deltadown
	global deltaup
	global start
	global first
	global timedownbuff

	key = str(key).replace('u\'','',1).replace('\'','')
	if(key == "Key.esc"):
		calculate_delta()
#		save_output()
		print ''.join(str(deltadown))
		print ''.join(str(deltaup))
		print(buff)
		sys.exit(0)

	if(first == 0):
		first = 1
		start = time.time()
		timedownbuff.append(0)
	else:
		timedownbuff.append(time.time() - start)
	downbuff.append(time.time())

	#buff += key


with keyboard.Listener(on_release = on_key_release, on_press = on_key_press) as listener:
	listener.join()
