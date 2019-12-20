import os, hashlib, sys, subprocess, json, time
#from __future__ import print_function
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import csv

def addlabel(file,label):
	csv_delim=','
	f = open(file,'r')
	# f2 = open("data_gannhan/"+output+".csv",'a')
	f2 = open("data/_"+file,'a')
	r = csv.reader(f,delimiter = ',')
	index = 0 
	for row in r:
		if index == 0:
			tmp = row
			f2.write(csv_delim.join(map(lambda x: str(x), tmp)) + ",Label\n")
			index += 1
		else:
			tmp = row
			f2.write(csv_delim.join(map(lambda x: str(x), tmp)) + ","+label+"\n")
	f.close()

	
		
if __name__ == '__main__':
	if len(sys.argv) == 3:
			file=sys.argv[1]
			label=sys.argv[2]
			# output=sys.argv[3]
			print("running......")
			addlabel(file,label)
	else:
		print("Sai cú pháp\nUse: py add_label.print 'file_path' 'label' ")