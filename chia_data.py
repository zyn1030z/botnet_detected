import os, hashlib, sys, subprocess, json, time
#from __future__ import print_function
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np 
import csv
import random
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt



def machine(dataset):
	#=======================================================
	print("[-] Processing data...")
	num_bot=0
	num_nor=0
	

	for file in os.listdir(dataset):
			print("\t[-] Readding: ",os.path.join(dataset,file))
			f = open(os.path.join(dataset,file),'r')
			f1= open("datachia/"+file,"a")
			r = csv.reader(f,delimiter = ',')
			index = 0
			data = []			
			for row in r:
				if index > 0:
					tmp = row
					data.append(tmp[0:len(tmp)])
				index += 1
			random.shuffle(data)
			i=0
			for row in data:
				if i > round(len(data)/7):#chia so % hien tại đang 50%
					break
				f1.write(",".join(map(lambda x: str(x), row)) + "\n")
				i+=1

			f1.close()	
			f.close()
	print("\t[-] Readding DONE!")


	


if __name__ == '__main__':
	start = time.time()
	if  (len(sys.argv) == 2):
		print("=============================Running==================================")
		machine(sys.argv[1])
	else:
		print("Sai cu phap: py chia_data.py 'path_to_data'")
	finish=time.time()
	print("Time working: %c minus",(finish-start)/60)