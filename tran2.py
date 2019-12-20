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

def cm2pr_binary(cm):
    p =float(cm[0,0])/np.sum(cm[:,0])
    r = float(cm[0,0])/np.sum(cm[0])
    return (p, r)

def machine(dataset):
	#=======================================================
	print("[-] Processing data...")
	num_bot=0
	num_nor=0
	data = []
	# X = []
	# Y = []
	for file in os.listdir(dataset):
			print("\t[-] Readding: ",os.path.join(dataset,file))
			f = open(os.path.join(dataset,file),'r')
			r = csv.reader(f,delimiter = ',')
			index = 0
			for row in r:
				if index > 0:
					tmp = row
					if len(row) == 116: #loai bo cac ban ghi ko du thuoc tính
						data.append(tmp[0:len(tmp)])
					# X.append(tmp[1:len(tmp)-1])
					# Y.append(tmp[len(tmp)-1])
				index += 1
			f.close()
	print("\t[-] Readding DONE!")
	random.shuffle(data)
	X = []
	Y = []
	index = 0
	for row in data:
		if index > 0:
			tmp = row
			X.append(tmp[0:len(tmp)-1])
			lb=tmp[len(tmp)-1]
			if lb == "botnet":
				num_bot += 1
			if lb == "normal":
				num_nor += 1
			Y.append(lb)

		index += 1
	print("[-] Number sample: ",len(X))
	print("[-] Number botnet: ",num_bot)
	print("[-] Number normal: ",num_nor)
	print("[-] Number features: ", len(X[0]))

	# x = np.array(X).astype(np.float64)
	# y = np.array(Y)

	
	# print("load to np")

	x=pd.DataFrame(X).fillna(0)
	x=pd.DataFrame(X).astype(np.float64)
	print("[-] Drop columns")
	cols = [92,93] #them so cot muon xóa
	x.drop(x.columns[cols],axis=1,inplace=True)
	print(cols)
	y=pd.DataFrame(Y)
	y=y.stack()
	#chuyên thuôc tính
	# le = MultiColumnLabelEncoder()
	# x=le.fit_transform(x)
	# y=le.fit_transform(y)
	print("=================================Some_data====================================")
	print(x.iloc[9123:9135,:])
	# print(y.iloc[0:10,:])

	print("==============================================================================")
	X_train, X_test ,y_train, y_test = train_test_split(x,y,test_size = 0.2)
	print("[-] Data ready...")


	print("[-] Start trainning...")
	#========================================================================#
	print("\n=======================Running RandomForestClassifier=======================")
	print("{'n_estimators': 250, 'max_features': 'auto', 'max_depth': 50}")
	forest = RandomForestClassifier(n_estimators=250,max_features='auto',max_depth=50)
	forest.fit(X_train,y_train)
	pickle.dump(forest, open('RandomForestClassifier.pkl', 'wb'))	#nhớ đổi tên model trước khi chạy model mới
	y_pred = forest.predict(X_test)
	print ('Accuracy : ', accuracy_score(y_pred,y_test))
	mt = confusion_matrix(y_pred,y_test)
	p,r = cm2pr_binary(mt)
	print("Precision = {0:.2f}, recall = {1:.2f}".format(p, r))
	print('F1 score = {0:.2f}\n'.format(2*(p*r)/(p+r)))
	print(mt)


	print("============================Feature ranking==================================")
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X_train.shape[1]):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X_train.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(X_train.shape[1]), indices)
	plt.xlim([-1, X_train.shape[1]])
	plt.show()
	print("===========================================================================")
	print("[+] DONE!")
	return 0
	


if __name__ == '__main__':
	if  (len(sys.argv) == 2):
		print("=============================Running==================================")
		machine(sys.argv[1])
	else:
		print("Sai cu phap: py train_model.py 'path_to_data'")
	finish=time.time()
