import os, hashlib, sys, subprocess, json, time

import numpy as np 
import csv
import random
import pandas as pd
import pickle



def info(dataset):
	#=======================================================
	print("[-] Processing data...")
	botnet=0
	normal=0
	for file in os.listdir(dataset):
			print("\t[-] Readding: ",os.path.join(dataset,file))
			f = open(os.path.join(dataset,file),'r')
			r = csv.reader(f,delimiter = ',')
			index = 0
			for row in r:
				if index > 0:
					tmp = row
					label=tmp[len(tmp)-1]
					if label=="normal":
						normal+=1
					if label =="botnet":
						botnet+=1
				index += 1
			f.close()
	print("Number botnet: ",botnet)
	print("Number normal: ", normal)
	print("\t[-] Readding DONE!")


	
	

if __name__ == '__main__':
	if  (len(sys.argv) == 2):
		print("========================Running============================")
		info(sys.argv[1])
	else:
		print("Sai cu phap: py CTU-13.py 'path_to_data'")