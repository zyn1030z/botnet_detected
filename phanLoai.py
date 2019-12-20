import scapy
from FeatureExtractor import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd



# [6, 17, 99, 2, 189, 1, 113, 177, 184, 147]
# [6, 17, 1, 99, 70, 2]


ip="210.89.164.90"
pcap_path="iot_intrusion_dataset/mirai-udpflooding-3-dec.pcap"
limit=np.inf
model="model100/RandomForestClassifier[86,85,93,92,107,100,82,43,17,35,36,4,16,42,30]-acc=0.99995456.pkl"

#============================================


pro_6 = 0
pro_17 = 0 
# pro_99 = 0
pro_2 = 0
# pro_189 = 0
pro_1 = 0
# pro_113 = 0
# pro_117 = 0

pro_6_botnet = 0
pro_17_botnet = 0 
# pro_99_botnet = 0
pro_2_botnet = 0
# pro_189_botnet = 0
pro_1_botnet = 0
# pro_113_botnet = 0
# pro_117_botnet = 0

IPlayer=0
IPlayer_botnet=0


#============================================


print("[+] Load model...")
clt = pickle.load(open(model,'rb'))


print("[+] Reading pcap file...")
pcap_pack=rdpcap(pcap_path)
fe =FE(pcap_path,limit)
print(len(pcap_pack))
i=0

for p in pcap_pack :
	#xu li data
	i+=1
	print (i)
	data115 = fe.get_next_vector()
	if len(data115) == 115: 
		data=[]
		data.append(data115)
		X_test = pd.DataFrame(data).astype(np.float64)
		# X_test.dropna(inplace=True)
		# print("[-] Drop columns")
		# print(X_test)
		cols = [86,85,93,92,107,100,82,43,17,35,36,4,16,42,30] #thêm index cột muốn xóa
		X_test.drop(X_test.columns[cols],axis=1,inplace=True)
		#print(cols)
		if p.haslayer(IP):
			if p[IP].dst == ip:
				IPlayer+=1
				Y = clt.predict(X_test)
				y = Y[0]
				print(y)
				if y == "botnet":
					IPlayer_botnet+=1		
				if p[IP].proto == 6: #TCP
					pro_6+=1
					if y == "botnet":
						pro_6_botnet+=1
				elif p[IP].proto == 17: #UDP
					pro_17+=1
					if y == "botnet":
						pro_17_botnet+=1
				elif p[IP].proto == 2: #IGMP
					pro_2+=1
					if y == "botnet":
						pro_2_botnet+=1
				elif p[IP].proto == 1: #ICMP
					pro_1+=1
					if y == "botnet":
						pro_1_botnet+=1

ret=[]
ret.append(pro_6_botnet/IPlayer)
ret.append(pro_6/IPlayer)
ret.append(pro_17_botnet/IPlayer)
ret.append(pro_17/IPlayer)
ret.append(pro_2_botnet/IPlayer)
ret.append(pro_2/IPlayer)
ret.append(pro_1_botnet/IPlayer)
ret.append(pro_1/IPlayer)
ret.append(IPlayer_botnet/IPlayer)


# print("pro_6_botnet,pro_6,,pro_17pro_17_botnet,pro_2_botnet,pro_2,pro_1_botnet,pro_1,IPlayer_botnet,IPlayer")
print(ret)





