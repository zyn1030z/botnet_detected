import numpy as np
import time
from FeatureExtractor import *


f =open("data/_tcp.csv","a")
path = "pcap/Network_traffic_capture_001.pcap" #sys.argv[1]
limit=500000
print(limit)
fe = FE(path,limit)
data=fe.get_next_vector()

while len(data) != 0:
	data=fe.get_next_vector()
	f.write(",".join(map(lambda x: str(x), data)) + "\n")
f.close()

