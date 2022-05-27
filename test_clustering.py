import numpy as np
import pandas as pd
from scapy.all import *
from sklearn.cluster import KMeans
np.random.seed(123)

# data = pd.read_csv("LANL_test.csv") # names=['time', 'duration', 'src_PC', 'src_port', 'dst_PC', 'dst_port', 'protocol', 'packet_count', 'byte_count'])

scapy_cap = rdpcap("smallFlows.pcap")
