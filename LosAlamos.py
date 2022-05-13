import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dns = pd.read_csv('dns.csv', header=None, names=['timestamp', 'srcPC', 'rsvdPC'])
dns = dns[1:100000]
flows = pd.read_csv('flows.csv', header=None, names=['timestamp', 'duration', 'srcPC', 'srcPort', 'dstPC', 'dstPort',
                                                     'protocol', 'pcktCount', 'byteCount'])
flows = flows[1:100000]
redteam = pd.read_csv('redteam.csv', header=None, names=['timestamp', 'user@domain', 'sourcePC', 'dstPC'])
redteam = redteam[1:100000]
print('done')

