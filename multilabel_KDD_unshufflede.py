import keras.callbacks
import pandas as pd
import tensorflow as tf
import numpy as np
import os

from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
import seaborn as sns
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

attack_dict = {
    'normal': 0,

    'back': 1,  # DOS
    'land': 1,
    'neptune': 1,
    'pod': 1,
    'smurf': 1,
    'teardrop': 1,
    'mailbomb': 1,
    'apache2': 1,
    'processtable': 1,
    'udpstorm': 1,

    'ipsweep': 2,  # PROBE
    'nmap': 2,
    'portsweep': 2,
    'satan': 2,
    'mscan': 2,
    'saint': 2,

    'ftp_write': 3,  # R2L
    'guess_passwd': 3,
    'imap': 3,
    'multihop': 3,
    'phf': 3,
    'spy': 3,
    'warezclient': 3,
    'warezmaster': 3,
    'sendmail': 3,
    'named': 3,
    'snmpgetattack': 3,
    'snmpguess': 3,
    'xlock': 3,
    'xsnoop': 3,
    'worm': 3,

    'buffer_overflow': 4,  #U2R
    'loadmodule': 4,
    'perl': 4,
    'rootkit': 4,
    'httptunnel': 4,
    'ps': 4,
    'sqlattack': 4,
    'xterm': 4
}

service_dict = {
    'aol': 1,
    'auth': 2,
    'bgp': 3,
    'courier': 4,
    'csnet_ns': 5,
    'ctf': 6,
    'daytime': 7,
    'discard': 8,
    'domain': 9,
    'domain_u': 10,
    'echo': 11,
    'eco_i': 12,
    'ecr_i': 13,
    'efs': 14,
    'exec': 15,
    'finger': 16,
    'ftp': 17,
    'ftp_data': 18,
    'gopher': 19,
    'harvest': 20,
    'hostnames': 21,
    'http': 22,
    'http_2784': 23,
    'http_443': 24,
    'http_8001': 25,
    'imap4': 26,
    'IRC': 27,
    'iso_tsap': 28,
    'klogin': 29,
    'kshell': 30,
    'ldap': 31,
    'link': 32,
    'login': 33,
    'mtp': 34,
    'name': 35,
    'netbios_dgm': 36,
    'netbios_ns': 37,
    'netbios_ssn': 38,
    'netstat': 39,
    'nnsp': 40,
    'nntp': 41,
    'ntp_u': 42,
    'other': 43,
    'pm_dump': 44,
    'pop_2': 45,
    'pop_3': 46,
    'printer': 47,
    'private': 48,
    'red_i': 49,
    'remote_job': 50,
    'rje': 51,
    'shell': 52,
    'smtp': 53,
    'sql_net': 54,
    'ssh': 55,
    'sunrpc': 56,
    'supdup': 57,
    'systat': 58,
    'telnet': 59,
    'tftp_u': 60,
    'tim_i': 61,
    'time': 62,
    'urh_i': 63,
    'urp_i': 64,
    'uucp': 65,
    'uucp_path': 66,
    'vmnet': 67,
    'whois': 68,
    'X11': 69,
    'Z39_50': 70

}

flag_dict = {
    'OTH': 1,
    'REJ': 2,
    'RSTO': 3,
    'RSTOS0': 4,
    'RSTR': 5,
    'S0': 6,
    'S1': 7,
    'S2': 8,
    'S3': 9,
    'SF': 10,
    'SH': 11
}

protocol_dict = {
    'tcp': 1,
    'udp': 2,
    'icmp': 3
}
KDDTrain = np.array(pd.read_csv("KDDTrain.csv"))
KDDTest = np.array(pd.read_csv("KDDTest.csv"))

train_data = KDDTrain[:, [0, 1, 2, 3, 5, 4, 28, 29, 33, 32, 34, 11, 22, 24, 37, 25, 38, 35, 41]]
test_data = KDDTest[:, [0, 1, 2, 3, 5, 4, 28, 29, 33, 32, 34, 11, 22, 24, 37, 25, 38, 35, 41]]
np.random.shuffle(KDDTrain)
np.random.shuffle(KDDTest)

size = train_data.shape[1] - 1
leng = train_data.shape[0]

for row in train_data:
    row[size] = attack_dict[row[size]]
    row[2] = service_dict[row[2]]
    row[1] = protocol_dict[row[1]]
    row[3] = flag_dict[row[3]]

for row in test_data:
    row[size] = attack_dict[row[size]]
    row[2] = service_dict[row[2]]
    row[1] = protocol_dict[row[1]]
    row[3] = flag_dict[row[3]]

train_examples = train_data[:, :size]
train_labels = train_data[:, size]

test_examples = test_data[:, :size]
test_labels = test_data[:, size]

test_examples = test_examples.astype('float32')
train_examples = train_examples.astype('float32')

test_labels = test_labels.astype('float32')
train_labels = train_labels.astype('float32')

encoder = LabelEncoder()
encoder.fit(train_labels)
encoded_Y = encoder.transform(train_labels)
train_categorical_labels = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(test_labels)
encoded_Y = encoder.transform(test_labels)
test_categorical_labels = np_utils.to_categorical(encoded_Y)

train_examples = np.expand_dims(train_examples, 1)
test_examples = np.expand_dims(test_examples, 1)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(12, input_dim=size, activation='linear'),
    tf.keras.layers.Dense(8, activation='sigmoid'),
    tf.keras.layers.Dense(5, activation='sigmoid'),

])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.FalsePositives(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.AUC()])

n_epochs = 20
history = model.fit(train_examples, train_categorical_labels,  # [train]
                    batch_size=32,
                    epochs=n_epochs,
                    shuffle=True,
                    validation_split=0.2)


test_acc = model.evaluate(test_examples, test_categorical_labels, verbose=0)
print(test_acc)
y_pred_keras = model.predict(test_examples).ravel()
fpr_keras, tpr_keras, threashold_keras = roc_curve(test_labels, y_pred_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'b--')
plt.plot(fpr_keras, tpr_keras, 'r')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ROC Curve")
plt.grid(color='green', linestyle='--', linewidth=0.2)

plt.figure(2)
plt.plot(np.arange(n_epochs), history.history['loss'], np.arange(n_epochs), history.history['val_loss'])
plt.xlabel("N° Epochs")
plt.ylabel("Loss")
plt.legend(('Training Loss', 'CV Loss'), loc='upper center', shadow=True)
plt.grid(color='green', linestyle='--', linewidth=0.2)
plt.show()