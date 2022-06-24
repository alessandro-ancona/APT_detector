import pandas as pd
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
random.seed(555)
np.random.seed(555)
tf.random.set_seed(555)

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

train_data = np.delete(KDDTrain, 42, 1)
test_data = np.delete(KDDTest, 42, 1)
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

index = 5000

train_data = train_data.astype('float32')
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
test_data = test_data.astype('float32')

train_examples = train_data[:5000, :size]
train_labels = train_data[:5000, size]

B_examples = train_data[5000:10000, :size]
B_labels = train_data[5000:10000, size]

new_dataset = np.concatenate((train_data[12500:15000, :], test_data[2500:5000, :]), axis=0)
C_examples = new_dataset[:5000, :size]
C_labels = new_dataset[:5000, size]

D_examples = test_data[10000:15000, :size]
D_labels = test_data[10000:15000, size]

encoder = LabelEncoder()
encoder.fit(train_labels)
encoded_Y = encoder.transform(train_labels)
train_categorical_labels = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(B_labels)
encoded_Y = encoder.transform(B_labels)
B_categorical_labels = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(C_labels)
encoded_Y = encoder.transform(C_labels)
C_categorical_labels = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(D_labels)
encoded_Y = encoder.transform(D_labels)
D_categorical_labels = np_utils.to_categorical(encoded_Y)

train_examples = np.expand_dims(train_examples, 1)
B_examples = np.expand_dims(B_examples, 1)
C_examples = np.expand_dims(C_examples, 1)
D_examples = np.expand_dims(D_examples, 1)

neurons1 = 39
neurons2 = 26

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.FalsePositives(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.AUC()])

print("Training for neurons1: " + str(neurons1) + ", neurons2: " + str(neurons2))
n_epochs = 50
history = model.fit(train_examples, train_categorical_labels,
                    batch_size=512,
                    epochs=n_epochs,
                    shuffle=False,
                    validation_split=0.2)

                # y_pred_keras = model.predict(test_examples).ravel()
                # fpr_keras, tpr_keras, threashold_keras = roc_curve(test_labels, y_pred_keras)

"""
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(np.arange(n_epochs), history.history['loss'], '-b',
         np.arange(n_epochs), history.history['val_loss'], '-r',
         linewidth="0.8")
plt.xlabel("\nN° Epochs", fontsize="medium")
plt.ylabel("Loss\n", fontsize="medium")
plt.legend(('Training Loss', 'Validation Loss'), fontsize="large", loc='upper right', shadow=True)
plt.grid(color='gray', linestyle='--', linewidth=0.2)
plt.show()

y_pred = model.predict(test_examples)
y_pred = np.argmax(y_pred, axis=1)
y_labels = np.argmax(test_categorical_labels, axis=1)

report = classification_report(test_labels, y_pred,
                               target_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
                               digits=4,
                               zero_division=1,
                               output_dict=True)
supports = [report['Normal']['support'],
            report['DoS']['support'],
            report['Probe']['support'],
            report['R2L']['support'],
            report['U2R']['support']]
cf_matrix = confusion_matrix(y_labels, y_pred)
plt.figure(figsize=(6, 4), dpi=300)
ax = sns.heatmap((cf_matrix.T/supports).T, annot=True, cmap='Blues', fmt='.2%') # annot_kws={"fontsize": "large"}
ax.set_xlabel('\nPredicted Values', fontsize="xx-large")
ax.set_ylabel('Actual Values\n', fontsize="xx-large");

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])
ax.yaxis.set_ticklabels(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])

## Display the visualization of the Confusion Matrix.
plt.show()
print(report)
"""
training_acc = model.evaluate(train_examples, train_categorical_labels, verbose=0)
B_acc = model.evaluate(B_examples, B_categorical_labels, verbose=0)
C_acc = model.evaluate(C_examples, C_categorical_labels, verbose=0)
D_acc = model.evaluate(D_examples, D_categorical_labels, verbose=0)

print(str(neurons1) + " " + str(neurons2) + ": TRAINING_LOSS = " + str(training_acc) + " B_PERFORMANCE = " + str(B_acc) + " C_PERFORMANCE = " + str(C_acc) + " D_PERFORMANCE = " + str(D_acc) + "\n")

