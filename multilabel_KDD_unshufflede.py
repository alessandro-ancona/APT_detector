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
import time
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


attack_dict = {
    'normal': 0,

    'back': 4,  # DOS
    'land': 4,
    'neptune': 4,
    'pod': 4,
    'smurf': 4,
    'teardrop': 4,
    'mailbomb': 4,
    'apache2': 4,
    'processtable': 4,
    'udpstorm': 4,

    'ipsweep': 1,  # PROBE
    'nmap': 1,
    'portsweep': 1,
    'satan': 1,
    'mscan': 1,
    'saint': 1,

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

    'buffer_overflow': 2,  #U2R
    'loadmodule': 2,
    'perl': 2,
    'rootkit': 2,
    'httptunnel': 2,
    'ps': 2,
    'sqlattack': 2,
    'xterm': 2
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
#np.random.shuffle(KDDTrain)
#np.random.shuffle(KDDTest)

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

train_data = train_data.astype('float32')
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
test_data = test_data.astype('float32')

train_data = np.delete(train_data, np.where(train_data[:, size] == 4), axis=0)
test_data = np.delete(test_data, np.where(test_data[:, size] == 4), axis=0)

train_examples = train_data[:, :size]
train_labels = train_data[:, size]

test_examples = test_data[:, :size]
test_labels = test_data[:, size]

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

L = 256  # 64, 128, 256
neurons1 = 39
neurons = [13, 26, 39]  # 13 26 39
file = open("results_performance_" + str(L) +".txt", "w")

for neurons2 in neurons:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(L),
        tf.keras.layers.Dense(neurons1, activation='relu'),
        tf.keras.layers.Dense(neurons2, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.TruePositives(),
                           tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])

    print("Training for LSTM: " + str(L) + ", neurons1: " + str(neurons1) + ", neurons2: " + str(neurons2))
    n_epochs = 100
    time_callback = TimeHistory()
    history = model.fit(train_examples, train_categorical_labels,
                        batch_size=512,
                        epochs=n_epochs,
                        shuffle=False,
                        validation_split=0.2,
                        callbacks=[time_callback])
    file.write("\nLSTM: " + str(L) + ", " + "NEURONS: " + str(neurons1) + ", " + str(neurons2) + " ***************************************** \n")
    # print("LSTM: " + str(L) + ", " + "NEURONS: " + str(neurons1) + ", " + str(neurons2) + ": " + str(model.evaluate(test_examples, test_categorical_labels, verbose=False)) + "\n")

                    # y_pred_keras = model.predict(test_examples).ravel()
                    # fpr_keras, tpr_keras, threashold_keras = roc_curve(test_labels, y_pred_keras)

    # LEARNING CURVES PLOT ---------------------------------------------------------------------------------------------

    plt.subplots(figsize=(6, 3), dpi=300)
    plt.semilogy(np.arange(n_epochs), history.history['loss'], '-b',
             np.arange(n_epochs), history.history['val_loss'], '-r',
             linewidth="0.8")
    plt.ylim([10**-3, 10])
    plt.xlabel("N° Epochs", fontsize="medium")
    plt.ylabel("Loss", fontsize="medium")
    plt.legend(('Training Loss', 'Validation Loss'), fontsize="medium", loc='upper right', shadow=True)
    plt.grid(color='gray', linestyle='--', linewidth=0.2)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.85)
    plt.savefig("logLC" + str(L) + "_39_" + str(neurons2) + ".eps")
    #plt.show()

    # ACCURACY PLOT ----------------------------------------------------------------------------------------------------

    plt.subplots(figsize=(6, 3), dpi=300)
    plt.plot(np.arange(n_epochs), history.history['categorical_accuracy'], '-b',
                 np.arange(n_epochs), history.history['val_categorical_accuracy'], '-r',
                 linewidth="0.8")
    plt.ylim([0, 1])
    plt.xlabel("N° Epochs", fontsize="medium")
    plt.ylabel("Accuracy", fontsize="medium")
    plt.legend(('Training Accuracy', 'Validation Accuracy'), fontsize="medium", loc='lower right', shadow=True)
    plt.grid(color='gray', linestyle='--', linewidth=0.2)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.85)
    plt.savefig("ACC" + str(L) + "_39_" + str(neurons2) + ".eps")

    # SAVE AND PRINT PERFORMANCE METRICS -------------------------------------------------------------------------------

    print("MEAN EPOCH DURATION: " + str(np.mean(time_callback.times)))
    print("FINAL EPOCH TRAINING ERROR: " + str(history.history['loss'][-1]))
    print("FINAL EPOCH VALIDATION ERROR: " + str(
        history.history['val_loss'][-1]))
    print("FINAL ACCURACY: " + str(history.history['categorical_accuracy'][-1]) + ", FINAL VALIDATION ACCURACY: " + str(history.history["val_categorical_accuracy"][-1]))
    file.write("MEAN EPOCH DURATION: " + str(np.mean(time_callback.times)) + "\n")
    file.write("FINAL EPOCH TRAINING ERROR: " + str(history.history['loss'][-1]) + "\n")
    file.write("FINAL EPOCH VALIDATION ERROR: " + str(history.history['val_loss'][-1]) + "\n")
    file.write("FINAL ACCURACY: " + str(history.history['categorical_accuracy'][-1]) + ", FINAL VALIDATION ACCURACY: " + str(history.history["val_categorical_accuracy"][-1]) + "\n")

    """y_pred = model.predict(test_examples)
    y_pred = np.argmax(y_pred, axis=1)
    y_labels = np.argmax(test_categorical_labels, axis=1)

    report = classification_report(test_labels, y_pred,
                                   target_names=['Normal', 'IC', 'LM', 'C2'],
                                   digits=4,
                                   zero_division=1,
                                   output_dict=True)
    supports = [report['Normal']['support'],
                report['IC']['support'],
                report['LM']['support'],
                report['C2']['support']]
    cf_matrix = confusion_matrix(y_labels, y_pred)
    plt.figure(figsize=(6, 4), dpi=300)
    ax = sns.heatmap((cf_matrix.T/supports).T, annot=True, cmap='Blues', fmt='.2%') # annot_kws={"fontsize": "large"}
    ax.set_xlabel('\nPredicted Values', fontsize="xx-large")
    ax.set_ylabel('Actual Values\n', fontsize="xx-large");

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Normal', 'IC', 'LM', 'C2'])
    ax.yaxis.set_ticklabels(['Normal', 'IC', 'LM', 'C2'])

    ## Display the visualization of the Confusion Matrix.
    #plt.show()
    print(report)
    #print(model.summary())"""

file.close()