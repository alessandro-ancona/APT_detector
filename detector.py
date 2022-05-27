import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from APTDetector import APTDetector
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
# APT DETECTOR ---------------------------------------------------------------------------------------------------------

random.seed(124)
np.random.seed(123)
tf.random.set_seed(123)

DS = pd.read_csv("training.csv")
dataset = np.array(DS)
# dataset = dataset[dataset[:, 1].argsort()]
# dataset = dataset[4500:, :]
np.random.shuffle(dataset)
dataset = dataset[dataset[:, 1].argsort()]
# dataset[0, 0] = 2
columns = list(['alert_type', 'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'infected_host', 'scanned_host',
                'alert_type_2', 'timestamp_2', 'src_ip_2', 'src_port_2', 'dst_ip_2', 'dst_port_2', 'infected_host_2',
                'scanned_host_2', 'score', 'APT'])

# problema: primo cluster creato, per alert 4,5,6 crea problemi.

clusters = APTDetector(dataset).cluster()

labelled_dat = APTDetector(dataset).score(clusters)

dataframe = pd.DataFrame(labelled_dat, columns=columns) #.drop(['scanned_host', 'scanned_host_2'], axis=1)

data = np.array(dataframe)
np.random.shuffle(data)
size = data.shape[1] - 1
leng = data.shape[0]

# plt.figure(figsize=(15, 6))
# sns.heatmap(dataframe.corr(), annot=True)
# plt.show()

test_set = round(leng * 15 / 100)

train_examples = data[test_set:, [6, 14, size-1]]
train_labels = data[test_set:, size]

test_examples = data[:test_set, [6, 14, size-1]]
test_labels = data[:test_set, size]

test_labels = test_labels.astype('float32')
train_labels = train_labels.astype('float32')

# train_examples = np.expand_dims(train_examples, 1)
# test_examples = np.expand_dims(test_examples, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_dim=3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),

])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalsePositives(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.AUC()])

n_epochs = 15

history = model.fit(train_examples, train_labels,
                    batch_size=1,
                    epochs=n_epochs,
                    shuffle=True,
                    validation_split=0.2)

test_acc = model.evaluate(test_examples, test_labels, verbose=0)
print(test_acc)
print("done")

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

a = model.predict(test_examples)
b = test_labels
print("hi")
