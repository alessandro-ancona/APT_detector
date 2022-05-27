import keras.callbacks
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
import seaborn as sns
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)


# class MyCallback(keras.callbacks.Callback):
#     def on_train_begin(self, logs=None):
#         self.data = []
#
#     def on_epoch_end(self, epoch, logs=None):
#         cv_loss = model.evaluate(cv_examples, cv_labels, verbose=0)
#         self.data.append(cv_loss[0])
#         print("CV loss is: " + str(cv_loss[0]))
#
#     def get_data(self):
#         return self.data

# calcolo correlazione
data1 = pd.read_csv('EKDD_Train.csv')
data2 = pd.read_csv('EKDD_Test.csv')
# plt.figure(figsize=(15, 6))
# sns.heatmap(dataset.iloc[:, [4, 2, 5, 3, 28, 29, 33, 32, 34, 11, 22, 24, 37, 25, 38, 35, 41]].corr(), annot=True)
# plt.show()
# [2, 3, 4, 5, 6, 7, 8, 10, 12, 23, 25, 29, 30, 35, 36, 37, 38, 40] selecting optimal subset of features [cite]

data_train = np.array(data1.iloc[:, [0, 1, 4, 2, 5, 3, 28, 29, 33, 32, 34, 11, 22, 24, 37, 25, 38, 35, 41]])
data_test = np.array(data2.iloc[:, [0, 1, 4, 2, 5, 3, 28, 29, 33, 32, 34, 11, 22, 24, 37, 25, 38, 35, 41]])

np.random.shuffle(data_train)
np.random.shuffle(data_test)
# train_labels = np.array(train['Class'])
size = data_train.shape[1] - 1
leng = data_train.shape[0]

train_labels = data_train[:, size]
train_examples = data_train[:, :size]

test_labels = data_test[:, size]
test_examples = data_test[:, :size]
# cv_set = round(leng * 20 / 100)
# indexes = np.array(random.sample(list(np.arange(leng)), cv_set))

# cv_data = red_data[indexes, :]
# train_data = np.delete(red_data, indexes, axis=0)

# cv_examples = cv_data[:, :size]
# cv_labels = cv_data[:, size]

test_labels = test_labels.astype('float32')
train_labels = train_labels.astype('float32')

# train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

# n_fold = 5
# kfold = KFold(n_splits=n_fold, shuffle=True, random_state=123)
# epoch_losses = np.zeros(n_fold)
# cv_losses = np.zeros(n_fold)

# fold_n = 1
# for train, test in kfold.split(train_examples, train_labels):
# a = train
# b = test

# 8 8 su 10 epoche da 90 per cento, su 50 è 79.4 per cento
# 12 12 su 30 epoche -> oltre le 30 epoche la loss converge ed è sul 80% accuracy sul test set
# 24 12 su 50 epoche -> siamo sul 79 per cento accuracy
# 14 12 su 50 epoche -> siamo sul 78 ma è incredibile l accuracy e convergenza sul training set
# 12 8 su 50 epoche -> siamo sul 79,4 ma è sbilanciato sui falsi negativi il test, ottima accuracy sul train

# 24 12 ottimale auc 99 per cento, loss basso e accuracy bassissima, sul test da 79.4 per cento

#train_examples = np.expand_dims(train_examples, 1)
#test_examples = np.expand_dims(test_examples, 1)

model = tf.keras.Sequential([
    #tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(12, input_dim=size, activation='linear'),
    tf.keras.layers.Dense(12, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid'),

])

# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalsePositives(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.AUC()])
# print('------------------------------------------------------------------------')
# print('Training for fold ' + str(fold_n))
n_epochs = 20
history = model.fit(train_examples, train_labels,  # [train]
                    batch_size=32,
                    epochs=n_epochs,
                    shuffle=True,
                    validation_split=0.2)

# f = open("Evaluation.txt", 'a')

test_acc = model.evaluate(test_examples, test_labels, verbose=0)
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

# f.write("Accuracy for neuron1: " + str(n_neurons1) + " and neurons2: " + str(n_neurons2) + " is: " + str(test_acc[1]) + "\n")

# f.close()

# alla fine delle epoche:
# loss sul test set della CV
# scores = model.evaluate(train_examples[test], train_labels[test], verbose=0)
# cv_losses[fold_n - 1] = str(scores[0])
# media delle loss nelle epoche per questa fold
# epoch_losses[fold_n - 1] = np.mean(history.history['loss'])
# fold_n = fold_n + 1

# cv_loss = np.mean(cv_losses)
# print('Error on CV set is: ' + str(cv_loss))
# test_acc = model.evaluate(test_examples, test_labels, verbose=0)

# print('Error on test set is: ' + str(test_acc[0]))
# print('Accuracy on test set is: ' + str(test_acc[1]))

# x = np.arange(5)
# fig, ax = plt.subplots()
# plt.plot(x, epoch_losses, x, cv_losses)
# plt.legend(('Training set Loss', 'CV set Loss'), loc='upper center', shadow=True)
# plt.xlabel('K-th iteration')
# plt.ylabel('CrossEntropy')
# plt.show()
# test_predictions = model.predict(test_dataset).T