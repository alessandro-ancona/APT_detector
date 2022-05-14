import keras.callbacks
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print()


train = pd.read_csv('EKDD_Train.csv')
# [2, 3, 4, 5, 6, 7, 8, 10, 12, 23, 25, 29, 30, 35, 36, 37, 38, 40]
train_examples = np.array(train.iloc[:, [2, 3, 4, 5, 6, 7, 8, 10, 12, 23, 25, 29, 30, 35, 36, 37, 38, 40]])
train_labels = np.array(train['Class'])

size = train_examples.shape[1]

test = pd.read_csv('EKDD_Test.csv')

test_examples = np.array(test.iloc[:, [2, 3, 4, 5, 6, 7, 8, 10, 12, 23, 25, 29, 30, 35, 36, 37, 38, 40]])
test_labels = np.array(test['Class'])

test_labels = test_labels.astype('float32')
train_labels = train_labels.astype('float32')

# train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

n_fold = 5
kfold = KFold(n_splits=n_fold, shuffle=True, random_state=123)
epoch_losses = np.zeros(n_fold)
cv_losses = np.zeros(n_fold)

fold_n = 1
#for train, test in kfold.split(train_examples, train_labels):
    #a = train
    #b = test


    # 8 8 su 10 epoche da 90 per cento, su 50 è 79.4 per cento
    #12 12 su 30 epoche -> oltre le 30 epoche la loss converge ed è sul 80% accuracy sul test set
    #24 12 su 50 epoche -> siamo sul 79 per cento accuracy
    #14 12 su 50 epoche -> siamo sul 78 ma è incredibile l accuracy e convergenza sul training set
    #12 8 su 50 epoche -> siamo sul 79,4 ma è sbilanciato sui falsi negativi il test, ottima accuracy sul train


model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=size, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),

])

    # model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalsePositives(),
                       tf.keras.metrics.FalseNegatives()])
    #print('------------------------------------------------------------------------')
    #print('Training for fold ' + str(fold_n))

history = model.fit(train_examples, train_labels, #[train]
                            batch_size=32,
                            epochs=50,
                            shuffle=True)
        #f = open("Evaluation.txt", 'a')
test_acc = model.evaluate(test_examples, test_labels, verbose=0)
print(test_acc)
y_pred_keras = model.predict(test_examples).ravel()
fpr_keras, tpr_keras, threashold_keras = roc_curve(test_labels, y_pred_keras)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras)

plt.figure(2)
plt.plot(history.history['loss'])

plt.show()

        #f.write("Accuracy for neuron1: " + str(n_neurons1) + " and neurons2: " + str(n_neurons2) + " is: " + str(test_acc[1]) + "\n")

#f.close()

    # alla fine delle epoche:
    # loss sul test set della CV
    #scores = model.evaluate(train_examples[test], train_labels[test], verbose=0)
    #cv_losses[fold_n - 1] = str(scores[0])
    # media delle loss nelle epoche per questa fold
    #epoch_losses[fold_n - 1] = np.mean(history.history['loss'])
    #fold_n = fold_n + 1

#cv_loss = np.mean(cv_losses)
#print('Error on CV set is: ' + str(cv_loss))
#test_acc = model.evaluate(test_examples, test_labels, verbose=0)

#print('Error on test set is: ' + str(test_acc[0]))
#print('Accuracy on test set is: ' + str(test_acc[1]))

# x = np.arange(5)
# fig, ax = plt.subplots()
# plt.plot(x, epoch_losses, x, cv_losses)
# plt.legend(('Training set Loss', 'CV set Loss'), loc='upper center', shadow=True)
# plt.xlabel('K-th iteration')
# plt.ylabel('CrossEntropy')
# plt.show()
    # test_predictions = model.predict(test_dataset).T
