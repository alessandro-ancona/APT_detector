import tensorflow as tf
from tensorflow import feature_column as fc
import pandas as pd
import numpy as np
import os
import random
import matplotlib
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
random.seed(123)

train = pd.read_csv('EKDD_Train.csv')
# [2, 3, 4, 5, 6, 7, 8, 10, 12, 23, 25, 29, 30, 35, 36, 37, 38, 40]
train_examples = np.array(train.drop(['Class'], axis=1))
train_labels = np.array(train['Class'])

size = train_examples.shape[1]

test = pd.read_csv('EKDD_Test.csv')

test_examples = np.array(test.drop(['Class'], axis=1))
test_labels = np.array(test['Class'])

test_labels = test_labels.astype('float32')
train_labels = train_labels.astype('float32')

# train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

fold_n = 1
for train, test in kfold.split(train_examples, train_labels):
    a = train
    b = test
    # [248.341064453125, 0.8877750039100647, 1491.0, 1039.0, 0.896488606929779]** Neurons1 = 25 Neurons2 = 11
    # [161.58827209472656, 0.8268275260925293, 589.0, 3315.0, 0.865807056427002]** Neurons1 = 23 Neurons2 = 22
    # [32.2400016784668, 0.9041873812675476, 858.0, 1302.0, 0.931381344795227]** Neurons1 = 12 Neurons2 = 11
    # [50.86280822753906, 0.8849804997444153, 1411.0, 1182.0, 0.8966571688652039]** Neurons1 = 12 Neurons2 = 9

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(12, input_dim=size, activation='relu'),
        tf.keras.layers.Dense(11, input_dim=size, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),

    ])

    # model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())
    print('------------------------------------------------------------------------')
    print('Training for fold ' + str(fold_n))

    history = model.fit(train_examples[train], train_labels[train],
                        batch_size=50,
                        epochs=1,
                        shuffle=False)

    scores = model.evaluate(train_examples[test], train_labels[test], verbose=0)
    print('Loss for fold ' + str(fold_n) + ': ' + str(scores[0]))
    print(history.history['loss'])
    fold_n = fold_n + 1

test_acc = model.evaluate(test_examples, test_labels, verbose=0)
print('Error on test set is: ' + str(test_acc[0]))
print('Accuracy on test set is: ' + str(test_acc[1]))

    # test_predictions = model.predict(test_dataset).T
