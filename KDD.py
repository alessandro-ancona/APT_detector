import tensorflow as tf
from tensorflow import feature_column as fc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt

# 43 features. 42 esima è tipologia di traffico. 43 esima è difficolta nella detection.
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train = pd.read_csv('KDDTrain.csv')

train_examples = train.drop(['Protocol-type', 'Service', 'Flag', 'Difficulty', 'Class'], axis=1)
train_labels = train['Class']

size = train_examples.shape[1]

i = 0
for item in train_labels:
    if item == 'normal':
        train_labels[i] = np.float32(0)
    else:
        train_labels[i] = np.float32(1)
    i = i+1

test = pd.read_csv('KDDTest.csv')

test_examples = test.drop(['Protocol-type', 'Service', 'Flag', 'Difficulty', 'Class'], axis=1)
test_labels = test['Class']

i = 0
for item in test_labels:
    if item == 'normal':
        test_labels[i] = np.float32(0)
    else:
        test_labels[i] = np.float32(1)
    i = i+1

# test_labels = np.asarray(test_labels).astype('float32')
# train_labels = np.asarray(train_labels).astype('float32')

# Protocol_type = fc.indicator_column(fc.categorical_column_with_vocabulary_list(key='Protocol-type', vocabulary_list=('tcp', 'udp', 'icmp')))
# Service = fc.indicator_column(fc.categorical_column_with_vocabulary_list(key='Service', vocabulary_list=('aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50')))
# Flag = fc.indicator_column(fc.categorical_column_with_vocabulary_list(key='Flag',  vocabulary_list=('OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH')))
Duration = fc.numeric_column('Duration')
Src_bytes = fc.numeric_column('Src-bytes')
Dst_bytes = fc.numeric_column('Dst-bytes')
Land = fc.numeric_column('Land')
Wrong_fragment = fc.numeric_column('Wrong-fragment')
Urgent = fc.numeric_column('Urgent')
Hot = fc.numeric_column('Hot')
Num_failed_logins = fc.numeric_column('Num-failed-logins')
Logged_in = fc.numeric_column('Logged-in')
Num_compromised = fc.numeric_column('Num-compromised')
Root_shell = fc.numeric_column('Root-shell')
Su_attempted = fc.numeric_column('Su-attempted')
Num_root = fc.numeric_column('Num-root')
Num_file_creations = fc.numeric_column('Num-file-creations')
Num_shells = fc.numeric_column('Num-shells')
Num_access_files = fc.numeric_column('Num-access-files')
Num_outbound_cmds = fc.numeric_column('Num-outbound-cmds')
Is_host_login = fc.numeric_column('Is-host-login')
Is_guest_login = fc.numeric_column('Is-guest-login')
Count = fc.numeric_column('Count')
Srv_count = fc.numeric_column('Srv-count')
Serror_rate = fc.numeric_column('Serror-rate')
Srv_serror_rate = fc.numeric_column('Srv-serror-rate')
Rerror_rate = fc.numeric_column('Rerror-rate')
Srv_rerror_rate = fc.numeric_column('Srv-rerror-rate')
Same_srv_rate = fc.numeric_column('Same-srv-rate')
Diff_srv_rate = fc.numeric_column('Diff-srv-rate')
Srv_diff_host_rate = fc.numeric_column('Srv-diff-host-rate')
Dst_host_count = fc.numeric_column('Dst-host-count')
Dst_host_srv_count = fc.numeric_column('Dst-host-srv-count')
Dst_host_same_srv_rate = fc.numeric_column('Dst-host-same-srv-rate')
Dst_host_diff_srv_rate = fc.numeric_column('Dst-host-diff-srv-rate')
Dst_host_same_src_portrate = fc.numeric_column('Dst-host-same-src-portrate')
Dst_host_srv_diff_hostrate = fc.numeric_column('Dst-host-srv-diff-hostrate')
Dst_host_serror_rate = fc.numeric_column('Dst-host-serror-rate')
Dst_host_srv_serror_rate = fc.numeric_column('Dst-host-srv-serror-rate')
Dst_host_rerror_rate = fc.numeric_column('Dst-host-rerror-rate')
Dst_host_srv_rerror_rate = fc.numeric_column('Dst-host-srv-rerror-rate')
#Class = fc.categorical_column_with_vocabulary_list(key='Class', vocabulary_list=('normal', 'anomaly'))
Difficulty = fc.numeric_column('Difficulty')

feature_col = [Duration, Src_bytes, Dst_bytes, Land, Wrong_fragment,Urgent, Hot, Num_failed_logins, Logged_in, Num_compromised, Root_shell, Su_attempted, Num_root, Num_file_creations, Num_shells, Num_access_files, Num_outbound_cmds,
               Is_host_login, Is_guest_login, Count, Srv_count, Serror_rate, Srv_serror_rate, Same_srv_rate, Diff_srv_rate, Srv_diff_host_rate, Dst_host_count, Dst_host_srv_count, Dst_host_same_srv_rate, Dst_host_diff_srv_rate,
               Dst_host_same_src_portrate, Dst_host_srv_diff_hostrate, Dst_host_serror_rate, Dst_host_rerror_rate]

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(train_examples,
                                                train_labels,
                                                batch_size=50,
                                                num_epochs=1000,
                                                shuffle=True)

eval_func = tf.compat.v1.estimator.inputs.pandas_input_fn(test_examples,
                                                test_labels,
                                                batch_size=50,
                                                num_epochs=1,
                                                shuffle=False)
predict_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=test_examples,
                                                       num_epochs=1,
                                                       shuffle=False)

estimator = tf.estimator.DNNClassifier(
    n_classes=2,
    feature_columns=feature_col,
    hidden_units=[50, 50],
    activation_fn=tf.nn.softmax,
    dropout=None,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)

# normal	neptune	warezclient	ipsweep	portsweep	teardrop	nmap	satan	smurf	pod	back	guess_passwd	ftp_write	multihop	rootkit	buffer_overflow	imap	warezmaster	phf	land	loadmodule	spy	perl

estimator.train(input_fn=input_func, steps=500)
estimator.evaluate(eval_func)
print('done')
