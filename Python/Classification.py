#########################################################################
#
# Copyright 2018 Mohammad Al-Sa'd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Mohammad F. Al-Sa'd (mohammad.al-sad@tut.fi)
#          Amr Mohamed         (amrm@qu.edu.qa)
#          Abdulla Al-Ali
#          Tamer Khattab
#
# The following reference should be cited whenever this script is used:
#     M. Al-Sa'd et al. "RF-based drone detection and identification using
#     deep learning approaches: an initiative towards a large open source
#     drone database", 2018.
#
# Last Modification: 19-11-2018
#########################################################################

############################## Libraries ################################
import os # [MR]
import time # [MR]
import numpy as np
import tensorflow as tf                             # [MR]
from tensorflow.keras.utils import to_categorical   # [MR]
from tensorflow.keras.models import Sequential      # [MR]
#from tensorflow.keras.layers import Dense           # [MR]
layers = tf.keras.layers                            # [MR]
from sklearn.model_selection import StratifiedKFold
############################## Functions ###############################
def decode(datum):
    y = np.zeros((datum.shape[0],1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y
def encode(datum):
    return to_categorical(datum)
############################# Parameters ###############################
np.random.seed(1)
K                    = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun   = 'mse'
optimizer_algorithm  = 'adam'
number_inner_layers  = 3
number_inner_neurons = 256
number_epoch         = 200
batch_length         = 10
show_inter_results   = 0

opt = 1; # [MR] DNN Results number
current_directory_working = os.getcwd() # [MR] Current working directory
results_path = rf"{current_directory_working}\Results_{opt}" # [MR]

running_time = {} # [MR] Timers

start_time = time.time() # [MR] Start timer

############################### Loading ##################################
print("\nLoading Data ...")                                         # [MR]
filepath = os.path.dirname(current_directory_working)               # [MR] Path for easier management
Data = np.loadtxt(rf"{filepath}\Data\RF_Data.csv", delimiter=",")   # [MR]
print("Loaded data.\n")                                             # [MR]
############################## Splitting #################################
print("\nPreparing Data ...")                                       # [MR]
x = np.transpose(Data[0:2047,:])
Label_1 = np.transpose(Data[2048:2049,:]); Label_1 = Label_1.astype(int);
Label_2 = np.transpose(Data[2049:2050,:]); Label_2 = Label_2.astype(int);
Label_3 = np.transpose(Data[2050:2051,:]); Label_3 = Label_3.astype(int);
# [MR] TODO: Understand if Label is related to the NN
Label_opt = globals()[f'Label_{opt}'] # [MR]
#y = encode(Label_3)
y = encode(Label_opt) # [MR]
print("Prepared data.\n")                                           # [MR]
################################ Main ####################################
cvscores    = []
cnt         = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
for train, test in kfold.split(x, decode(y)):
    start_time_phase = time.time() # [MR] Start phase timer
    cnt = cnt + 1
    print(f'| {cnt = }') # [MR]
    model = Sequential()
    for i in range(number_inner_layers):
        # TODO: Compare different types of layers
        model.add(layers.Dense(int(number_inner_neurons/2), input_dim = x.shape[1], activation = inner_activation_fun))
        #model.add(layers.Conv1D(int(number_inner_neurons/2), kernel_size=3, input_shape=(x.shape[1], 1), activation=inner_activation_fun))
        #model.add(layers.Conv2D(int(number_inner_neurons/2), kernel_size=(3, 3), input_shape=(x.shape[1], 1, 1), activation=inner_activation_fun))
    model.add(layers.Dense(y.shape[1], activation = outer_activation_fun))
    model.compile(loss = optimizer_loss_fun, optimizer = optimizer_algorithm, metrics =         ['accuracy'])
    model.fit(x[train], y[train], epochs = number_epoch, batch_size = batch_length, verbose = show_inter_results)
    scores = model.evaluate(x[test], y[test], verbose = show_inter_results)
    print(f'scores = {scores[1]*100}') # [MR]
    cvscores.append(scores[1]*100)
    y_pred = model.predict(x[test])
    # [MR] (Results_{1,2,3} - Demo_4) - Only saving results for the 3rd NN (?)
    # np.savetxt("Results_3%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')
    np.savetxt(rf"{results_path}\Results_{opt}{cnt}.csv", \
        np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s') # [MR]
    ## [MR] Elapsed time
    print('Ended | %s' % cnt)
    end_time_phase = time.time()
    elapsed_time_phase = end_time_phase - start_time_phase
    running_time[f'elapsed_time_{cnt}'] = elapsed_time_phase
    print("Elapsed time: %.4f seconds\n" % (elapsed_time_phase))
#########################################################################
## [MR] Elapsed time
print('Ended | Total')
end_time = time.time()
elapsed_time = end_time - start_time
running_time['elapsed_time_total'] = elapsed_time
print("Elapsed time: %.4f seconds\n" % (elapsed_time))

## [MR] Print running time
longest_name_length = max(len(name) for name in running_time.keys())
longest_time_length = max(len(str(format(time, '.4f'))) for time in running_time.values())
print('\nRunning Time:\n')
for phase_name, phase_elapsed_time in running_time.items():
    print(f'| {phase_name:<{longest_name_length}} = {phase_elapsed_time:>{longest_time_length}.4f} seconds')