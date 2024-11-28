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
import time # [MR]
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
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

running_time = {} # [MR] Timers

start_time = time.time() # [MR] Start timer

############################### Loading ##################################
print("Loading Data ...")
filepath = 'D:\ISCTE\Thesis\DroneRF'; # [MR] Path for easier management
Data = np.loadtxt(filepath + "\Data\RF_Data.csv", delimiter=",") # [MR]
############################## Splitting #################################
print("Preparing Data ...")
x = np.transpose(Data[0:2047,:])
Label_1 = np.transpose(Data[2048:2049,:]); Label_1 = Label_1.astype(int);
Label_2 = np.transpose(Data[2049:2050,:]); Label_2 = Label_2.astype(int);
Label_3 = np.transpose(Data[2050:2051,:]); Label_3 = Label_3.astype(int);
y = encode(Label_3)
################################ Main ####################################
running_times = [] # [MR] Timers
cvscores    = []
cnt         = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
for train, test in kfold.split(x, decode(y)):
    start_time_phase = time.time() # [MR] Start phase timer
    cnt = cnt + 1
    print(f'| {cnt = }') # [MR]
    model = Sequential()
    for i in range(number_inner_layers):
        model.add(Dense(int(number_inner_neurons/2), input_dim = x.shape[1], activation = inner_activation_fun))
    model.add(Dense(y.shape[1], activation = outer_activation_fun))
    model.compile(loss = optimizer_loss_fun, optimizer = optimizer_algorithm, metrics =         ['accuracy'])
    model.fit(x[train], y[train], epochs = number_epoch, batch_size = batch_length, verbose = show_inter_results)
    scores = model.evaluate(x[test], y[test], verbose = show_inter_results)
    print(f'scores = {scores[1]*100}') # [MR]
    cvscores.append(scores[1]*100)
    y_pred = model.predict(x[test])
    np.savetxt("Results_3%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')
    ## [MR] Elapsed time
    print('Ended' + ' | %s | ' % cnt, end="")
    end_time_phase = time.time()
    elapsed_time_phase = end_time_phase - start_time_phase
    running_time[f'elapsed_time_{cnt}'] = elapsed_time_phase
    print("Elapsed time: %.4f seconds\n" % (elapsed_time_phase))
#########################################################################
## [MR] Elapsed time
print('Ended' + ' | Total')
end_time = time.time()
elapsed_time_total = end_time - start_time
running_time['elapsed_time_total'] = elapsed_time_total
print("Elapsed time: %.4f seconds\n" % (elapsed_time_total))

## [MR] Print running time
longest_key_length = max(len(key) for key in running_time)
print('Running Time:')
for phase_name, phase_elapsed_time in running_time.items():
    print(f'| {phase_name:<{longest_key_length}} = {phase_elapsed_time:.4f} seconds')