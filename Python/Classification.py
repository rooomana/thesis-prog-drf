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
import math # [MR]
import threading # [MR]
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
print("Loaded Data.\n")                                             # [MR]
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
print("Prepared Data.\n")                                           # [MR]
################################ Main ####################################
cvscores    = []
#cnt         = 0
#kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
#for train, test in kfold.split(x, decode(y)):
# [MR] Function to handle each fold's training and evaluation
def process_fold(train, test, fold_index, results_lock):
    global cvscores, running_time
    start_time_phase = time.time() # [MR] Start phase timer
    #cnt = cnt + 1
    #print(f'| {cnt = }') # [MR]
    digits_K = math.floor(math.log10(abs(K))) + 1 # [MR]
    print(f'| Fold {fold_index:>{digits_K}} |') # [MR]
    fold_x = x.reshape(x.shape[0], x.shape[1], 1) # [MR] Reshape input (add channel dimension)
    
    # [MR] Build the model
    model = Sequential()
    for i in range(number_inner_layers):
        # TODO: Compare different types of layers
        # [MR] Dense layer
        #model.add(layers.Dense(int(number_inner_neurons/2), 
        #                       input_dim = x.shape[1], 
        #                       activation = inner_activation_fun))
        # [MR] Convolutional layer
        model.add(layers.Conv1D(filters = int(number_inner_neurons/2), 
                                kernel_size = 3, 
                                activation = inner_activation_fun, 
                                input_shape = (x.shape[1], 1) if i == 0 else None))
    model.add(layers.Flatten()) # [MR] Flatten before output
    model.add(layers.Dense(y.shape[1], activation = outer_activation_fun))
    model.compile(loss = optimizer_loss_fun, optimizer = optimizer_algorithm, metrics = ['accuracy'])
    
    # [MR] Train and evaluate the model
    model.fit(fold_x[train], y[train], epochs = number_epoch, batch_size = batch_length, verbose = show_inter_results)
    scores = model.evaluate(fold_x[test], y[test], verbose = show_inter_results)
    print(f'| Fold {fold_index:>{digits_K}} | Scores = {scores[1] * 100}') # [MR]
    
    # [MR] Save results
    y_pred = model.predict(fold_x[test]) # [MR]
    # [MR] (Results_{1,2,3} - Demo_4) - Only saving results for the 3rd NN (?)
    # np.savetxt("Results_3%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')
    results_file = rf"{results_path}\Results_{opt}{fold_index}.csv" # [MR] Saved results path
    np.savetxt(results_file, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s') # [MR]
    
    ## [MR] Elapsed time
    print(f'| Fold {fold_index:>{digits_K}} | Ended')
    end_time_phase = time.time()
    elapsed_time_phase = end_time_phase - start_time_phase
    with results_lock:
        cvscores.append(scores[1] * 100)
        running_time[f'elapsed_time_{fold_index}'] = elapsed_time_phase
    print("| Fold %*d | Elapsed time: %.4f seconds\n" % (digits_K, fold_index, elapsed_time_phase))
    
# Use threading for the k-fold loop
threads = []
results_lock = threading.Lock()  # Protect shared resources
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
print("K-fold training (w/ threading) \nStarting...\n")

# Start threads (each fold)
for fold_index, (train, test) in enumerate(kfold.split(x, decode(y)), start=1):
    thread = threading.Thread(target=process_fold, args=(train, test, fold_index, results_lock))
    threads.append(thread)
    thread.start()

# Complete all threads
for thread in threads:
    thread.join()
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