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
#########################################################################

############################## Libraries ################################
import os  # [MR]
import time  # [MR]
import math # [MR]
import json  # [MR]
import numpy as np
import tensorflow as tf  # [MR]
from tensorflow.keras.utils import to_categorical  # [MR]
from tensorflow.keras.models import Sequential  # [MR]
layers = tf.keras.layers  # [MR]
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor, as_completed  # [MR] Threading
from threading import Lock  # [MR] Thread safety

############################## Functions ###############################
def decode(datum):
    y = np.zeros((datum.shape[0], 1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y

def encode(datum):
    return to_categorical(datum)

############################# Parameters ###############################
np.random.seed(1)
K = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun = 'mse'
optimizer_algorithm = 'adam'
number_inner_layers = 3
number_inner_neurons = 256
number_epoch = 200
batch_length = 32  # [MR] Increased batch size for better performance
show_inter_results = 0

opt = 1  # [MR] DNN Results number
current_directory_working = os.getcwd()  # [MR] Current working directory
results_path = rf"{current_directory_working}\Results_{opt}"  # [MR]

running_time = {}  # [MR] Timers
start_time = time.time()  # [MR] Start timer

############################### Loading ##################################
print("\nLoading Data ...")  # [MR]
filepath = os.path.dirname(current_directory_working)  # [MR]
Data = np.loadtxt(rf"{filepath}\Data\RF_Data.csv", delimiter=",")  # [MR]
print("Loaded Data.\n")  # [MR]

############################## Splitting #################################
print("\nPreparing Data ...")  # [MR]
x = np.transpose(Data[0:2047, :])
Label_1 = np.transpose(Data[2048:2049, :]).astype(int)
Label_2 = np.transpose(Data[2049:2050, :]).astype(int)
Label_3 = np.transpose(Data[2050:2051, :]).astype(int)
Label_opt = globals()[f'Label_{opt}']  # [MR]
y = encode(Label_opt)  # [MR]
print("Prepared Data.\n")  # [MR]

################################ Main ####################################
cvscores = []
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
results_lock = Lock()  # [MR] Thread safety for results

def train_fold(train, test, fold_index):
    start_time_phase = time.time()  # [MR] Start phase timer
    digits_K = math.floor(math.log10(abs(K))) + 1  # [MR]
    print(f'| Fold {fold_index:>{digits_K}} |')  # [MR]
    model = Sequential()
    model.add(layers.Dense(int(number_inner_neurons / 2), input_dim=x.shape[1], activation=inner_activation_fun))
    for _ in range(number_inner_layers - 1):
        model.add(layers.Dense(int(number_inner_neurons / 2), activation=inner_activation_fun))
    model.add(layers.Dense(y.shape[1], activation=outer_activation_fun))
    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])
    
    model.fit(x[train], y[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
    scores = model.evaluate(x[test], y[test], verbose=show_inter_results)
    y_pred = model.predict(x[test])
    
    
    np.savetxt(rf"{results_path}\Results_{opt}{fold_index}.csv", np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')  # [MR]
    
    ## [MR] Elapsed time
    print(f'| Fold {fold_index:>{digits_K}} | Ended')
    with results_lock:
        end_time_phase = time.time()
        elapsed_time_phase = end_time_phase - start_time_phase
        running_time[f'elapsed_time_{fold_index}'] = elapsed_time_phase
        cvscores = np.append(cvscores, scores[1] * 100)
    print("| Fold %*d | Elapsed time: %.4f seconds\n" % (digits_K, fold_index, running_time[f'elapsed_time_{fold_index}']))

############################ Threaded Execution #########################
print("\nStarting k-fold training with threading...\n")
with ThreadPoolExecutor(max_workers=K) as executor:  # [MR] Threading
    futures = {executor.submit(train_fold, train, test, fold_index): fold_index for fold_index, (train, test) in enumerate(kfold.split(x, decode(y)), start=1)}
    for future in as_completed(futures):
        try:
            future.result()  # Ensure exceptions are caught
        except Exception as e:
            print(f"Error in fold {futures[future]}: {e}")  # [MR] Error handling

#########################################################################
## [MR] Elapsed time
print('Ended | Total')
end_time = time.time()
elapsed_time = end_time - start_time
running_time['elapsed_time_total'] = elapsed_time
print(f"Elapsed time: {elapsed_time:.4f} seconds\n")

## [MR] Print running time
longest_name_length = max(len(name) for name in running_time.keys())
longest_time_length = max(len(str(format(time, '.4f'))) for time in running_time.values())
print('\nRunning Time:\n')
for phase_name, phase_elapsed_time in running_time.items():
    print(f'| {phase_name:<{longest_name_length}} = {phase_elapsed_time:>{longest_time_length}.4f} seconds')