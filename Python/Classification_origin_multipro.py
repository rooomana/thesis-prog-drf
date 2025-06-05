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
import os  # [MR]
import time  # [MR]
import json  # [MR]
import numpy as np
import tensorflow as tf  # [MR]
from tensorflow.keras.utils import to_categorical  # [MR]
from tensorflow.keras.models import Sequential  # [MR]
layers = tf.keras.layers  # [MR]
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool, Manager, Lock  # [MR] Multiprocessing
import traceback  # [MR] For exception tracebacks in subprocesses

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
batch_length = 10
show_inter_results = 0

opt = 1  # [MR] DNN Results number
current_directory_working = os.getcwd()  # [MR] Current working directory
results_path = rf"{current_directory_working}\Results_{opt}"  # [MR]

############################# Worker Function #############################
def train_fold_mp(args):
    train, test, fold_index, cvscores, running_time, results_lock, x_local, y_local = args
    start_time_phase = time.time()  # [MR] Start phase timer
    try:
        print(f'| Fold {fold_index} |')  # [MR]
        model = Sequential()
        for i in range(number_inner_layers):
            model.add(layers.Dense(
                int(number_inner_neurons / 2),
                input_dim=x_local.shape[1],
                activation=inner_activation_fun
            ))
        model.add(layers.Dense(
            y_local.shape[1],
            activation=outer_activation_fun
        ))
        model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])

        # [MR] Print model parameters only for fold 1
        if fold_index == 1:
            print("\n| Summary of the model:")
            model.summary()
            print("\n| Config of each layer:")
            for layer in model.layers:
                print(f"|| Layer \"{layer.name}\":")
                print(json.dumps(layer.get_config(), indent=4))

        model.fit(x_local[train], y_local[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
        scores = model.evaluate(x_local[test], y_local[test], verbose=show_inter_results)
        print(f'scores = {scores[1] * 100}')  # [MR]

        y_pred = model.predict(x_local[test])

        # Save results per fold
        np.savetxt(
            rf"{results_path}\Results_{opt}{fold_index}.csv",
            np.column_stack((y_local[test], y_pred)),
            delimiter=",",
            fmt='%s'
        )

        with results_lock:
            end_time_phase = time.time()
            elapsed_time_phase = end_time_phase - start_time_phase
            running_time[f'elapsed_time_{fold_index}'] = elapsed_time_phase
            cvscores.append(scores[1] * 100)

        print(f'| Fold {fold_index} | Ended')
        print(f"Elapsed time: {elapsed_time_phase:.4f} seconds\n")

    except Exception as e:
        print(f"Exception in fold {fold_index}: {e}")
        traceback.print_exc()

############################# Main #####################################
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # [MR] Safe on Windows

    # Create results folder if it does not exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ############################### Loading ##################################
    print("\nLoading Data ...")  # [MR]
    filepath = os.path.dirname(current_directory_working)  # [MR] Path for easier management
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

    ############################# Shared objects ############################
    manager = Manager()
    running_time = manager.dict()  # [MR] Shared dictionary for timers
    cvscores = manager.list()  # [MR] Shared list for cross-validation scores
    results_lock = manager.Lock()  # [MR] Lock for synchronisation

    start_time = time.time()  # [MR] Start timer

    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

    # Prepare arguments for multiprocessing: pass all required data explicitly
    args_list = [
        (train, test, fold_index + 1, cvscores, running_time, results_lock, x, y)
        for fold_index, (train, test) in enumerate(kfold.split(x, decode(y)))
    ]

    print("\nStarting k-fold training with multiprocessing...\n")

    with Pool(processes=K) as pool:  # [MR] Multiprocessing pool with K workers
        pool.map(train_fold_mp, args_list)

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