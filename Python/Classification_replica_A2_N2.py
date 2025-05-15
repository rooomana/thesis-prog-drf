#########################################################################
#
# | Replicating architecture from:
#
# Drone Detection Approach Based on Radio-Frequency Using Convolutional Neural Network
#
# Authors: Sara Al-Emadi
#          Felwa Al-Senaid
#
# Objective: Drone Classification
#
#########################################################################

############################## Libraries ################################
import os                   # [MR]
import time                 # [MR]
import math                 # [MR]
import json                 # [MR]
import threading            # [MR]
import numpy as np
from concurrent.futures import ThreadPoolExecutor   # [MR]
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

# [MR] (?) Model training improvement
# TODO: Check need for optimization
@tf.function(reduce_retracing=True)
def train_step(model, train_data, train_labels):
    model.fit(train_data, train_labels, epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)

############################# Parameters ###############################
np.random.seed(1)
K                    = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun   = 'categorical_crossentropy'
optimizer_algorithm  = 'adam'
number_inner_layers  = 3
linked_conv_pool_layers  = 2 # [MR]
linked_conv_only_layers  = 2 # [MR]
number_inner_neurons = 256
#number_epoch         = 100 # NN 1 | Two-class
number_epoch         = 350 # NN 2 | Multi-class
batch_length         = 50 # NN 1 & 2 | Two or Multi-class
#batch_length         = 32  # [MR] Increase for better performance
show_inter_results   = 0

opt = 2     # [MR] DNN Results number
current_directory_working = os.getcwd()     # [MR] Current working directory
results_path = rf"{current_directory_working}\Results_{opt}"    # [MR]

running_time = {}           # [MR] Timers
start_time = time.time()    # [MR] Start timer

############################### Loading ##################################
print("\nLoading Data ...")                                         # [MR]
filepath = os.path.dirname(current_directory_working)               # [MR] Path for easier management
Data = np.loadtxt(rf"{filepath}\Data\RF_Data.csv", delimiter=",")   # [MR]
print("Loaded Data.\n")                                             # [MR]

############################## Splitting #################################
print("\nPreparing Data ...")                                       # [MR]
x = np.transpose(Data[0:2047,:])
Label_1 = np.transpose(Data[2048:2049,:]).astype(int)
Label_2 = np.transpose(Data[2049:2050,:]).astype(int)
Label_3 = np.transpose(Data[2050:2051,:]).astype(int)
Label_opt = globals()[f'Label_{opt}']       # [MR]
y = encode(Label_opt)                       # [MR]
print("Prepared Data.\n")                                           # [MR]

################################ Main ####################################
cvscores = np.array([])     # [MR]
#cnt = 0
#kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
#for train, test in kfold.split(x, decode(y)):

## [MR] Each fold's process
def process_fold(train, test, fold_index, results_lock):
    global cvscores, running_time
    start_time_phase = time.time()  # [MR] Start phase timer
    #cnt = cnt + 1
    #print(f'\n| {cnt = }') # [MR]
    digits_K = math.floor(math.log10(abs(K))) + 1   # [MR]
    print(f'\n| Fold {fold_index:>{digits_K}} |')   # [MR]
    fold_x = x.reshape(x.shape[0], x.shape[1], 1)   # [MR] Reshape input (add channel dimension)
    
    ## [MR] Build model
    filters =       [32, 64, 64, 128, 128, 256]
    kernel_sizes =  [ 3,  3,  3,   3,   3,   3]
    model = Sequential()
    ## Input layer
    model.add(layers.Input(
        shape=(x.shape[1], 1)
    ))

    # [1-2] Linked Conv (w/ Pooling) layers
    for i in range(linked_conv_pool_layers):
        print(f"| Fold {fold_index:>{digits_K}} | {i = }") # [MR] Remove after testing
        model.add(layers.Conv1D(
            filters=filters[i],
            kernel_size=kernel_sizes[i],
            strides=1,
            padding='same',
            activation='relu'
        ))
        model.add(layers.AveragePooling1D(pool_size=3))
    
    # [3-4] Linked Conv only layers
    for i in range(linked_conv_pool_layers, 
                   linked_conv_pool_layers + linked_conv_only_layers):
        print(f"| Fold {fold_index:>{digits_K}} | {i = }") # [MR] Remove after testing
        model.add(layers.Conv1D(
            filters=filters[i],
            kernel_size=kernel_sizes[i],
            strides=1,
            padding='same',
            activation='relu'
        ))
    print(f"| Fold {fold_index:>{digits_K}} | {linked_conv_pool_layers = }") # [MR] Remove after testing
    print(f"| Fold {fold_index:>{digits_K}} | {linked_conv_only_layers = }") # [MR] Remove after testing
    linked_layers = linked_conv_pool_layers + linked_conv_only_layers
    print(f"| Fold {fold_index:>{digits_K}} | {linked_layers = }") # [MR] Remove after testing
    # [5] Conv (w/ Pooling) layer
    i = linked_layers
    print(f"| Fold {fold_index:>{digits_K}} | {i = }") # [MR] Remove after testing
    model.add(layers.Conv1D(
        filters=filters[i],
        kernel_size=kernel_sizes[i],
        strides=1,
        padding='same',
        activation='relu'
    ))
    model.add(layers.AveragePooling1D(pool_size=1))
    
    # [6] Conv only layer
    i = linked_layers + 1
    print(f"| Fold {fold_index:>{digits_K}} | {i = }") # [MR] Remove after testing
    model.add(layers.Conv1D(
        filters=filters[i],
        kernel_size=kernel_sizes[i],
        strides=1,
        padding='same',
        activation='relu'
    ))
    
    # Dropout to prevent overfitting
    model.add(layers.Dropout(0.2))
    # Flatten before fully connected layers
    model.add(layers.Flatten())
    # Fully connected layers
    model.add(layers.Dense(256, activation='relu'))
#    model.add(layers.Dense(y.shape[1], activation='sigmoid'))                                           # NN 1 | Two-class
    model.add(layers.Dense(y.shape[1], activation='softmax'))                                          # NN 2 | Multi-class
    
#    model.compile(loss='binary_crossentropy', optimizer=optimizer_algorithm, metrics=['accuracy'])      # NN 1 | Two-class
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_algorithm, metrics=['accuracy']) # NN 2 | Multi-class
    
    ## [MR] Display parameters
    if fold_index == 1: # Displays only for defined fold
        print("\n| Summary of the model:")
        model.summary()
        print("\n| Config of each layer:")
        for layer in model.layers:
            print(f"\n|| Layer \"{layer.name}\":")
            print(json.dumps(layer.get_config(), indent=4))

    ## [MR] Train and eval model
    # TODO: Fix model training - test with prints
    model.fit(fold_x[train], y[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
    #train_step(model, fold_x[train], y[train])
    scores = model.evaluate(fold_x[test], y[test], verbose=show_inter_results)
    print(f'\n| Fold {fold_index:>{digits_K}} | Scores = {scores[1] * 100}') # [MR]
    
    ## [MR] Save results
    y_pred = model.predict(fold_x[test])    # [MR]
    # [MR] (Results_{1,2,3} - Demo_4) - Only saving results for the 3rd NN (?)
    # np.savetxt("Results_3%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')
    results_file = rf"{results_path}\Results_{opt}{fold_index}.csv"     # [MR] Saved results path
    np.savetxt(results_file, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')   # [MR]
    
    ## [MR] Elapsed time
    print(f'\n| Fold {fold_index:>{digits_K}} | Ended')
    # TODO: Test time records
    # TODO: Time calculations inside or outside lock?
    with results_lock:
        end_time_phase = time.time()
        elapsed_time_phase = end_time_phase - start_time_phase
        running_time[f'elapsed_time_{fold_index}'] = elapsed_time_phase
        cvscores = np.append(cvscores, scores[1] * 100)
    print("\n| Fold %*d | Elapsed time: %.4f seconds\n" % (digits_K, fold_index, running_time[f'elapsed_time_{fold_index}']))

#########################################################################
## [MR] Threading
# Using ThreadPoolExecutor for efficient threading
results_lock = threading.Lock()  # Protect shared resources
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
print("\n> K-fold training (w/ threading) \nStarting...\n")

with ThreadPoolExecutor(max_workers=K) as executor:
    for fold_index, (train, test) in enumerate(kfold.split(x, decode(y)), start=1):
        executor.submit(process_fold, train, test, fold_index, results_lock)

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
print('\n')