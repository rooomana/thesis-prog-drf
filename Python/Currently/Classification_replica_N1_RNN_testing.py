#########################################################################
#
# Objective: Drone Detection
#
# | Replicating architecture from:
#
# Ideas shared by supervisor
#
# Authors: Me
#
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
from sklearn.utils import class_weight              # [MR] For balance
import matplotlib.pyplot as plt                     # [MR] For analysis

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
#@tf.function(reduce_retracing=True)
#def train_step(model, train_data, train_labels, test_data, test_labels):
#    model.fit(train_data, train_labels, epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results, validation_data=(test_data, test_labels))

############################# Parameters ###############################
np.random.seed(1)
K                    = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun   = 'categorical_crossentropy'
optimizer_algorithm  = 'adam'
lstm_join_layers  = 1
number_inner_layers  = 3
number_inner_neurons = 256
number_epoch         = 12
batch_length         = 128 # TODO: Temporarily - for faster training
#batch_length         = 50 # NN 1 & 2 | Two or Multi-class
#batch_length         = 32  # [MR] Increase for better performance
show_inter_results   = 1

opt = 1     # [MR] DNN Results number
current_directory_working = os.getcwd()     # [MR] Current working directory
results_path = rf"{current_directory_working}\Results_{opt}"    # [MR]
histories = [None] * K  # [MR] For analysis

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

############################### Balance ##################################
# [MR] Balancing dataset
print("\Balancing Data ...")
y_int = np.argmax(y, axis=1) # one-hot to int
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_int),
    y=y_int)
class_weights = dict(enumerate(class_weights))
print("Balanced Data.\n")

################################ Main ####################################
cvscores = np.array([])     # [MR]
#cnt = 0
#kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
#for train, test in kfold.split(x, decode(y)):

## [MR] Each fold's process
def process_fold(train, test, fold_index):
    global digits_K, cvscores, running_time, histories
    start_time_phase = time.time()  # [MR] Start phase timer
    #cnt = cnt + 1
    #print(f'\n| {cnt = }') # [MR]
    digits_K = math.floor(math.log10(abs(K))) + 1   # [MR]
    print(f'\n| Fold {fold_index:>{digits_K}} |')   # [MR]
    fold_x = x.reshape(x.shape[0], x.shape[1], 1)   # [MR] Reshape input
    
    ## [MR] Build model
    model = Sequential()
    ## Input layer
    model.add(layers.Input(
        shape=(fold_x.shape[1], fold_x.shape[2])
    ))

    ## TODO:
    ########## Ideas yet to try ##########################################################################################################################
    ## - batch size       | increase for faster training - return back for real training
    ## - dropout          | increase amount - prevent overfitting + vanishing weights
    ## - recurrent layers | add LSTM layers (intermediate layers with return_sequence=True) - better results?
    ## - normalisation    | implement - for stable training + vanishing weights [LayerNormalization() before dropout]
    ######################################################################################################################################################

    #################### Tests
    ## T1:  1_layer
    ######  | [10 epoch] return=True   w pooling  w dropout
    ## T2:  1_layer
    ######  | [ 4 epoch] return=True   w pooling  - dropout
    ## T3:  1_layer
    ######  | [ 4 epoch] return=False  - pooling  - dropout
    ## T4:  1_layer
    ######  | [50 epoch] " "
    ## T5:  1_layer
    ######  | [20 epoch] " "           - dense
    ## T6:  1_layer
    ######  | [20 epoch] " "           + rnn[relu + units(80)]
    ## T7:  1_layer
    ######  | [20 epoch] " "           + rnn[tanh + units(64)] w dense   + flatten
    ## T8:  1_layer
    ######  | [20 epoch] return=True   w pooling  w dropout    w denses  - flatten
    ## T9:  1_layer
    ######  | [12 epoch] return=False  w class weights balance - denses
    ######  |                          - pooling  - dropout
    ## T10: 1_layer
    ######  | [12 epoch]               w threads  w dropout
    ## T11: 1_layer
    ######  | [12 epoch] w normalisa.
    ## T12: 1_layer
    ######  | [12 epoch] - normalisa.  w lay.(2)  w drop.(2)
    ## T13: 1_layer
    ######  | [12 epoch] (?) w normalisa.

    #################### Final
    # TODO: Review ideas
    ## F1: Execute 1_layer w/ 200 epoch
    ## F2: Execute 2_layer w/ 200 epoch

    # RNN layers + Pooling layers
    for i in range(lstm_join_layers):
        # RNN layers
        model.add(layers.LSTM(
            units=64, 
            activation='tanh', 
            return_sequences=False if i == lstm_join_layers - 1 else True
        ))
        # Pooling layers
        #model.add(layers.GlobalAveragePooling1D())
        #model.add(layers.GlobalMaxPooling1D())

        # TODO: (?) Include normalization
        #model.add(layers.LayerNormalization()) # Preferred for LSTM
        #model.add(layers.BatchNormalization())

        # Dropout to prevent overfitting
        # TODO: (?) Implement more dropout layers
        model.add(layers.Dropout(0.25))

    # Fully connected layers
    #model.add(layers.Dense(128, activation='relu'))
    ## Output layer
    model.add(layers.Dense(y.shape[1], activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer_algorithm, metrics=['accuracy'])
    
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
    print(f'\n| Fold {fold_index:>{digits_K}} | Training starting...') # [MR]
    history = model.fit(
        fold_x[train], y[train],
        epochs=number_epoch,
        batch_size=batch_length,
        verbose=show_inter_results,
        class_weight=class_weights,
        validation_data=(fold_x[test], y[test])  # [MR] For analysis
    )
    # history = train_step(model, fold_x[train], y[train], fold_x[test], y[test])
    print(f'\n| Fold {fold_index:>{digits_K}} | Training finished.') # [MR]
    histories[fold_index - 1] = history.history  # [MR] For analysis
    scores = model.evaluate(fold_x[test], y[test], verbose=show_inter_results)
    print(f'\n| Fold {fold_index:>{digits_K}} | Scores = {scores[1] * 100}') # [MR]
    
    ## [MR] Save results
    y_pred = model.predict(fold_x[test])    # [MR]
    # [MR] Debugging
    print(f'\n| Fold {fold_index:>{digits_K}} | Mean | {np.mean(y_pred):.6f}') # [MR]
    print(f'\n| Fold {fold_index:>{digits_K}} | Min  |  {np.min(y_pred):.6f}') # [MR]
    print(f'\n| Fold {fold_index:>{digits_K}} | Max  |  {np.max(y_pred):.6f}') # [MR]
    
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
#print("\n> K-fold training (no threading) \nStarting...\n")

# TODO: Maybe replace threading with multiprocessing for thread-safe model
# TODO: Implement ProcessPoolExecutor
with ThreadPoolExecutor(max_workers=K) as executor:
    for fold_index, (train, test) in enumerate(kfold.split(x, decode(y)), start=1):
        executor.submit(process_fold, train, test, fold_index, results_lock)

#for fold_index, (train, test) in enumerate(kfold.split(x, decode(y)), start=1):
#    process_fold(train, test, fold_index)

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

#########################################################################
# [MR] Graphs for analysis

# Accuracy
train_accuracies = []
valid_accuracies = []
# Loss
train_losses     = []
valid_losses     = []

# Join histories from each fold
for fold_index, history in enumerate(histories):
    if history is None: continue
    train_accuracies.append(history["accuracy"])
    train_losses.append(history["loss"])
    if "val_accuracy" in history:
        valid_accuracies.append(history["val_accuracy"])
    if "val_loss" in history:
        valid_losses.append(history["val_loss"])

## Means
# Accuracy
mean_train_accuracies = np.mean(np.array(train_accuracies), axis=0)
mean_valid_accuracies = np.mean(np.array(valid_accuracies), axis=0)
# Loss
mean_train_loss       = np.mean(np.array(train_losses), axis=0)
mean_valid_loss       = np.mean(np.array(valid_losses), axis=0)

## Plot
plt.figure(figsize=(12, 6))

# Accuracy | Train
plt.subplot(2, 2, 1)
plt.plot(mean_train_accuracies, color='red')
plt.title("Accuracy | Train")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

# Loss | Train
plt.subplot(2, 2, 2)
plt.plot(mean_train_loss, color='blue')
plt.title("Loss | Train")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

# Accuracy | Validation
plt.subplot(2, 2, 3)
plt.plot(mean_valid_accuracies, color='red')
plt.title("Accuracy | Validation")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

# Loss | Validation
plt.subplot(2, 2, 4)
plt.plot(mean_valid_loss, color='blue')
plt.title("Loss | Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()