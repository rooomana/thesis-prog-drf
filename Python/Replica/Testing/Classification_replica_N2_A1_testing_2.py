#########################################################################
#
# Objective: Drone Classification
#
# | Replicating architecture from:
# RF-Based UAV Surveillance System: A Sequential Convolution Neural Networks Approach
#
# Authors: Rubina Akter, Van-Sang Doan, Godwin Brown Tunze, Jae-Min Lee, Dong-Seong Kim
#
#########################################################################

############################## Libraries ################################
import os
import time
import math
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

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
optimizer_loss_fun   = 'categorical_crossentropy'
optimizer_algorithm  = 'adam'
number_epoch         = 60
batch_length         = 10
show_inter_results   = 0

opt = 2     # DNN Results number
current_directory_working = os.getcwd()
results_path = rf"{current_directory_working}\Results_{opt}"

conv_pool_layers = 4
conv_only_layers = 5

filters = [64, 64, 64, 64, 128, 64, 128, 128, 96]
kernel_sizes = [3, 3, 3, 3, 3, 3, 5, 5, 7]

running_time = {}
start_time = time.time()

############################### Loading ##################################
print("\nLoading Data ...")
filepath = os.path.dirname(current_directory_working)
Data = np.loadtxt(rf"{filepath}\Data\RF_Data.csv", delimiter=",")
print("Loaded Data.\n")

############################## Splitting #################################
print("\nPreparing Data ...")
x = np.transpose(Data[0:2047,:])
Label_1 = np.transpose(Data[2048:2049,:]).astype(int)
Label_2 = np.transpose(Data[2049:2050,:]).astype(int)
Label_3 = np.transpose(Data[2050:2051,:]).astype(int)
Label_opt = globals()[f'Label_{opt}']
y = encode(Label_opt)
print("Prepared Data.\n")

# Diagnostics
print("Number of classes (output layer):", y.shape[1])
print("Unique classes in labels:", np.unique(decode(y)))
unique, counts = np.unique(decode(y), return_counts=True)
print("Label distribution:", dict(zip(unique, counts)))

fold_x = x.reshape(x.shape[0], x.shape[1], 1)

############################## Process Fold ##############################
def process_fold(train, test, fold_index, fold_x, y, digits_K, results_path, opt, number_epoch, batch_length, show_inter_results, optimizer_loss_fun, optimizer_algorithm, conv_pool_layers, conv_only_layers, filters, kernel_sizes):
    import time, math, json
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    def decode(datum):
        y = np.zeros((datum.shape[0],1))
        for i in range(datum.shape[0]):
            y[i] = np.argmax(datum[i])
        return y

    start_time_phase = time.time()
    print(f'\n| Fold {fold_index:>{digits_K}} |')

    # Build model
    model = Sequential()
    model.add(layers.Input(shape=(fold_x.shape[1], 1)))
    for i in range(conv_pool_layers):
        model.add(layers.Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], strides=1, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=3))
    for i in range(conv_pool_layers, conv_pool_layers + conv_only_layers):
        model.add(layers.Conv1D(filters=filters[i], kernel_size=kernel_sizes[i], strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(y.shape[1], activation='softmax'))

    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])

    if fold_index == 1:
        print("\n| Summary of the model:")
        model.summary()

    # Train
    history = model.fit(fold_x[train], y[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results, validation_data=(fold_x[test], y[test]))

    # Evaluate
    scores = model.evaluate(fold_x[test], y[test], verbose=show_inter_results)
    print(f'\n| Fold {fold_index:>{digits_K}} | Scores = {scores[1] * 100}')

    # Predict and save
    y_pred = model.predict(fold_x[test])
    results_file = rf"{results_path}\Results_{opt}{fold_index}.csv"
    np.savetxt(results_file, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')

    # Confusion matrix
    decoded_y_test = decode(y[test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(decoded_y_test.flatten(), y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Fold {fold_index} | Confusion Matrix")
    plt.show()

    # Elapsed time
    end_time_phase = time.time()
    elapsed_time_phase = end_time_phase - start_time_phase
    print("\n| Fold %*d | Elapsed time: %.4f seconds\n" % (digits_K, fold_index, elapsed_time_phase))

    return scores[1] * 100, elapsed_time_phase, history.history

################################ Main ####################################
print("\n> K-fold training (w/ multiprocessing) \nStarting...\n")
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
digits_K = math.floor(math.log10(abs(K))) + 1
y_decoded = decode(y)

args_list = []
for fold_index, (train, test) in enumerate(kfold.split(x, y_decoded), start=1):
    args_list.append((
        train, test, fold_index, fold_x, y, digits_K, results_path, opt, number_epoch, batch_length, show_inter_results, optimizer_loss_fun, optimizer_algorithm, conv_pool_layers, conv_only_layers, filters, kernel_sizes
    ))

cvscores = []
fold_histories = []

with ProcessPoolExecutor(max_workers=K) as executor:
    futures = [executor.submit(process_fold, *args) for args in args_list]
    for fold_index, future in enumerate(futures, start=1):
        score, elapsed_time_phase, history = future.result()
        running_time[f'elapsed_time_{fold_index}'] = elapsed_time_phase
        cvscores.append(score)
        fold_histories.append(history)

############################## Elapsed time ##############################
end_time = time.time()
elapsed_time = end_time - start_time
running_time['elapsed_time_total'] = elapsed_time
print("Elapsed time: %.4f seconds\n" % (elapsed_time))

longest_name_length = max(len(name) for name in running_time.keys())
longest_time_length = max(len(str(format(time, '.4f'))) for time in running_time.values())
print('\nRunning Time:\n')
for phase_name, phase_elapsed_time in running_time.items():
    print(f'| {phase_name:<{longest_name_length}} = {phase_elapsed_time:>{longest_time_length}.4f} seconds')
print('\n')

############################## Analysis Graphs ###########################
# Accuracy & Loss
train_accuracies, valid_accuracies = [], []
train_losses, valid_losses = [], []

for history in fold_histories:
    if history is None:
        continue
    train_accuracies.append(history["accuracy"])
    train_losses.append(history["loss"])
    if "val_accuracy" in history:
        valid_accuracies.append(history["val_accuracy"])
    if "val_loss" in history:
        valid_losses.append(history["val_loss"])

# Means
mean_train_accuracies = np.mean(np.array(train_accuracies), axis=0)
mean_valid_accuracies = np.mean(np.array(valid_accuracies), axis=0)
mean_train_loss = np.mean(np.array(train_losses), axis=0)
mean_valid_loss = np.mean(np.array(valid_losses), axis=0)

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(mean_train_accuracies, color='red')
plt.title("Accuracy | Train")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.subplot(2, 2, 2)
plt.plot(mean_train_loss, color='blue')
plt.title("Loss | Train")
plt.ylabel("Loss")
plt.xlabel("Epoch")

plt.subplot(2, 2, 3)
plt.plot(mean_valid_accuracies, color='red')
plt.title("Accuracy | Validation")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.subplot(2, 2, 4)
plt.plot(mean_valid_loss, color='blue')
plt.title("Loss | Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")

plt.tight_layout()
plt.show()