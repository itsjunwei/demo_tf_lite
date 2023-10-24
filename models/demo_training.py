"""
Full training code for the demo dataset

Similar to the 
"""
from loss_and_metrics import *
from full_model import * 
import os 
import sys
import tensorflow as tf
import logging
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split

# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)

# Import relative path modules 
sys.path.append('../')
import dataset 



# Global model settings
resnet_style = 'basic'
n_classes = 3

# Load dataset
demo_dataset_dir    = "../dataset/demo_dataset"
feature_data_fp     = os.path.join(demo_dataset_dir, 'demo_salsalite_features.npy')
class_label_fp      = os.path.join(demo_dataset_dir, 'demo_class_labels.npy')
doa_label_fp        = os.path.join(demo_dataset_dir, "demo_doa_labels.npy")
feature_dataset     = np.load(feature_data_fp, allow_pickle=True)
class_gt_labels     = np.load(class_label_fp, allow_pickle=True)
doa_gt_labels       = np.load(doa_label_fp, allow_pickle=True)

# Get input size of one input 
total_samples = len(feature_dataset)
input_shape = feature_dataset[0].shape

# Get the salsa-lite model
salsa_lite_model = get_model(input_shape=input_shape, 
                             resnet_style=resnet_style, 
                             n_classes=n_classes)
# salsa_lite_model.summary()

# Model Training Configurations
checkpoint = ModelCheckpoint("../experiments/salsalite_demo_{epoch:03d}_loss_{loss:.4f}.h5",
                             monitor="loss",
                             verbose=1,
                             save_weights_only=False,
                             save_best_only=True)

early = EarlyStopping(monitor="loss",
                      mode="min",
                      patience=10)

def scheduler(epoch, lr):
    if epoch < 35:
        lr = 3e-4
        return lr
    else:
        lr = 1e-4
        return lr

schedule = LearningRateScheduler(scheduler)

csv_logger = CSVLogger(filename = '../experiments/training_demo.csv', append=False)
tensorboard_callback = TensorBoard(log_dir='../experiments/logs', histogram_freq=1)

callbacks_list = [checkpoint, early, tensorboard_callback, csv_logger]

# Split the dataset into train (60%) , validation (20%) , test (20%)
x_train , x_test , cls_train , cls_test, doa_train, doa_test  = train_test_split(feature_dataset, 
                                                                                 class_gt_labels, 
                                                                                 doa_gt_labels,
                                                                                 test_size = 0.2, 
                                                                                 random_state=2023)
# 0.25 * 0.8 = 0.2
x_train , x_val , cls_train , cls_val, doa_train, doa_val = train_test_split(x_train, 
                                                                             cls_train, 
                                                                             doa_train,
                                                                             test_size = 0.25, 
                                                                             random_state=2023)

print("training split size   : {}".format(len(x_train)))
print("validation split size : {}".format(len(x_val)))
print("test split size       : {}".format(len(x_test)))

# Checking if GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Checking model input, outputs
print("inputs")


demo_model_hist = salsa_lite_model.fit(x_train,
                                       [cls_train, doa_train],
                                       batch_size=10,
                                       epochs=2,
                                       initial_epoch=0,
                                       verbose=2,
                                       validation_data=(x_val, [cls_val, doa_val]),
                                       callbacks = callbacks_list,
                                       shuffle=True)

salsa_lite_model.save_weights('../experiments/demo_model_hist.npy', salsa_lite_model.history, allow_pickle=True)