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
os.system('cls')
print('Screen cleared')
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)

# Global model settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
resnet_style = 'basic'
n_classes = 3
batch_size = 5

# Load dataset
demo_dataset_dir    = "../dataset/demo_dataset"
feature_data_fp     = os.path.join(demo_dataset_dir, 'demo_salsalite_features.npy')
class_label_fp      = os.path.join(demo_dataset_dir, 'demo_class_labels.npy')
doa_label_fp        = os.path.join(demo_dataset_dir, "demo_doa_labels.npy")
feature_dataset     = np.load(feature_data_fp, allow_pickle=True)
class_gt_labels     = np.load(class_label_fp, allow_pickle=True)
doa_gt_labels       = np.load(doa_label_fp, allow_pickle=True)
single_array_list   = np.concatenate((class_gt_labels, doa_gt_labels), axis=-1)

# Create dataset generator 
def dataset_gen():
    for d, l in zip(feature_dataset, single_array_list):
        yield (d,l)

dataset = tf.data.Dataset.from_generator(
    dataset_gen,
    output_signature=(
        tf.TensorSpec(shape = feature_dataset.shape[1:],
                      dtype = tf.float32),
        tf.TensorSpec(shape = single_array_list.shape[1:],
                      dtype = tf.float32)
    )
)

# Get input size of one input 
total_samples = len(feature_dataset)
input_shape = feature_dataset[0].shape

# Get the salsa-lite model
salsa_lite_model = get_model(input_shape=input_shape, 
                             resnet_style=resnet_style, 
                             n_classes=n_classes,
                             azi_only=True,
                             batch_size=batch_size)
# salsa_lite_model.summary()

# Model Training Configurations
checkpoint = ModelCheckpoint("../experiments/salsalite_demo_{epoch:03d}_loss_{loss:.4f}.h5",
                             monitor="val_loss",
                             verbose=1,
                             save_weights_only=False,
                             save_best_only=True)

early = EarlyStopping(monitor="val_loss",
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

callbacks_list = [checkpoint, early, tensorboard_callback, csv_logger, schedule]

x_train , x_test , y_train , y_test = train_test_split(feature_dataset,
                                                       single_array_list,
                                                       test_size=0.2)

x_train, x_val, y_train, y_val      = train_test_split(x_train,
                                                       y_train,
                                                       test_size=0.25)

print("training split size   : {}".format(len(x_train)))
print("validation split size : {}".format(len(x_val)))
print("test split size       : {}".format(len(x_test)))

# Checking if GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # Train model
# demo_model_hist = salsa_lite_model.fit(x_train,
#                                        y_train,
#                                        batch_size=batch_size,
#                                        epochs=1,
#                                        initial_epoch=0,
#                                        verbose=1,
#                                        validation_data=(x_val, y_val),
#                                        callbacks = callbacks_list,
#                                        shuffle=True)

# salsa_lite_model.save_weights('../experiments/model_last.h5')
# np.save('../experiments/demo_model_hist.npy', salsa_lite_model.history, allow_pickle=True)

# # Evaluation 
# salsa_lite_model.load_weights('../experiments/model_last.h5')
# predictions = salsa_lite_model.predict(x_test)
# cls_predictions = [i.flatten() for i in predictions[0]]
# doa_predictions = [j.flatten() for j in predictions[1]]

# # cls_flattened = [cls['event_gt'].flatten() for cls in y_test]
# # doa_flattened = [doa['doa_gt'].flatten() for doa in y_test]
# cls_flattened = [k.flatten() for k in cls_test]
# doa_flattened = [l.flatten() for l in doa_test]

# pd.DataFrame(cls_predictions).to_csv('../experiments/outputs/cls_pred.csv', index=False, header=False)
# pd.DataFrame(doa_predictions).to_csv('../experiments/outputs/doa_pred.csv', index=False, header=False)
# pd.DataFrame(cls_flattened).to_csv('../experiments/outputs/cls_gt.csv', index=False, header=False)
# pd.DataFrame(doa_flattened).to_csv('../experiments/outputs/doa_gt.csv', index=False, header=False)