"""
Full training code for the demo dataset

Similar to the ASC code

To do:
    - Fix batch size error with the GRU layer
    - Add noise data to the mix and test again 
    - Add logger
    - Global settings into .yml file
"""
from loss_and_metrics import *
from full_model import * 
import pandas as pd
import os 
import tensorflow as tf
import logging
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger, TensorBoard

# Ensure that script working directory is same directory as the script
os.system('cls')
print('Screen cleared')
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)

# Global model settings, put into configs / .yml file eventually
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
resnet_style = 'basic'
n_classes = 3
batch_size = 10
dataset_split = [0.6, 0.2, 0.2]


"""
Dataset loading functions
"""
# Load dataset
demo_dataset_dir    = "../dataset/demo_dataset"
feature_data_fp     = os.path.join(demo_dataset_dir, 'demo_salsalite_features.npy')
class_label_fp      = os.path.join(demo_dataset_dir, 'demo_class_labels.npy')
doa_label_fp        = os.path.join(demo_dataset_dir, "demo_doa_labels.npy")
feature_dataset     = np.load(feature_data_fp, allow_pickle=True)
class_gt_labels     = np.load(class_label_fp, allow_pickle=True)
doa_gt_labels       = np.load(doa_label_fp, allow_pickle=True)
single_array_list   = np.concatenate((class_gt_labels, doa_gt_labels), axis=-1)
dataset_size        = len(feature_dataset)

# Create dataset generator 
def dataset_gen():
    while True:
        for d, l in zip(feature_dataset, single_array_list):
            yield (d,l)

# Create the dataset class itself
dataset = tf.data.Dataset.from_generator(
    dataset_gen,
    output_signature=(
        tf.TensorSpec(shape = feature_dataset.shape[1:],
                      dtype = tf.float32),
        tf.TensorSpec(shape = single_array_list.shape[1:],
                      dtype = tf.float32)
    )
)

# Get number of training, validation and test samples
train_size = int(dataset_split[0] * dataset_size)
val_size   = int(dataset_split[1] * dataset_size)
test_size  = int(dataset_split[2] * dataset_size)

# Shuffle the dataset before splitting
shuffled_dataset = dataset.shuffle(dataset_size)

# Create the training dataset
train_dataset = shuffled_dataset.take(train_size)
train_dataset = train_dataset.batch(batch_size=batch_size)

"""Most dataset input pipelines should end with a call to prefetch. This 
allows later elements to be prepared while the current element is being processed.
Tensorflow says this is best practice so just going to follow"""
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) 
remaining_dataset = shuffled_dataset.skip(train_size)

# Create validation dataset
validation_dataset = remaining_dataset.take(val_size)
validation_dataset = validation_dataset.batch(batch_size=batch_size)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

# Create test dataset
test_dataset = remaining_dataset.skip(val_size)
test_dataset = test_dataset.batch(batch_size=batch_size)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Get input size of one input 
total_samples = len(feature_dataset)
input_shape = feature_dataset[0].shape
print("Train size   : ", train_size)
print("Val size     : ", val_size)
print("Test size    : ", test_size)
print("Batch size   : ", batch_size)
print("Input shape  : ", input_shape)

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
                             verbose=2,
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

# callbacks_list = [checkpoint, early, tensorboard_callback, csv_logger, schedule]
callbacks_list = [tensorboard_callback, csv_logger, schedule]

# Checking if GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Train model
"""Adjusting the code to run in such a way that it will train once per epoch, track the losses
and then predict on the validation set after each training epoch. It will then take the 
predictions and do manual metrics calculation. To track val_loss, need to include the validation
dataset into model.fit(). 

In this case, specifying steps_per_epoch will cause the dataset generator to not run
infinitely (we want it to run once per epoch). So in this case, we do not specify the value.
The first epoch will show xxx/unknown for the progress bar (if verbose = 1) and that is fine. 
"""
total_epochs = 10
train_stats = np.zeros(total_epochs)
for epoch_count in range(total_epochs):
    
    demo_model_hist = salsa_lite_model.fit(train_dataset,
                                           epochs           = epoch_count+1,
                                           initial_epoch    = epoch_count,
                                           validation_data  = validation_dataset,
                                           callbacks        = callbacks_list,
                                           verbose          = 2,
                                           max_queue_size   = 50,
                                           workers          = 4) # testing workers param
    
    seld_metrics = SELDMetrics(model        = salsa_lite_model,
                               val_dataset  = validation_dataset,
                               epoch_count  = epoch_count,
                               n_classes    = n_classes)
    
    seld_metrics.update_seld_metrics()
    er_sed , sed_F1 , loc_err , loc_F1 = seld_metrics.calculate_seld_metrics()
    seld_err = 0.25 * (er_sed + (1 - sed_F1) + (loc_err/180) + (1-loc_F1))
    train_stats[epoch_count] = [seld_err, er_sed, sed_F1, loc_err, loc_F1]
    print("SELD Error : {:.3f} , ER : {:.3f} , F1 : {:.3f}, LE : {:.3f}, LR : {:.3f}".format(seld_err, er_sed, sed_F1, loc_err, loc_F1))

salsa_lite_model.save_weights('../experiments/model_last.h5')
np.save('../experiments/demo_model_hist.npy', salsa_lite_model.history, allow_pickle=True)

testing = False
if testing: 
    # Inference Section on Test Set
    csv_data = []
    for x_test, y_test in test_dataset:
        test_predictions = salsa_lite_model.predict(x_test, verbose = 0)
        SED_pred = remove_batch_dim(np.array(test_predictions[:, :, :n_classes]))
        SED_gt   = remove_batch_dim(np.array(y_test[:, :, :n_classes]))
        SED_pred = (SED_pred > 0.3).astype(int)      
        
        AZI_gt   = convert_xy_to_azimuth(remove_batch_dim(np.array(y_test[:, : , n_classes:])))
        AZI_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(test_predictions[:, : , n_classes:])))

        for i in range(len(SED_pred)):
            output = np.concatenate([SED_pred[i], AZI_pred[i], SED_gt[i], AZI_gt[i]], axis=-1)
            csv_data.append(output.flatten())
            
    df = pd.DataFrame(csv_data, columns = ['SED_pred', 'Azimuth_pred', 'SED_gt', 'Azimuth_gt'])
    df.to_csv('../experiments/outputs/test_data.csv', index=False, header=True)