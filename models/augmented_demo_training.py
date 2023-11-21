"""
Full training code for the demo dataset

Similar to the ASC code

To do:
    - Fix batch size error with the GRU layer
    - Add logger
    - Global settings into .yml file
"""
from loss_and_metrics import *
from full_model import * 
import pandas as pd
import os 
import gc
import tensorflow as tf
import inference_model as iml
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger, TensorBoard, Callback
from datetime import datetime
from augmentations import freq_mask, random_bool


now = datetime.now()
now = now.strftime("%Y%m%d_%H%M")

# Ensure that script working directory is same directory as the script
os.system('cls')
print('Screen cleared')
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)
# Clearing the memory seems to improve training speed
gc.collect()
tf.keras.backend.clear_session()

# Global model settings, put into configs / .yml file eventually
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
resnet_style = 'bottleneck'
n_classes = 4
batch_size = 32 # fixed because the GRU layer cannot recognise new batch sizes (not sure why)
dataset_split = [0.6, 0.2, 0.2]
total_epochs = 50 # For training


"""
Dataset loading functions
"""
# Load dataset
demo_dataset_dir = "../dataset/training_datasets/demo_dataset_0.5s_0.25s_wgn20"
feature_data_fp  = os.path.join(demo_dataset_dir, 'demo_salsalite_features.npy')
gt_label_fp      = os.path.join(demo_dataset_dir, 'demo_gt_labels.npy')
print("Features taken from : {}, size : {:.2f} MB".format(feature_data_fp, os.path.getsize(feature_data_fp)/(1024*1024)))
print("Labels taken from   : {}, size : {:.2f} MB".format(gt_label_fp, os.path.getsize(gt_label_fp)/(1024*1024)))
feature_dataset  = np.load(feature_data_fp, allow_pickle=True)
gt_labels        = np.load(gt_label_fp, allow_pickle=True)

# Load augmented dataset
augmented_data_fp = os.path.join(demo_dataset_dir, 'augmented_salsalite_features.npy')
augmented_gt_fp   = os.path.join(demo_dataset_dir, 'augmented_salsalite_labels.npy')
print("Augmented features taken from : {}, size : {:.2f} MB".format(augmented_data_fp, os.path.getsize(augmented_data_fp)/(1024*1024)))
print("Augmented labels taken from   : {}, size : {:.2f} MB".format(augmented_gt_fp, os.path.getsize(augmented_gt_fp)/(1024*1024)))
aug_dataset       = np.load(augmented_data_fp, allow_pickle=True)
aug_labels        = np.load(augmented_gt_fp, allow_pickle=True)

# Combine them
combined_dataset = np.concatenate((feature_dataset, aug_dataset), axis=0)
combined_labels  = np.concatenate((gt_labels, aug_labels), axis=0)
dataset_size = len(combined_dataset)

# Create dataset generator 
def dataset_gen():
    for d, l in zip(combined_dataset, combined_labels):
        yield (d,l)

# Create the dataset class itself
dataset = tf.data.Dataset.from_generator(
    dataset_gen,
    output_signature=(
        tf.TensorSpec(shape = combined_dataset.shape[1:],
                      dtype = tf.float32),
        tf.TensorSpec(shape = combined_labels.shape[1:],
                      dtype = tf.float32)
    )
)

# Create another dataset to apply augmentations to
# duplicated_dataset = dataset.shuffle().take(int(0.3 * dataset_size))
# augmented_ds = duplicated_dataset.map(lambda x, y: (freq_mask(x, F=30, num_masks=1, replace_with_zero=True), y))
# dataset = dataset.concatenate(augmented_ds)
# dataset_size = int(len(combined_dataset) * 1.3)

"""# First, filter the dataset to keep only 70% of the images
filtered_ds = ds.filter(lambda image, label: tf.py_function(func=random_bool, inp=[], Tout=tf.bool))

# Then, apply the augmentation to the remaining images
augmented_ds = filtered_ds.map(augment)
ds = ds.concatenate(augmented_ds)"""


# Make a copy of the clean dataset first
non_augmented_dataset = dataset.take(dataset_size)
# Shuffle the dataset first (at each epoch)
shuffled_dataset = dataset.shuffle(dataset_size, reshuffle_each_iteration=True)

# Apply the augmentation of 50% of the data
augmented_dataset = shuffled_dataset.take(int(dataset_size * 0.5))
augmented_dataset = augmented_dataset.map(lambda x, y : (freq_mask(x), y))

# Combine them back together
final_dataset = augmented_dataset.concatenate(non_augmented_dataset)

# Post augmentation
# Get number of training, validation and test samples
dataset_size = int(dataset_size * 1.5) # Update the size of the dataset
train_size = int(dataset_split[0] * dataset_size)
val_size   = int(dataset_split[1] * dataset_size)
test_size  = int(dataset_split[2] * dataset_size)

# Create the training dataset, drop_remainder otherwise will throw error with irregular batch sizes
final_dataset = final_dataset.cache().shuffle(dataset_size, reshuffle_each_iteration=True) # Reshuffle the combined dataset
train_dataset = final_dataset.take(train_size)
train_dataset = train_dataset.batch(batch_size     = batch_size,
                                    drop_remainder = True)

remaining_dataset  = final_dataset.skip(train_size)
validation_dataset = remaining_dataset.take(val_size).batch(batch_size = batch_size, 
                                                            drop_remainder = True)
test_dataset       = remaining_dataset.skip(val_size).batch(batch_size = batch_size, 
                                                            drop_remainder = True)


"""Most dataset input pipelines should end with a call to prefetch. This 
allows later elements to be prepared while the current element is being processed.
Tensorflow says this is best practice so just going to follow"""
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) 
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Get input size of one input 
input_shape = combined_dataset[0].shape
print("Train size  : ", train_size)
print("Val size    : ", val_size)
print("Test size   : ", test_size)
print("Batch size  : ", batch_size)
print("Input shape : ", input_shape)

# Get the salsa-lite model
salsa_lite_model = iml.get_inference_model(input_shape  = input_shape,
                                           resnet_style = resnet_style,
                                           n_classes    = n_classes,
                                           azi_only     = True)
"""Removed batch_size param in the model to allow for TF to infer the batch size itself.
Hopefully fixes the issue"""
salsa_lite_model.reset_states() # attempt to fix the stateful BIGRU
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
class LR_schedule:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
    
    def scheduler(self, epoch, lr):
        """Learning rate schedule should be 3x10^-4 for the first 70% of epochs, and it should reduce to 
        10^-4 for the remaining 30% of epochs. The purpose of this is to help the model converge faster
        during the initial, higher learning rate, as well as to escape the local minima in the loss landscape.
        The reduce learning rate is meant to help the model make smaller, more refined updates to its weights.
        A smaller learning rate towards the end of training can also lead to more stable training, as the
        updates to the weights become smaller."""
        
        if epoch < int(0.7 * self.total_epochs):
            lr = 3e-4
            return lr
        else:
            decay_per_epoch = (3e-4 - 1e-4) / (0.3 * self.total_epochs)
            lr_decay = (epoch - int(0.7 * self.total_epochs)) * decay_per_epoch
            lr = 3e-4 - lr_decay
            return lr

training_lr = LR_schedule(total_epochs = total_epochs)
schedule = LearningRateScheduler(training_lr.scheduler)

csv_logger = CSVLogger(filename = '../experiments/{}/training_demo.csv'.format(now), 
                       append=True)

tensorboard_callback = TensorBoard(log_dir='../experiments/{}/logs'.format(now), 
                                   histogram_freq=1)

# callbacks_list = [checkpoint, early, tensorboard_callback, csv_logger, schedule]
callbacks_list = [tensorboard_callback, csv_logger, schedule] # Removed unneeded callbacks

# Checking if GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Train model
"""Adjusting the code to run in such a way that it will train once per epoch, track the losses
and then predict on the validation set after each training epoch. It will then take the 
predictions and do manual metrics calculation. 

In this case, specifying steps_per_epoch will cause the dataset generator to not run
infinitely (we want it to run once per epoch). So in this case, we do not specify the value.
As such, the progress bar will show xxx/unknown (if verbose = 1) and that is fine. 

TODO
    - Fix batch size limitation (GRU layer)
"""
# Store SELD Metrics
train_stats = []
os.makedirs('../experiments/{}/seld_model/'.format(now), exist_ok=True)

# Training Loop
for epoch_count in range(total_epochs):
    
    demo_model_hist = salsa_lite_model.fit(train_dataset,
                                           epochs           = epoch_count+1,
                                           initial_epoch    = epoch_count,
                                           validation_data  = validation_dataset,
                                           callbacks        = callbacks_list,
                                           verbose          = 1)
    
    seld_metrics = SELDMetrics(model        = salsa_lite_model,
                               val_dataset  = validation_dataset,
                               epoch_count  = epoch_count,
                               n_classes    = n_classes,
                               n_val_iter   = int(val_size//batch_size)) # SELD Metrics class
    
    seld_error, error_rate, f_score, le_cd, lr_cd = seld_metrics.calc_csv_metrics() # Get the SELD metrics
    train_stats.append([epoch_count + 1, seld_error, error_rate, f_score, le_cd, lr_cd]) # Store metrics history
    
    # Check if lowest SELD Error
    min_SELD_error_array = min(train_stats, key = lambda x : x[1])
    if min_SELD_error_array[0] == epoch_count+1 : # Save the epoch model with lowest SELD Error
        best_performing_epoch_path = "../experiments/{}/seld_model/epoch_{}_seld_{:.3f}.h5".format(now, min_SELD_error_array[0], min_SELD_error_array[1])
        print("Best performing epoch : {}, SELD Error : {:.4f}\n".format(min_SELD_error_array[0], min_SELD_error_array[1]))
        salsa_lite_model.save_weights(best_performing_epoch_path, overwrite=True)
    if ((epoch_count - min_SELD_error_array[0]) == 10): 
        print("SELD Error not decreasing any further for 10 epochs. Stopping training now")
        break # Early stopping criterion

# Present the SELD metrics for the best performing model
min_SELD_error_array = min(train_stats, key = lambda x : x[1])
print("\nBest performing epoch : {}, SELD Error : {:.4f}".format(min_SELD_error_array[0], min_SELD_error_array[1]))

# Currently saving the last epoch model and model history
salsa_lite_model.save_weights('../experiments/{}/model_last.h5'.format(now))
np.save('../experiments/{}/demo_model_hist.npy'.format(now), 
        salsa_lite_model.history, 
        allow_pickle=True)

is_inference = True
if is_inference: 
    print("\n\nInfering on test set now...")
    # Inference Section on Test Set
    csv_data = []
    print("Using the model from : {}".format(best_performing_epoch_path))
    salsa_lite_model.load_weights(best_performing_epoch_path)
    for x_test, y_test in tqdm(test_dataset, total = int(test_size//batch_size)):
        test_predictions = salsa_lite_model.predict(x_test, verbose = 0)
        SED_pred = remove_batch_dim(np.array(test_predictions[:, :, :n_classes]))
        SED_gt   = remove_batch_dim(np.array(y_test[:, :, :n_classes]))
        SED_pred = (SED_pred > 0.3).astype(int)      
        
        AZI_gt   = convert_xy_to_azimuth(remove_batch_dim(np.array(y_test[:, : , n_classes:])))
        AZI_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(test_predictions[:, : , n_classes:])))

        for i in range(len(SED_pred)):
            masked_azimuths = SED_pred[i] * AZI_pred[i]
            output = np.concatenate([SED_pred[i], SED_gt[i], masked_azimuths, AZI_gt[i]], axis=-1)
            csv_data.append(output.flatten())
            
    df = pd.DataFrame(csv_data)
    inference_csv_filepath = '../experiments/{}/outputs/test_data.csv'.format(now)
    os.makedirs("../experiments/{}/outputs".format(now), exist_ok = True)
    df.to_csv(inference_csv_filepath, index=False, header=False)
    seld_metrics.calc_csv_metrics(filepath = inference_csv_filepath)
    print("Inference CSV stored at :  {}".format(inference_csv_filepath))
else:
    print("Not inferring!")