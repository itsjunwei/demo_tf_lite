"""
Full training code for the demo dataset

Similar to the ASC code

To do:
    - Fix batch size error with the GRU layer
    - Add logger
    - Global settings into .yml file
"""
from inference_model import * 
import pandas as pd
import numpy as np
import os 
import gc
import tensorflow as tf
from extract_salsalite import extract_features
import time
import librosa

def remove_batch_dim(tens):
    """Remove the batch dimension from an input tensor or 3D array
    Assumes that the input is of shape (batch_size x frames_per_batch x n_classes)
    
    Combines the batches and returns (frames_total x n_classes)
    """
    # tens : (batch size, frames, n_classes)
    full_frames = tens.shape[0] * tens.shape[1] # combine all batches
    tens = tens.reshape(full_frames, tens.shape[2]) # should be (n_frames_total, n_classes) final
    return tens

def convert_xy_to_azimuth(array, 
                          n_classes=3):
    """Converting an array of X,Y predictions into an array of azimuths.
    [x1, x2, ... , xn, y1, y2, ... , yn] into [azi1, azi2, ... , azin]
    
    Inputs:
        array       : (np.ndarray) An array of X,Y predictions
        n_classes   : (int) `n` or number of possible active classes. Code will
                       manually set n_classes if it is incorrect.
                       
    Returns:
        azimuth_deg : (np.ndarray) Array of azimuths in the range [-180, 180)"""
        
    if not array.shape[-1] == 2*n_classes:
        print("Check  ", array.shape)
        n_classes = array.shape[-1]//2
        print("Manually setting n_classes to be half of last dim, ", n_classes)
    
    x_coords = array[: , :n_classes]
    y_coords = array[: , n_classes:]
    azimuths = np.around(np.arctan2(y_coords, x_coords) * 180.0 / np.pi)
    azimuths[azimuths == 180] = -180
    
    return azimuths

# Ensure that script working directory is same directory as the script
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
n_classes = 3
fs = 48000
trained_model_filepath = "./saved_models/bottleneck_w0.4s.h5"

# For JW testing
window_duration_s = 0.5
feature_len = int(window_duration_s * 10 * 16 + 1)

input_shape = (95, feature_len, 7) # Height, Width , Channels shape
print("Input shape : ", input_shape)
# Get the salsa-lite model
salsa_lite_model = get_model(input_shape    = input_shape, 
                             resnet_style   = resnet_style, 
                             n_classes      = n_classes,
                             azi_only       = True,
                             batch_size     = 1)
salsa_lite_model.reset_states() # attempt to fix the stateful BIGRU


"""Load the pre-trained model"""
print("Loading model from : ", trained_model_filepath)
salsa_lite_model.load_weights(trained_model_filepath)

"""Creating and predicting simulated data"""
# audio_fp = "./saved_models/test_0.001var.wav"
# audio_data, _ = librosa.load(audio_fp, sr=fs, mono=False, dtype=np.float32)
# frames = librosa.util.frame(audio_data, 
#                             frame_length=int(window_duration_s*fs), 
#                             hop_length=int(0.5*window_duration_s*fs))
# frames = frames.T
# pred_data = []
# for frame in frames:
#     four_channel = frame.T
#     feature = extract_features(four_channel)
#     feature = np.expand_dims(feature, axis = 0)
#     feature = np.transpose(feature, [0, 3, 2, 1])

#     predictions = salsa_lite_model.predict(feature, verbose=0)
#     sed_pred = remove_batch_dim(np.array(predictions[:, :, :n_classes]))
#     sed_pred = (sed_pred > 0.7).astype(int)  
#     azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(predictions[:, : , n_classes:])))
#     for i in range(len(sed_pred)):
#         output = np.concatenate([sed_pred[i], azi_pred[i]], axis=-1)
#         pred_data.append(output.flatten())
        
# df = pd.DataFrame(pred_data)
# df.to_csv('./test.csv', index=False, header=False)
    
"""Implement streaming into audio reading here"""
# while True:
#     # audio_data = read_mic(...)
#     # features = extract_features(audio_data)
#     # predictions = salsa_lite_model.predict(features)
#     pass

"""JW Testing the model processing speed here"""
iterations = 1000
print("Testing for {} times".format(iterations))
timings = [] # To calculate mean, variance
feature_timings = [] 
for i in range(iterations):
    start_time = time.time()
    features = extract_features(np.random.rand(4,int(window_duration_s * fs))) # Feature shape of input_shape
    features = np.expand_dims(features, axis=0) # Need to expand dims to form batch size = 1
    feat_time = time.time()
    # Going from batch, n_channels, width, height to 
    # batch, height , width, n_channels
    features = np.transpose(features, [0, 3, 2, 1])
    predictions = salsa_lite_model.predict(features, verbose=0) # Get predictions of shape (1, 10 , 9) --> 10fps
    end_time = time.time()
    time_taken = end_time - start_time
    timings.append(time_taken)
    extract_time = feat_time - start_time
    feature_timings.append(extract_time)
    
# Process timings 
print("Mean time taken : {:.4f}s".format(np.mean(timings)))
print("Variance time   : {:.4f}s".format(np.var(timings)))

# Process timings 
print("Feature Extraction mean time : {:.4f}s".format(np.mean(feature_timings)))
print("Feature Extraction variance  : {:.4f}s".format(np.var(feature_timings)))
