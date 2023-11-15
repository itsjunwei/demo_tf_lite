"""
Full training code for the demo dataset

Similar to the ASC code

To do:
    - Fix batch size error with the GRU layer
    - Add logger
    - Global settings into .yml file
"""
from inference_model import *
from util_funcs import * 
import pandas as pd
import numpy as np
import os 
import gc
import tensorflow as tf
from extract_salsalite import extract_features
import time
import librosa
from tqdm import tqdm
from datetime import datetime
now = datetime.now()
now = now.strftime("%Y%m%d_%H%M")

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
trained_model_filepath = "./saved_models/bottleneck_w0.5s_scaled_with_noise.h5"

# For JW testing
window_duration_s = 0.5
feature_len = int(window_duration_s * 10 * 16 + 1) # General FFT formula

input_shape = (95, feature_len, 7) # Height, Width , Channels shape
print("Input shape : ", input_shape)
# Get the salsa-lite model
salsa_lite_model = get_model(input_shape    = input_shape, 
                             resnet_style   = resnet_style, 
                             n_classes      = n_classes,
                             azi_only       = True,
                             batch_size     = 1)


"""Load the pre-trained model"""
# print("Loading model from : ", trained_model_filepath)
# salsa_lite_model.load_weights(trained_model_filepath)


"""Load the tflite model"""
with open('./saved_models/tflite_model.tflite', 'rb') as fid:
    tflite_model = fid.read()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


"""Converting and saving as TFLITE, only need to do this once"""
# salsa_lite_model.save("./saved_models/tf_model")
# converter = tf.lite.TFLiteConverter.from_saved_model("./saved_models/tf_model")
# tflite_model = converter.convert()

# with open('./saved_models/tflite_model.tflite' , 'wb') as f:
#     f.write(tflite_model)
    
# with open('./saved_models/tflite_model.tflite', 'rb') as fid:
#     tflite_model = fid.read()

# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()


"""TFLite creating and predicting simulated data"""
# audio_fp = "./_test_audio/test_add_ambience.wav"
# # audio_fp = r"G:\datasets\testfile_untrain\180degree.wav"
# audio_data, _ = librosa.load(audio_fp, sr=fs, mono=False, dtype=np.float32)
# frames = librosa.util.frame(audio_data, 
#                             frame_length=int(window_duration_s * fs), 
#                             hop_length=int(window_duration_s * fs))
# frames = frames.T

# tflite_data = []
# for frame in frames:
#     four_channel = frame.T
#     feature = extract_features(four_channel)
#     feature = np.expand_dims(feature, axis = 0)
#     feature = np.transpose(feature, [0, 3, 2, 1])

#     # TFLite prediction
#     interpreter.set_tensor(input_details[0]['index'], feature)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
    
#     sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
#     sed_pred = (sed_pred > 0.7).astype(int)  
#     azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
#     for i in range(len(sed_pred)):
#         final_azi_pred = sed_pred[i] * azi_pred[i]
#         output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
#         tflite_data.append(output.flatten())

# df = pd.DataFrame(tflite_data)
# os.makedirs("./csv_outputs", exist_ok=True)
# df.to_csv("./csv_outputs/test_{}.csv".format(now), index=False, header=False)


"""Creating and predicting simulated data"""
# audio_fp = "./_test_audio/test_add_ambience.wav"
# # audio_fp = r"G:\datasets\testfile_untrain\180degree.wav"
# audio_data, _ = librosa.load(audio_fp, sr=fs, mono=False, dtype=np.float32)
# frames = librosa.util.frame(audio_data, 
#                             frame_length=int(window_duration_s * fs), 
#                             hop_length=int(window_duration_s * fs))
# frames = frames.T

# pred_data = []
# tflite_data = []
# for frame in frames:
#     four_channel = frame.T
#     feature = extract_features(four_channel)
#     feature = np.expand_dims(feature, axis = 0)
#     feature = np.transpose(feature, [0, 3, 2, 1])

#     interpreter.set_tensor(input_details[0]['index'], feature)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
#     sed_pred = (sed_pred > 0.7).astype(int)  
#     azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
#     for i in range(len(sed_pred)):
#         final_azi_pred = sed_pred[i] * azi_pred[i]
#         output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
#         tflite_data.append(output.flatten())
    

#     predictions = salsa_lite_model.predict(feature, verbose=0)
#     sed_pred = remove_batch_dim(np.array(predictions[:, :, :n_classes]))
#     sed_pred = (sed_pred > 0.7).astype(int)  
#     azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(predictions[:, : , n_classes:])))
#     for i in range(len(sed_pred)):
#         final_azi_pred = sed_pred[i] * azi_pred[i]
#         output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
#         pred_data.append(output.flatten())

# df = pd.DataFrame(pred_data)
# df.to_csv('./csv_outputs/test_{}.csv'.format(now), index=False, header=False)

# df2 = pd.DataFrame(tflite_data)
# df2.to_csv("./csv_outputs/test_2_{}.csv".format(now), index=False, header=False)
    
    
"""Implement streaming into audio reading here"""
# while True:
#     # audio_data = read_mic(...)
#     # features = extract_features(audio_data)
#     # predictions = salsa_lite_model.predict(features)
#     pass


"""JW Testing the model processing speed here"""
# iterations = 1000
# print("Testing for {} times".format(iterations))
# timings = [] # To calculate mean, variance
# feature_timings = [] 
# for i in range(iterations):
#     random_audio = np.random.rand(4, int(window_duration_s * fs)) # Keep the generated audio out of the timer
    
#     start_time = time.time()
    
#     scaled_random_audio = local_scaling(random_audio) # Scale the audio to fit [-1, 1]
#     features = extract_features(scaled_random_audio) # Feature shape of input_shape
#     features = np.expand_dims(features, axis=0) # Need to expand dims to form batch size = 1
    
#     feat_time = time.time()
    
#     # Going from batch, n_channels, width, height to 
#     # batch, height , width, n_channels
#     features = np.transpose(features, [0, 3, 2, 1])
#     predictions = salsa_lite_model.predict(features, verbose=0) # Get predictions of shape (1, 10 , 9) --> 10fps
    
#     end_time = time.time()
    
#     time_taken = end_time - start_time
#     timings.append(time_taken)
#     extract_time = feat_time - start_time
#     feature_timings.append(extract_time)
    
# # Process timings 
# print("Mean time taken : {:.4f}s".format(np.mean(timings)))
# print("Variance time   : {:.4f}s".format(np.var(timings)))

# # Process timings 
# print("Feature Extraction mean time : {:.4f}s".format(np.mean(feature_timings)))
# print("Feature Extraction variance  : {:.4f}s".format(np.var(feature_timings)))
