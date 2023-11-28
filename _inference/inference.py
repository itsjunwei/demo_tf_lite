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
import pyaudio
now = datetime.now()
now = now.strftime("%Y%m%d_%H%M")
print("Time now : {}".format(now))
# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)
os.system('cls')
# Clearing the memory seems to improve training speed
gc.collect()
tf.keras.backend.clear_session()

# Global model settings, put into configs / .yml file eventually
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
resnet_style = 'bottleneck'
n_classes = 4
fs = 48000
trained_model_filepath = "./saved_models/seld_model_271123.h5"

"""
Dataset loading functions
"""
# # Load dataset
# demo_dataset_dir    = "./_test_audio/demo_dataset_0.5s_0.25s_NHWC_scaled_with_noise"
# feature_data_fp     = os.path.join(demo_dataset_dir, 'demo_salsalite_features_1000.npy')
# gt_label_fp         = os.path.join(demo_dataset_dir, 'demo_gt_labels_1000.npy')
# print("Features taken from : {}, size : {:.2f} MB".format(feature_data_fp, os.path.getsize(feature_data_fp)/(1024*1024)))
# print("Labels taken from   : {}, size : {:.2f} MB".format(gt_label_fp, os.path.getsize(gt_label_fp)/(1024*1024)))
# feature_dataset     = np.load(feature_data_fp, allow_pickle=True)
# gt_labels           = np.load(gt_label_fp, allow_pickle=True)
# dataset_size        = len(feature_dataset)

# # Create dataset generator 
# def dataset_gen():
#     for d, l in zip(feature_dataset, gt_labels):
#         yield (d,l)

# # Create the dataset class itself
# dataset = tf.data.Dataset.from_generator(
#     dataset_gen,
#     output_signature=(
#         tf.TensorSpec(shape = feature_dataset.shape[1:],
#                       dtype = tf.float32),
#         tf.TensorSpec(shape = gt_labels.shape[1:],
#                       dtype = tf.float32)
#     )
# )
# testing_dataset = dataset.shuffle(buffer_size=1000, seed=2023).take(1000).batch(batch_size = 1)

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
print("Loading model from : ", trained_model_filepath)
salsa_lite_model.load_weights(trained_model_filepath)


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


"""Load the tflite model"""
# with open('./saved_models/tflite_model.tflite', 'rb') as fid:
#     tflite_model = fid.read()

# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

"""Tensorflow predicting re-re-recorded audio data"""
# new_recorded_audio_dir = os.path.join('_test_audio', 'ambisonics_combined')
# new_recorded_audio_dir = os.path.join('_test_audio', 'untrained')
new_recorded_audio_dir = './hybrid'
for new_audio in os.listdir(new_recorded_audio_dir):
    if new_audio.endswith('.wav'):
        print(new_audio)
        audio_fp = os.path.join(new_recorded_audio_dir, new_audio)

        audio_data, _ = librosa.load(audio_fp, sr=fs, mono=False, dtype=np.float32)
        frames = librosa.util.frame(audio_data, 
                                    frame_length=int(window_duration_s * fs), 
                                    hop_length=int(window_duration_s * fs))
        frames = frames.T

        tf_data = []
        for frame in frames:
            four_channel = frame.T
            feature = extract_features(four_channel)
            feature_filename = new_audio.replace('.wav', '.npy')
            np.save(os.path.join('./_features_display', feature_filename), feature, allow_pickle=True)
            feature = np.expand_dims(feature, axis = 0)
            feature = np.transpose(feature, [0, 3, 2, 1])

            output_data = salsa_lite_model.predict(feature,
                                                   verbose=0)
            
            sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
            sed_pred = (sed_pred > 0.3).astype(int)  
            azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
            azi_pred[: , -1] = 0
            for i in range(len(sed_pred)):
                final_azi_pred = sed_pred[i] * azi_pred[i]
                output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
                tf_data.append(output.flatten())

        df = pd.DataFrame(tf_data)
        os.makedirs("./csv_outputs", exist_ok=True)
        df.to_csv("./csv_outputs/test_{}".format(new_audio.replace('.wav', '.csv')), index=False, header=False)



"""TFLite predicting and displaying SALSA-Lite features for all files in folder"""
# new_recorded_audio_dir = os.path.join('_test_audio', 'ambisonics_combined')
# for new_audio in os.listdir(new_recorded_audio_dir):
#     if new_audio.endswith('.wav'):
#         print(new_audio)
#         audio_fp = os.path.join(new_recorded_audio_dir, new_audio)

#         audio_data, _ = librosa.load(audio_fp, sr=fs, mono=False, dtype=np.float32)
#         frames = librosa.util.frame(audio_data, 
#                                     frame_length=int(window_duration_s * fs), 
#                                     hop_length=int(window_duration_s * fs))
#         frames = frames.T

#         tflite_data = []
#         for frame in frames:
#             four_channel = frame.T
#             feature = extract_features(four_channel)
#             feature_filename = new_audio.replace('.wav', '.npy')
#             np.save(os.path.join('./_features_display', feature_filename), feature, allow_pickle=True)
#             feature = np.expand_dims(feature, axis = 0)
#             feature = np.transpose(feature, [0, 3, 2, 1])

#             # TFLite prediction
#             interpreter.set_tensor(input_details[0]['index'], feature)
#             interpreter.invoke()
#             output_data = interpreter.get_tensor(output_details[0]['index'])
            
#             sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
#             sed_pred = (sed_pred > 0.7).astype(int)  
#             azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
#             for i in range(len(sed_pred)):
#                 final_azi_pred = sed_pred[i] * azi_pred[i]
#                 output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
#                 tflite_data.append(output.flatten())

#         df = pd.DataFrame(tflite_data)
#         os.makedirs("./csv_outputs", exist_ok=True)
#         df.to_csv("./csv_outputs/test_{}".format(new_audio.replace('.wav', '.csv')), index=False, header=False)


"""Inference Tests with trained model"""
# seld_metrics = SELDMetrics(model        = salsa_lite_model,
#                            val_dataset  = testing_dataset,
#                            epoch_count  = 1,
#                            n_classes    = n_classes,
#                            n_val_iter   = 1000)
# # Tensorflow predictions + metrics
# seld_error, error_rate, f_score, le_cd, lr_cd = seld_metrics.calc_csv_metrics() # Get the SELD metrics

# TFLite Prediction + Metrics
# tflite_prediction_data = []
# for x_test, y_test in tqdm(testing_dataset, total = 1000):
#     interpreter.set_tensor(input_details[0]['index'], x_test)
#     interpreter.invoke()
#     test_predictions = interpreter.get_tensor(output_details[0]['index'])
    
#     SED_pred = remove_batch_dim(np.array(test_predictions[:, :, :n_classes]))
#     SED_gt   = remove_batch_dim(np.array(y_test[:, :, :n_classes]))
#     SED_pred = (SED_pred > 0.3).astype(int)      
    
#     AZI_gt   = convert_xy_to_azimuth(remove_batch_dim(np.array(y_test[:, : , n_classes:])))
#     AZI_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(test_predictions[:, : , n_classes:])))

#     for i in range(len(SED_pred)):
#         masked_azimuths = SED_pred[i] * AZI_pred[i]
#         output = np.concatenate([SED_pred[i], SED_gt[i], masked_azimuths, AZI_gt[i]], axis=-1)
#         tflite_prediction_data.append(output.flatten())
# df = pd.DataFrame(tflite_prediction_data)
# dfcsv_filepath = "./csv_outputs/inference_test.csv"
# df.to_csv(dfcsv_filepath, index=False, header=False)
# seld_error, error_rate, f_score, le_cd, lr_cd = seld_metrics.calc_csv_metrics(filepath = dfcsv_filepath)


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

    # interpreter.set_tensor(input_details[0]['index'], feature)
    # interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
    # sed_pred = (sed_pred > 0.7).astype(int)  
    # azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
    # for i in range(len(sed_pred)):
    #     final_azi_pred = sed_pred[i] * azi_pred[i]
    #     output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
    #     tflite_data.append(output.flatten())
    

    # predictions = salsa_lite_model.predict(feature, verbose=0)
    # sed_pred = remove_batch_dim(np.array(predictions[:, :, :n_classes]))
    # sed_pred = (sed_pred > 0.7).astype(int)  
    # azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(predictions[:, : , n_classes:])))
    # for i in range(len(sed_pred)):
    #     final_azi_pred = sed_pred[i] * azi_pred[i]
    #     if int(sed_pred[i][-1]) == 1:
    #         final_azi_pred[-1] = 0
    #     output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
    #     print(output)
    #     pred_data.append(output.flatten())

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
# tf_timings = [] # To calculate mean, variance
# feature_timings = [] 
# tflite_timings = []
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
    
#     prediction_start = time.time()
#     predictions = salsa_lite_model.predict(features, verbose=0) # Get predictions of shape (1, 10 , 9) --> 10fps
#     prediction_end = time.time()
    
#     tflite_start = time.time()
#     interpreter.set_tensor(input_details[0]['index'], features)
#     interpreter.invoke()
#     test_predictions = interpreter.get_tensor(output_details[0]['index'])
#     tflite_end = time.time()
    
#     tensorflow_time_taken = prediction_end - prediction_start
#     tf_timings.append(tensorflow_time_taken)
#     extract_time = feat_time - start_time
#     feature_timings.append(extract_time)
#     tflite_time_taken = tflite_end - tflite_start
#     tflite_timings.append(tflite_time_taken)
    
# # Process timings 
# print("Tensorflow inference mean time taken : {:.4f}s".format(np.mean(tf_timings)))
# print("Tensorflow inference variance time   : {:.4f}s".format(np.var(tf_timings)))

# # Process timings 
# print("TFLite inference mean time taken : {:.4f}s".format(np.mean(tflite_timings)))
# print("TFLite inference variance time   : {:.4f}s".format(np.var(tflite_timings)))

# # Process timings 
# print("Feature Extraction mean time : {:.4f}s".format(np.mean(feature_timings)))
# print("Feature Extraction variance  : {:.4f}s".format(np.var(feature_timings)))
