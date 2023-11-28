import librosa
from inference_model import *
from util_funcs import * 
import pandas as pd
import numpy as np
import os 
import gc
import tensorflow as tf
from extract_salsalite import extract_features
from datetime import datetime
import pyaudio
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
n_classes = 4
active_classes = ['dog' , 'impact' , 'speech', 'noise']
fs = 48000
trained_model_filepath = "./saved_models/remove_silence_wgn_random_all_aug.h5"


# Set the path to the folder containing audio files
audio_folder = "./hybrid"
os.makedirs(audio_folder, exist_ok=True)

# Get a list of all files in the folder
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

# Iterate through each audio file in the folder
for audio_file in audio_files:
    # Construct the full path to the audio file
    audio_path = os.path.join(audio_folder, audio_file)

    # Read audio data from the file
    audio_data, _ = librosa.load(audio_path, sr=fs, mono=False)


    # Rest of the code remains the same as before
    feature = extract_features(audio_data)
    feature = np.expand_dims(feature, axis=0)
    feature = np.transpose(feature, [0, 3, 2, 1])


    """Load the tflite model"""
    with open('./tflite_models/remove_silence_wgn_random_all_aug/tflite_model.tflite', 'rb') as fid:
        tflite_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    
    
    # Get input and output tensors and TFLite prediction
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], feature)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process output of prediction (rest of the existing code) and Print  the results as needed
    sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
    sed_pred = apply_sigmoid(sed_pred)
    sed_pred = (sed_pred > 0.5).astype(int)  
    azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
    frame_outputs = []
    for i in range(len(sed_pred)):
        final_azi_pred = sed_pred[i] * azi_pred[i]
        if int(sed_pred[i][-1]) == 1:
            final_azi_pred[-1] = 0
        output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
        print(output)
        frame_outputs.append(output.flatten())
    
    # ...
