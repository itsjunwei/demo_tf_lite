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
trained_model_filepath = "./saved_models/seld_model_271123.h5"

"""Create the salsa-lite model class to load the weights into"""
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

"""Load the tflite model"""
with open('./tflite_models/seld_model_271123/tflite_model.tflite', 'rb') as fid:
    tflite_model = fid.read()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

"""Implement streaming into audio reading here"""
channels = 4  # Number of audio channels
device_index = 1  # Input device index (change to the desired input device)
chunk_size = 500 # Number of frames per chunk
buffer_size = 48 # Number of chunks to accumulate in the buffer

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open the audio input stream with the specified number of channels
stream = audio.open(
    format=pyaudio.paInt16,  # 16-bit PCM format
    channels=channels,  # Set to 4 channels
    rate=fs,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=chunk_size
)

print("Streaming audio...")

# Initialize a buffer to accumulate audio data
audio_buffer = []

try:
    while True:
        # Read a chunk of audio data from the stream
        audio_chunk = stream.read(chunk_size)

        # Convert the audio chunk into a NumPy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_data = audio_data = audio_data.astype(np.float32) / 32760.0
        audio_data = audio_data.reshape(channels, -1)

        # Append the chunk to the buffer
        audio_buffer.append(audio_data)

        # If the buffer is full, process its contents
        if len(audio_buffer) == buffer_size:
            # Stack the chunks in the buffer to create a larger data array
            audio_data = np.concatenate(audio_buffer, axis=1)
            feature = extract_features(audio_data)
            feature = np.expand_dims(feature, axis = 0)
            feature = np.transpose(feature, [0, 3, 2, 1])

            # TFLite prediction
            interpreter.set_tensor(input_details[0]['index'], feature)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Process output of prediction
            sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
            sed_pred = apply_sigmoid(sed_pred)
            sed_pred = (sed_pred > 0.3).astype(int)  
            azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
            frame_outputs = []
            for i in range(len(sed_pred)):
                final_azi_pred = sed_pred[i] * azi_pred[i]
                if int(sed_pred[i][-1]) == 1:
                    final_azi_pred[-1] = 0
                output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
                # print(output)
                frame_outputs.append(output.flatten())
            
            # Anything beyond this point is just my formating for the output data for personal testing
            # replace as per demo requirements
            frame_outputs = np.array(frame_outputs)
            averaged_output = np.mean(frame_outputs, axis=0)
            any_class = 0
            out_string = ""
            for j in range(3):
                if averaged_output[j] > 0.2:
                    out_string += "{} : {}, ".format(active_classes[j], averaged_output[j+n_classes])
                    any_class += 1
            if any_class == 0:
                print("No class present")
            else:
                print("Class    |   Average Azimuth")
                print(out_string)
                
            # Clear the buffer
            audio_buffer = []
except KeyboardInterrupt:
    print("Streaming stopped by user")
    
    
stream.stop_stream()
stream.close()
audio.terminate()