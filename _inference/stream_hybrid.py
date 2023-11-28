
import librosa
from inference_model import *
from util_funcs import * 
import pandas as pd
import numpy as np
import os
import threading
import gc
import tensorflow as tf
from extract_salsalite import extract_features
from datetime import datetime
from queue import Queue
import time
import pyaudio
import soundfile
import numpy as np

# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)
# Clearing the memory seems to improve training speed
gc.collect()
tf.keras.backend.clear_session()


# Global variables
channels = 4  # Number of audio channels
device_index = 1  # Input device index (change to the desired input device)
chunk_size = 500  # Number of frames per chunk
fs = 48000
n_classes = 4

# Function to record audio and save it to the specified folder
def record_and_save_audio(audio_folder, duration=1):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=fs,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size
        
    )

    try:
        while True:
            # Read a chunk of audio data from the stream
            audio_chunk = stream.read(int(fs * duration))
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16).reshape(channels, -1)
            # Save the audio data to a WAV file using soundfile.write
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_filename = os.path.join(audio_folder, f"recording_{timestamp}.wav")
            soundfile.write(audio_filename, audio_data.T, fs)
            
            # Sleep for a short duration to avoid excessive recordings
            time.sleep(1)
    except KeyboardInterrupt:
        print("Recording stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


# Function to process existing WAV files in a folder
def process_existing_files(audio_folder, stop_processing):
    

    try:
        while not stop_processing.is_set():
            
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
                with open('./tflite_models/seld_model_1s_input/tflite_model.tflite', 'rb') as fid:
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
                sed_pred = (sed_pred > 0.3).astype(int)  
                azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, : , n_classes:])))
                frame_outputs = []
                for i in range(len(sed_pred)):
                    final_azi_pred = sed_pred[i] * azi_pred[i]
                    if int(sed_pred[i][-1]) == 1:
                        final_azi_pred[-1] = 0
                    output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
                    frame_outputs.append(output.flatten())
                    
                for idx, val  in enumerate(frame_outputs):
                    print(idx, val)
                    
                


    except KeyboardInterrupt:
        print("Processing stopped by user")



# ... (rest of your code)

# Set the path to the folder containing audio files
audio_folder = "./hybrid"
os.makedirs(audio_folder, exist_ok=True)

# Create a flag to signal the processing thread to stop
stop_processing = threading.Event()

# Create threads for parallel audio recording and processing
record_thread = threading.Thread(target=record_and_save_audio, args=(audio_folder,))
process_thread = threading.Thread(target=process_existing_files, args=(audio_folder, stop_processing))

# Start both threads
record_thread.start()
process_thread.start()

# Wait for user input to stop the script
input("Press Enter to stop the script...\n")

# Signal the processing thread to stop
stop_processing.set()

# Wait for the processing thread to finish
process_thread.join()
