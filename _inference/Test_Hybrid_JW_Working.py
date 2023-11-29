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
from datetime import datetime
import pyaudio
import soundfile
import wave
import numpy as np

now = datetime.now()
now = now.strftime("%Y%m%d_%H%M")

# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)

# Set the path to the folder containing audio files
"""JW : Create a new folder each time just to make sure. Since we are doing demo and not hyperparam
testing this should not incur too much resources"""
audio_folder = "./hybrid_test_{}".format(now)
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

# Global variables
CHUNK = 500
FORMAT = pyaudio.paInt16
CHANNELS = 4
RATE = 48000
WAVE_OUTPUT_FILENAME_PREFIX = os.path.join(audio_folder, "working_files")
MAX_RECORDINGS = 48
INPUT_DEVICE_INDEX = 1
FS = 48000  # Assuming this is the correct sampling rate

# TensorFlow Lite model path
TFLITE_MODEL_PATH = './tflite_models/seld_model_271123/tflite_model.tflite'

# Global variables for TFLite model
with open(TFLITE_MODEL_PATH, 'rb') as fid:
    TFLITE_MODEL_CONTENT = fid.read()

# TFLite model interpreter setup
interpreter = tf.lite.Interpreter(model_content=TFLITE_MODEL_CONTENT)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
n_classes = 4  # Update with the actual number of classe
segment_count = 0
# Lock for thread-safe access to shared resources
lock = threading.Lock()

def record_and_save_audio(audio_folder, duration=0.5):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=INPUT_DEVICE_INDEX)

    frames = []
    global segment_count

    try:
        while True:    
            data = stream.read(CHUNK)
            frames.append(data)
            
            if len(frames) == MAX_RECORDINGS:
                break

        segment_count += 1
        index_no = segment_count % 10
        wave_file_path = WAVE_OUTPUT_FILENAME_PREFIX + str(index_no) + '.wav'
        waveFile = wave.open(wave_file_path, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)

        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
           

    except KeyboardInterrupt:
        print("Streaming stopped by user")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return wave_file_path  # Return the path of the saved audio file

def process_audio_file(audio_path):
    audio_data, _ = librosa.load(audio_path, sr=FS, mono=False, dtype=np.float32)
    feature = extract_features(audio_data)
    feature = np.expand_dims(feature, axis=0)
    feature = np.transpose(feature, [0, 3, 2, 1])

    """JW : In this case, we reset the Interpreter each time we call for a prediction.
    This ensures that we can make sure that the Interpreter is empty each time. This should
    have a resource toll as well so unsure of effects when transfer over to RPI"""
    interpreter = tf.lite.Interpreter(model_content=TFLITE_MODEL_CONTENT)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], feature)
    interpreter.invoke()
    # Get a copy of the output tensor to avoid RuntimeError
    output_data = np.copy(interpreter.get_tensor(output_details[0]['index']))

    with lock:
        sed_pred = remove_batch_dim(np.array(output_data[:, :, :n_classes]))
        sed_pred = apply_sigmoid(sed_pred)
        sed_pred = (sed_pred > 0.5).astype(int)

        azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(output_data[:, :, n_classes:])))
        frame_outputs = []

        for i in range(len(sed_pred)):
            final_azi_pred = sed_pred[i] * azi_pred[i]

            if int(sed_pred[i][-1]) == 1:
                final_azi_pred[-1] = 0

            output = np.concatenate([sed_pred[i], final_azi_pred], axis=-1)
            print(output)
            frame_outputs.append(output.flatten())

        try:# Delete the processed WAV file
            os.remove(audio_path)
        except:
            pass

def process_existing_files(audio_folder, stop_processing):
    while not stop_processing.is_set():
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

        for audio_file in audio_files:
            audio_path = os.path.join(audio_folder, audio_file)
            process_audio_file(audio_path)

def main():
    stop_processing = threading.Event()

    # Create a thread for processing existing files
    processing_thread = threading.Thread(target=process_existing_files, args=(audio_folder, stop_processing))
    processing_thread.start()

    try:
        while True:
            # Record and save audio
            audio_path = record_and_save_audio(audio_folder)
            # Process the recorded audio
            process_audio_file(audio_path)
    except KeyboardInterrupt:
        print("Recording stopped by user")
        stop_processing.set()
        processing_thread.join()

if __name__ == "__main__":
    main()
