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
resnet_style = 'basic'
n_classes = 3
input_shape = (7, 161, 95)
# Get the salsa-lite model
salsa_lite_model = get_model(input_shape    = input_shape, 
                             resnet_style   = resnet_style, 
                             n_classes      = n_classes,
                             azi_only       = True,
                             batch_size     = 1)
salsa_lite_model.reset_states() # attempt to fix the stateful BIGRU


"""Load the pre-trained model"""
trained_model_filepath = "./saved_models/test_model.h5"
salsa_lite_model.load_weights(trained_model_filepath)


# """Implement streaming into audio reading here"""
# while True:
#     # audio_data = read_mic(...)
#     # features = extract_features(audio_data)
#     # predictions = salsa_lite_model.predict(features)
#     pass

"""JW Testing the model processing speed here"""
iterations = 100
timings = [] # To calculate mean, variance
for i in range(iterations):
    start_time = time.time()
    features = extract_features(np.random.rand(4,48000)) # Feature shape of input_shape
    features = np.expand_dims(features, axis=0) # Need to expand dims to form batch size = 1
    predictions = salsa_lite_model.predict(features, verbose=0) # Get predictions of shape (1, 10 , 9) --> 10fps
    end_time = time.time()
    time_taken = end_time - start_time
    timings.append(time_taken)
    
# Process timings 
print("Mean time taken : {:.4f}s".format(np.mean(timings)))
print("Variance time   : {:.4f}s".format(np.var(timings)))
