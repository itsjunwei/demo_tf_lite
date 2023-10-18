"""
Reference code

Refer to full_model.py

"""

from keras.layers import Input, Conv2D, BatchNormalization
from keras.layers import MaxPool2D, GlobalAvgPool2D, AveragePooling2D, Dropout, GRU, Bidirectional, TimeDistributed, Concatenate
from keras.layers import Add, ReLU, Dense
from keras import backend
from keras import Model

import tensorflow as tf



def frequency_pooling(x, pooling_type='avg'):
    # x = (batchsize, channels, timebins, freqbins)

    # The frequency axis is the last one
    if pooling_type == 'avg':
        x = backend.mean(x , axis=-1)
    elif pooling_type == 'max':
        x = backend.max(x, axis=-1)
    elif pooling_type == 'avg_max':
        x1 = backend.mean(x, axis=-1)
        x2 = backend.max(x, axis=-1)
        x = x1 + x2
    return x

def bigru_unit(x):
    
    bigru1 = Bidirectional(GRU(units=256, dropout=0.3, return_sequences=True))(x)
    bigru2 = Bidirectional(GRU(units=256, dropout=0.3, return_sequences=True), merge_mode='mul')(bigru1)
    bigru2 = TimeDistributed(Dense(512))(bigru2)
    
    return bigru2

def sed_fcn(x, n_classes=12):
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(n_classes)(x)
    return x

def doa_fcn(input, n_classes=12):

    # X-Direction
    x = Dropout(0.2)(input)
    x = Dense(256)(x)
    x = Dropout(0.2)(x)
    x_out = Dense(n_classes, activation='tanh')(x)

    # Y-Direction
    y = Dropout(0.2)(input)
    y = Dense(256)(y)
    y = Dropout(0.2)(y)
    y_out = Dense(n_classes, activation='tanh')(y)

    # Z-Direction
    z = Dropout(0.2)(input)
    z = Dense(256)(z)
    z = Dropout(0.2)(z)
    z_out = Dense(n_classes, activation='tanh')(z)

    doa_output = Concatenate()([x_out, y_out, z_out])
    
    return doa_output

if __name__ == "__main__":
    x = Input(shape=(512,300,11), batch_size=1)

    y = frequency_pooling(x)

    # swap dimension: batch_size, n_timesteps, n_channels/n_features
    z = tf.transpose(y, perm=[0,2,1])
    print(z.shape)
    z = bigru_unit(z)
    print(z.shape)
    event_frame_logits = sed_fcn(z, 12)
    print(event_frame_logits.shape)

    doa_output = doa_fcn(z,12)
    print(doa_output.shape)

