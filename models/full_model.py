from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import DepthwiseConv2D, AveragePooling2D, Dropout, GRU, Bidirectional, TimeDistributed, Concatenate
from keras.layers import Add, ReLU, Dense
from keras import backend, Model
from keras import optimizers
import tensorflow as tf
from loss_and_metrics import *


"""
Split into 2 main sections

1. Encoding functions 
    - Conv blocks
    - ResNet blocks

2. Decoding functions
    - utility functions 
    - decoder modules (BiGRU)
    - fully connected layers for SED and DOA

    
Main Code

Uses Keras Functional API to design the model flow

To-do
-----

Implement ResNet optimizations
- Combination of both

Training
- Input/Ground Truths
- SELD Metrics
- Optimizations / Momentum / Learning Rate etc. 
"""


def conv_block(x, out_channels):
    """
    Basic Convolution Block that is used in the start of the model, right
    after the input  
    """

    # 2 Convolution subblocks, so we do this process twice
    for subblock in range(2):
        x = Conv2D(filters=out_channels,
                kernel_size=3,
                strides=1,
                padding='same',
                data_format='channels_first',
                use_bias=False,
                kernel_initializer="glorot_uniform",
                name = "conv_init_{}".format(subblock+1)
                )(x)
        x = BatchNormalization(axis=1)(x)
        x = ReLU()(x)

    # Default Keras AveragePooling2D parameters will do
    x = AveragePooling2D(data_format='channels_first')(x)
    
    return x

def resnet_block(x, out_channels, stride, resnet_style='basic'):
    """
    ResNet Blocks consists of 2 microblocks
    For SALSA-Lite, total of 4 "macro" ResNet blocks

    ResNet Block    |   Micro   |   Conv 3x3 (n filters, s stride)
                    |           |   Conv 3x3 (n filters, 1 stride)
                    |           |   Add skip connection
                    ----------------------------------------------
                    |   Micro   |   Conv 3x3 (n filters, 1 stride)
                    |           |   Conv 3x3 (n filters, 1 stride)
                    |           |   Add skip connection 
    """
    # Only basic , bottleneck , depthwise-separable conv implemented
    assert resnet_style in ['basic', 'bottleneck', 'dsc'], "{} not implemented".format(resnet_style)

    if resnet_style == "basic":
        x = micro_resnet_block(x, out_channels, stride)
        x = micro_resnet_block(x, out_channels, 1)
    elif resnet_style == "bottleneck":
        x = micro_bottleneck_block(x, out_channels, stride)
        x = micro_bottleneck_block(x, out_channels, 1)
    elif resnet_style == "dsc":
        x = micro_dsc_block(x, out_channels, stride)
        x = micro_dsc_block(x, out_channels, 1)
    

    return x

def micro_resnet_block(x, out_channels, stride):
    """
    x               : input data
    out_channels    : the number of filters/channels
    stride          : (s) in this case it is either 1 or 2

    Micro architecture 
    - Conv 3x3
    - Conv 3x3
    - Skip Connection
    """

    identity = x

    if stride == 2:
        x = AveragePooling2D(data_format='channels_first')(x)

        identity = AveragePooling2D(data_format='channels_first')(identity)
        identity = Conv2D(out_channels, 
                          kernel_size=1, 
                          strides=1, 
                          data_format='channels_first', 
                          use_bias=False)(identity)
        identity = BatchNormalization(axis=1)(identity)
    
    x = Conv2D(filters=out_channels,
                kernel_size=3,
                strides=1,
                padding='same',
                data_format='channels_first',
                use_bias=False,
                kernel_initializer="glorot_uniform"
                )(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(filters=out_channels,
                kernel_size=3,
                strides=1,
                padding='same',
                data_format='channels_first',
                use_bias=False,
                kernel_initializer="glorot_uniform"
                )(x)
    x = BatchNormalization(axis=1)(x)

    x = Add()([identity, x])
    x = ReLU()(x)
    return x

def micro_bottleneck_block(x, out_channels, stride, downsample_factor = 4):
    """
    Bottleneck blocks works by downsizing the input by factor d
    Doing standard conv. functions on the downsized input
    Upsizing the feature space by factor d (or d') to intended out_channels size
    """
    
    identity = x
    if stride == 2:
        x = AveragePooling2D(data_format='channels_first')(x)

        identity = AveragePooling2D(data_format='channels_first')(identity)
        identity = Conv2D(out_channels, 
                          kernel_size=1, 
                          strides=1, 
                          data_format='channels_first', 
                          use_bias=False)(identity)
        identity = BatchNormalization(axis=1)(identity)
    
    down_channels = int(out_channels/downsample_factor)

    x = Conv2D(filters=down_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                data_format='channels_first',
                use_bias=False,
                kernel_initializer="glorot_uniform"
                )(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)
    
    x = Conv2D(filters=down_channels,
                kernel_size=3,
                strides=1,
                padding='same',
                data_format='channels_first',
                use_bias=False,
                kernel_initializer="glorot_uniform"
                )(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(filters=out_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                data_format='channels_first',
                use_bias=False,
                kernel_initializer="glorot_uniform"
                )(x)
    x = BatchNormalization(axis=1)(x)

    x = Add()([identity, x])
    x = ReLU()(x)

    return x

def micro_dsc_block(x, out_channels, stride):
    """
    Similar to normal convolution block, but each 3x3 convolution is replaced by

    Depthwise -- Pointwise conv
    """

    identity = x

    if stride == 2:
        x = AveragePooling2D(data_format='channels_first')(x)

        identity = AveragePooling2D(data_format='channels_first')(identity)
        identity = Conv2D(out_channels, 
                          kernel_size=1, 
                          strides=1, 
                          data_format='channels_first', 
                          use_bias=False)(identity)
        identity = BatchNormalization(axis=1)(identity)

    
    for i in range(2):
        x = DepthwiseConv2D(kernel_size=3,
                            strides=1,
                            padding='same',
                            data_format='channels_first',
                            use_bias=False)(x)

        x = Conv2D(filters=out_channels,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    data_format='channels_first',
                    use_bias=False,
                    kernel_initializer="glorot_uniform"
                    )(x)
        
        x = BatchNormalization(axis=1)(x)

        if i == 0:
            x = ReLU()(x)
            x = Dropout(0.1)(x)


    x = Add()([identity, x])
    x = ReLU()(x)
    return x

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

    """
    Implementation of the BiGRU decoder

    This one need to double check as unsure of how to merge the bigru decoders
    """
    
    # To do : check how to merge the final bigru
    bigru1 = Bidirectional(GRU(units=256, dropout=0.3, return_sequences=True, name="GRU1"), name="BiGRU1")(x)
    bigru2 = Bidirectional(GRU(units=256, dropout=0.3, return_sequences=True, name="GRU2"), name="BiGRU2")(bigru1)
    bigru2 = TimeDistributed(Dense(512))(bigru2)
    
    return bigru2

def sed_fcn(x, n_classes=12):
    """
    Fully connected layer for SED (multi-label, multi-class classification), without sigmoid
    """
    
    # default values will do
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(n_classes, name='event_frame_logits')(x)
    x = Activation('sigmoid', name='event_frame_pred')(x)

    # (batch_size, time_steps, n_classes)
    return x

def doa_fcn(input, n_classes=12):
    """
    Fully connected layer for DOA (regression)
    """

    # X-Direction
    x = Dropout(0.2)(input)
    x = Dense(256, name='dense_x')(x)
    x = Dropout(0.2)(x)
    x_out = Dense(n_classes, activation='tanh', name='x_output')(x)

    # Y-Direction
    y = Dropout(0.2)(input)
    y = Dense(256, name='dense_y')(y)
    y = Dropout(0.2)(y)
    y_out = Dense(n_classes, activation='tanh', name='y_output')(y)

    # Z-Direction
    z = Dropout(0.2)(input)
    z = Dense(256, name='dense_z')(z)
    z = Dropout(0.2)(z)
    z_out = Dense(n_classes, activation='tanh', name='z_output')(z)

    # (batch_size, time_steps, 3 * n_classes)
    doa_output = Concatenate(name='doa_frame_output')([x_out, y_out, z_out])
    
    return doa_output

def get_model(input_shape, resnet_style='basic', n_classes=12):
    """
    The entire SALSA-Lite model, using Keras functional API to design and flow

    The model should flow as:

    2 x (3x3) convolution blocks
    4 x ResNet macro blocks (64,128,256,512 channels)
    2 x BiGRU units
    1 x FCN + Sigmoid for SED
    3 x FCN + tanh for (x,y,z) DOA
    """
    
    # Create input of salsa-lite features (remove batch_size=1 during training)
    inputs = Input(shape=input_shape, batch_size=1,name = "salsa-lite_features", sparse=False)
    
    # Initial 2 x conv blocks for input
    input = conv_block(inputs, 64)

    # ResNet layers 
    channels = [64,128,256,512]
    strides = [1,2,2,2]
    for i in range(len(channels)):
        input = resnet_block(input, channels[i], strides[i], resnet_style)

    # Decoding layers
    start_decoder = frequency_pooling(input)
    start_decoder = tf.transpose(start_decoder, perm=[0,2,1])
    bigru_output = bigru_unit(start_decoder)

    # Create output
    event_frame_pred = sed_fcn(bigru_output, n_classes)
    doa_output = doa_fcn(bigru_output,n_classes)


    # Create model 
    model = Model(inputs, 
                  {'event_frame_output' : event_frame_pred, 
                   'doa_frame_output' : doa_output}, 
                  name='SALSA_model_test')
    
    # To do : figure out how to configure optimizers
    opt = tf.keras.optimizers.Adam(learning_rate=0.3)

    # To do : custom metrics, loss 
    model.compile(optimizer=opt, 
                  loss={'event_frame_output' : 'binary_crossentropy', 
                        'doa_frame_output' : compute_doa_reg_loss}, 
                  loss_weights = {'event_frame_output' : 0.3,
                                  'doa_frame_output' : 0.7})

    return model


if __name__ == "__main__":

    # Simulate one full minute input for testing
    input_size = (7, 4801, 191)

    # Generate input, output pointers
    # use one of basic , bottleneck , dsc for resnet_style
    resnet_style = 'basic'
    n_classes = 12
    salsa_lite_model = get_model(input_size, resnet_style, n_classes)

    salsa_lite_model.summary(show_trainable=True)

    # To convert and save the model into tflite version 
    convert = True
    if convert: 
        converter = tf.lite.TFLiteConverter.from_keras_model(salsa_lite_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        converter.target_spec._experimental_enable_select_tf_ops = True
        tflite_model = converter.convert()
        filename = "./saved_models/model_{}_selectops.tflite".format(resnet_style)
        with open(filename, 'wb') as f:
            f.write(tflite_model)

