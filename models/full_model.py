from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import DepthwiseConv2D, AveragePooling2D, Dropout, GRU, Bidirectional, TimeDistributed, Concatenate
from keras.layers import Add, ReLU, Dense
from keras import backend, Model
import tensorflow as tf
from loss_and_metrics import *


"""
Split into sections

1. Encoding functions 
    - Conv blocks
    - ResNet blocks

2. Decoding functions
    - utility functions 
    - decoder modules (BiGRU)
    - fully connected layers for SED and DOA

3. Model Setup 
    - setup model flow and configurations
    - return the model

To-do
-----

Implement ResNet optimizations
- Combination of both DSC and BTN
- Is it possible to combine all the layers into one model class for readability (?)
"""


def conv_block(x, out_channels):
    """
    Basic Convolution Block that is used in the start of the model

    Inputs
    ------
    x               : (np.array) input data
    out_channels    : (int) number of output channels

    Returns
    -------
    x               : (np.array) feature maps of (output channels x _ x _) shape 
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
    x = AveragePooling2D(data_format='channels_first',
                         name = "avg_pool_init")(x)
    
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

    Input
    -----
    x               : (np.array) input data
    out_channels    : (int) number of output channels of the block
    stride          : (int) for SALSA, only the first micro conv has a stride of s
    resnet_style    : (str) one of 'basic' (base), 'bottleneck' or 'dsc' corresponding to 
                            the types of convolutions that all micro-blocks will use

    Returns
    -------
    x               : (np.array) output data
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
    Standard micro resnet block architecture 
        x -> conv -> conv -> adder -> y
        |                      |
        -->-skip connection-->--

    Input
    -----
    x               : (np.array) input data
    out_channels    : (int) number of output channels of the block
    stride          : (s) in this case it is either 1 or 2

    Returns
    ------
    x               : output data
    
    """

    identity = x

    # Stride = 2 actually does an average pooling on x
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
    """Bottleneck blocks works by downsizing the input by factor d, followed by applying
    standard convolution filters on the downsized input. Finally, it will upsample the
    resulting feature maps to a higher number of channels (original or not).
    
    Input
    -----
    x                   : (np.array) input data
    out_channels        : (int) number of output channels of the block
    stride              : (int) in this case, the value `s` is either 1 or 2
    downsample_factor   : (int) the factor at which the inputs will be downsized

    Returns
    ------
    x                   : (np.array) output data
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
    """Similar to the micro_resnet_block, but we replace each 3x3 convolution
    filter with a depthwise convolution, followed by a pointwise convolution
    
    Input
    -----
    x               : (np.array) input data
    out_channels    : (int) number of output channels of the block
    stride          : (int) in this case, `s` is either 1 or 2

    Returns
    ------
    x               : (np.array) output data
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
    """Implementation of the frequency pooling layer

    Input
    -----
    x             : (np.array) input data
    pooling_type  : (str) how we wish to pool the frequency bins, one of 
                            `avg`, `max` or `avg_max`

    Returns
    ------
    x             : (np.array) output data 
    """
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

def bigru_unit(x, add_dense=False):
    """Implementation of the BiGRU decoder

    Unsure if there is a need for `TimeDistributed(Dense(512))` to further 
    refine the BiGRU output
    
    Input
    -----
    x         : (np.array) input data
    add_dense : (boolean) True if adding the `TimeDistributed Dense` layer, False otherwise

    Returns
    -------
    x         : (np.array) output data
    """
    
    # To do : check how to merge the final bigru
    bigru1 = Bidirectional(GRU(units=256, return_sequences=True, name="GRU1"), 
                           merge_mode = 'concat', name="BiGRU1")(x)
    bigru1 = Dropout(0.3)(bigru1)
    bigru2 = Bidirectional(GRU(units=256, return_sequences=True, name="GRU2"), 
                           merge_mode = 'concat', name="BiGRU2")(bigru1)

    # Unsure about the cost/benefits of adding this layer
    if add_dense : bigru2 = TimeDistributed(Dense(256))(bigru2)
    
    return bigru2

def sed_fcn(x, n_classes=3):
    """
    Fully connected layer for SED (multi-label, multi-class classification)

    Inputs
    ------
    x         : (np.array) input data
    n_classes : (int) number of possible event classes

    Returns
    -------

    x         : (np.array) output data of shape (batch_size, time_steps, n_classes)
    """
    
    # default values will do
    x = Dropout(0.2)(x)
    x = Dense(256, activation=None, name = 'sed_fcn1')(x)
    x = ReLU(name = 'sed_relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(n_classes, name='event_frame_logits')(x)
    x = Activation('sigmoid', name='event_pred')(x)

    # (batch_size, time_steps, n_classes)
    return x

def doa_fcn(input, azi_only=False, n_classes=3):
    """
    Fully connected layer for DOA (regression)

    Inputs
    ------
    x         : (np.array) input data
    azi_only  : (boolean) True if only predicting Azimuth (X,Y) , False otherwise
    n_classes : (int) number of possible event classes

    Returns
    -------

    doa_output : (np.array) output data of shape (batch_size, time_steps, 2/3 * n_classes)
    """

    # X-Direction
    x = Dropout(0.2)(input)
    x = Dense(256, name='dense_x')(x)
    x = Dropout(0.2)(x)
    x_out = Dense(n_classes, activation=None, name='x_output')(x)
    x_out = Activation('tanh', name = 'x_tanh')(x_out)

    # Y-Direction
    y = Dropout(0.2)(input)
    y = Dense(256, name='dense_y')(y)
    y = Dropout(0.2)(y)
    y_out = Dense(n_classes, activation=None, name='y_output')(y)
    y_out = Activation('tanh', name = 'y_tanh')(y_out)

    # If Azimuth only, no need the fully connected layer for the Z-direction predictions
    if not azi_only:
        # Z-Direction
        z = Dropout(0.2)(input)
        z = Dense(256, name='dense_z')(z)
        z = Dropout(0.2)(z)
        z_out = Dense(n_classes, activation=None, name='z_output')(z)
        z_out = Activation('tanh', name = 'z_tanh')(z_out)
        
    if azi_only:
        # (batch_size, time_steps, 2 * n_classes)
        doa_output = Concatenate(name="doa_pred")([x_out, y_out])
    else:
        # (batch_size, time_steps, 3 * n_classes)
        doa_output = Concatenate(name='doa_pred')([x_out, y_out, z_out])
    
    return doa_output

def get_model(input_shape, 
              resnet_style='basic', 
              n_classes=12, 
              azi_only = False, 
              batch_size = 1):
    """
    The entire SALSA-Lite model, using Keras functional API to design and flow

    The model should flow as:

    ResNet Layers
    -------------
    2 x (3x3) convolution blocks
    4 x ResNet macro blocks (64,128,256,512 channels)
    2 x BiGRU units
    
    2 Branches (SED, DOA)
    ----------
    1 x FCN + Sigmoid for SED
    3 x FCN + tanh for (x,y,z) DOA

    Inputs
    ------
    input_shape  : (tuple) the shape of the input features
    resnet_style : (str) the type of ResNet to be used in the model
    n_classes    : (int) number of possible event classes 
    azi_only     : (boolean) True if only predicting Azimuth (X,Y) , False otherwise
    batch_size   : (int) batch size. Meant to pass through the Input to the model

    Returns
    -------
    model        : (model) Keras model
    """
    
    # Create input of salsa-lite features
    inputs = Input(shape      = input_shape,
                   batch_size = batch_size,
                   name       = "salsa_lite_features", 
                   sparse     = False)
    
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
    doa_output = doa_fcn(bigru_output, azi_only, n_classes)
    single_array_output = Concatenate(name="final_output")([event_frame_pred, doa_output])

    # Create model 
    model = Model(inputs  = inputs,
                  outputs = single_array_output, 
                  name    = 'full_SALSALITE_model')

    model.compile(optimizer    = tf.keras.optimizers.Adam(learning_rate = 3e-4),
                  loss         = [seld_loss],
                  metrics      = [seld_loss])

    return model


if __name__ == "__main__":

    # Shape of a single input feature
    input_size = (7, 33, 96)

    # use one of basic , bottleneck , dsc for resnet_style
    resnet_style = 'basic'
    n_classes = 3
    azi_only = True
    salsa_lite_model = get_model(input_size, resnet_style, n_classes, azi_only)
    salsa_lite_model.summary(show_trainable=True)

    # To convert and save the model into tflite version 
    convert = False
    if convert: 
        converter = tf.lite.TFLiteConverter.from_keras_model(salsa_lite_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        converter.target_spec._experimental_enable_select_tf_ops = True
        tflite_model = converter.convert()
        filename = "./saved_models/model_{}_selectops_notimedist.tflite".format(resnet_style)
        with open(filename, 'wb') as f:
            f.write(tflite_model)

