"""
Reference code

Refer to full_model.py

"""

from keras.layers import Input, Conv2D, BatchNormalization
from keras.layers import AveragePooling2D, Dropout
from keras.layers import Add, ReLU

from keras import Model

import tensorflow as tf


"""
Basic Convolution Block that is used in the start of the model, right
after the input  
"""
def conv_block(x, out_channels):
    
    # 2 Convolution subblocks, so we do this process twice
    for subblock in range(2):
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

    # Default Keras AveragePooling2D parameters will do
    x = AveragePooling2D(data_format='channels_first')(x)
    
    return x

"""
Macro ResNet Basic Blocks consists of 2 microblocks

Macro   |   Micro   |   Conv 3x3 (n filters, s stride)
        |           |   Conv 3x3 (n filters, 1 stride)
        |           |   Add skip connection
        ----------------------------------------------
        |   Micro   |   Conv 3x3 (n filters, 1 stride)
        |           |   Conv 3x3 (n filters, 1 stride)
        |           |   Add skip connection 
"""
def resnet_block(x, out_channels, stride):
    x = micro_resnet_block(x, out_channels, stride)
    x = micro_resnet_block(x, out_channels, 1)
    return x

def micro_resnet_block(x, out_channels, stride):
    """
    x               : input data
    out_channels    : the number of filters/channels
    stride          : (s) in this case it is either 1 or 2
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
    return(x)


if __name__ == "__main__":

    input_size = (7, 4801, 191)

    inputs = Input(shape=input_size, batch_size=1)
    a = conv_block(inputs, 64)
    a = resnet_block(a, 64, 1)
    a = resnet_block(a, 128, 2)
    a = resnet_block(a, 256, 2)
    a = resnet_block(a, 512, 2)

    outputs = a


    model = Model(inputs, outputs, name='Conv2D_test')
    model.summary()

