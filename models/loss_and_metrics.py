"""
Implementing custom loss and metrics functions for Tensorflow Training/Inference


SED Loss --> Binary Cross Entropy (can just use tf implementation)

DOA Loss --> Masked Regression Loss (need to self implement)

SELD Metrics --> DCASE metrics (need to figure out how to implement)
"""

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy 
import pandas as pd
from keras.losses import Loss, MeanAbsoluteError

def seld_loss(y_true, y_pred):
    """
    Assuming we our model takes in one input and outputs one concatenated array
        - [sed_pred .... doa_pred]
        - sed_pred -> n_classes
        - doa_pred -> n_classes * 2 (x,y coordinates)
        
    From there, we calculate each loss seperately and compute a weighted loss
    
    Also noted that Tensorflow loss function is defined by y_true, y_pred in that order
    
    Returns:
        seld_loss : weighted seld_loss
        
    To do:
        Is it possible to have external configuration parameters passed through the loss
        function instead of hard-coding self-defined variables? 
    """

    # y_pred and y_true shapes are the same
    # shape : (batch_size, n_timesteps, n_classes * 3) 
    # first n_classes --> SED output
    # remaining --> DOA output for [x * n_classes ... y * n_classes]

    n_classes = 3
    sed_pred = y_pred[:, : , :n_classes]
    doa_pred = y_pred[:, : , n_classes:]
    
    sed_gt   = y_true[:, : , :n_classes]
    doa_gt   = y_true[:, : , n_classes:]
    
    sed_loss = binary_crossentropy(y_true=sed_gt,
                                   y_pred=sed_pred,
                                   from_logits=False)

    doa_loss = masked_reg_loss_azimuth(event_frame_gt=sed_gt,
                                       doa_frame_gt=doa_gt,
                                       doa_frame_output=doa_pred,
                                       n_classes=n_classes)
    
    # hardcoded for now
    weights = [0.3, 0.7]
    loss = weights[0] * sed_loss + weights[1] * doa_loss
    
    return loss

def masked_reg_loss_azimuth(event_frame_gt, doa_frame_gt, doa_frame_output, n_classes):
    """
    Higher function to calculate regression loss for azimuth predictions.
    
    Will calculate the loss for each X and Y coordinate predictions and add them
    
    Inputs:
        event_frame_gt      : (tensor) ground truth for the sound event detection, shape : (batch_size, n_timesteps, n_classes)
        doa_frame_gt        : (tensor) ground truth for DOA, shape : (batch_size, n_timesteps, n_classes*2)
        doa_frame_output    : (tensor) DOA predictions, shape : (batch_size, n_timesteps, n_classes*2)
        n_classes           : (int) number of possible active event classes  
        
    Returns:
        azi_loss    : (float) regression loss for the azimuth predictions
    """
    
    x_loss = compute_masked_reg_loss(input=doa_frame_output[:, :, : n_classes],
                                     target=doa_frame_gt[:, :, : n_classes],
                                     mask=event_frame_gt)
    y_loss = compute_masked_reg_loss(input=doa_frame_output[:, :, n_classes:2*n_classes],
                                     target=doa_frame_gt[:, :, n_classes:2*n_classes],
                                     mask=event_frame_gt)
    azi_loss = x_loss + y_loss
    
    return azi_loss

def compute_masked_reg_loss(input, target, mask):
    """
    Compute masked mean loss. Currently, only masked mean absolute error is implemented
    
    Inputs:
        note that all inputs are of shape (batch_size, n_timesteps, n_classes)
        input   : DOA predictions, usually of only one coordinate
        target  : DOA ground truth
        mask    : Active event classes 
        
    Returns:
        reg_loss : masked regression loss
    """    
    # Align the time_steps of output and target
    N = min(input.shape[1], target.shape[1])

    input = input[:, 0: N, :]
    target = target[:, 0: N, :]
    mask = mask[:, 0: N, :]

    # we use keras backend functions to do math
    
    # Formula:
    # sum{ |input - target| * mask } 
    # ______________________________
    # sum{          mask           }

    reg_loss = tf.keras.backend.sum(tf.keras.backend.abs(input-target)*mask)/tf.keras.backend.sum(mask)
    
    return reg_loss

def location_dependent_accuracy(y_true, y_pred):
    """Calculate the location dependent accuracy for the predictions.
    
    To classify as a correct prediction, the active class must be within a certain,
    predefined range of error. This parameter is hard-coded for now as 10 degrees.
    """
    # hard coded value to split the array
    n_classes = 3
    
    sed_pred = y_pred[:, : , :n_classes]
    doa_x_pred = y_pred[:, : , n_classes:n_classes*2]
    doa_y_pred = y_pred[:, : , n_classes*2:]
    
    sed_gt   = y_true[:, : , :n_classes]
    doa_x_gt = y_true[:, : , n_classes:n_classes*2]
    doa_y_gt = y_true[:, : , n_classes*2:]

    
    # Generate the one-hot output for the SED predictions
    class_indicies      = tf.argmax(sed_pred, axis=-1)
    one_hot_predictions = tf.one_hot(class_indicies, depth=sed_pred.shape[-1])
    
    azimuth_gt      = tf.math.atan2(doa_y_gt, doa_x_gt)
    azimuth_pred    = tf.math.atan2(doa_y_pred, doa_x_pred)
    
    TP = 0
    for idx in range(len(sed_gt)):
        if tf.reduce_all(tf.math.equal(one_hot_predictions[idx], sed_gt[idx])):
            doa_difference = tf.keras.backend.abs(azimuth_gt[idx] - azimuth_pred[idx])
            masked_doa_difference = tf.multiply(sed_gt[idx], doa_difference)
            difference_tensor = tf.reduce_max(input_tensor = masked_doa_difference,
                                             axis = -1)
            difference_value = tf.get_static_value(difference_tensor)
            try:
                if difference_value <= 10:
                    TP += 1
            except:
                pass
    er_cd = 1 - (TP/len(sed_gt))
    
    return er_cd