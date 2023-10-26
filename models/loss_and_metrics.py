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

def interpolate_tensor(tensor, ratio: float = 1.0):
    """
    Upsample or Downsample tensor in time dimension
    :param tensor: (batch_size, n_timesteps, n_classes,...) or (batch_size, n_timesteps). Torch tensor.
    :param: ratio. If ratio > 1: upsample, ratio < 1: downsample # ratio = output rate/input rate
    :return: new_tensor (batch_size, n_timestepss*ratio, ...)
    """
    ratio = float(ratio)

    batch_size, n_input_frames = tensor.shape[0], tensor.shape[1]

    # compute indexes
    n_output_frames = int(round(n_input_frames * ratio))
    output_idx = np.arange(n_output_frames)
    input_idx = np.floor(output_idx / ratio).long()

    new_tensor = tensor[:, input_idx]

    return new_tensor

def write_classwise_output(pred_dict, filename):
    """
    :param pred_dict:
    # pred_dict = {
    #     'event_frame_logit': event_frame_logit,
    #     'doa_frame_output': doa_output,
    # }
    """
    doa_frame_output = pred_dict['doa_frame_output'].detach().cpu().numpy()
    doa_frame_output = interpolate_tensor(doa_frame_output, 2.0)
    event_frame_output = pred_dict['event_frame_output'].detach().cpu().numpy()
    event_frame_output = interpolate_tensor(event_frame_output, 2.0)

    # remove batch dimension
    if event_frame_output.shape[0] == 1:  # full file -> remove batch dimension
        event_frame_output = event_frame_output[0]
        doa_frame_output = doa_frame_output[0]

    sed_threshold = 0.3
    n_classes = 12

    # convert sed prediction to binary
    event_frame_output = (event_frame_output >= sed_threshold)

    x = doa_frame_output[:, : n_classes]
    y = doa_frame_output[:, n_classes: 2 * n_classes]
    z = doa_frame_output[:, 2 * n_classes:]

    # convert to polar rad -> polar degree
    # elevation can ignore for now (dataset has no elevation)
    azi_frame_output = np.around(np.arctan2(y, x) * 180.0 / np.pi)
    ele_frame_output = np.around(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180.0 / np.pi)

    # Loop through all the frame
    outputs = []
    for iframe in np.arange(600):  # trim any excessive length
        event_classes = np.where(event_frame_output[iframe] == 1)[0]
        for idx, class_idx in enumerate(event_classes):
            azi = int(azi_frame_output[iframe, class_idx])
            if azi == 180:
                azi = -180
            ele = int(ele_frame_output[iframe, class_idx])
            outputs.append([iframe, class_idx, 0, azi, ele])
    df_columns = ['frame_idx', 'event', 'track_number', 'azimuth', 'elevation']
    submission_df = pd.DataFrame(outputs, columns=df_columns)
    submission_df.to_csv(filename, index=False, header=False)
    pass

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