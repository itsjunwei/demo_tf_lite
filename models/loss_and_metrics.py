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

def compute_azimuth_regression_loss(y_true, y_pred):
    
    # azi_gt = (batch_size ,n_time_steps, azimuth)
    
    x_gt = np.cos(y_true)
    y_gt = np.sin(y_true)
    xy_gt = np.concatenate((x_gt, y_gt), axis=-1)
    loss_object = MeanAbsoluteError()
    doa_loss = loss_object(xy_gt, y_pred)
    
    return doa_loss
    
def weighted_seld_loss(y_true, y_pred):
    """
    Generate weighted SELD loss. Keras custom loss functions must only take (y_true, y_pred)
    as parameters. 
    
    Number of classes (n_classes = 3) and weights (0.3 , 0.7) are hard-coded.
    
    The model output has a label rate of 5fps. For the demo, each input is 200ms, which will
    result in one label per input. 

    Args:
        y_true : ground truth
        y_pred : predictions

    Returns:
        seld_loss : weighted seld loss
    """
    # y_true : [[class], [azimuth]]
    # y_pred : [[event_frame_pred], [doa_output]]
    
    n_classes = 3 # hardcoded
    weights = [0.3, 0.7]
    
    sed_gt = y_true[0]
    azi_gt = y_true[1]
    event_frame_pred = y_pred[0]
    doa_output = y_pred[1]
    sed_loss = binary_crossentropy(sed_gt, event_frame_pred)
    doa_loss = masked_reg_loss_azimuth(event_frame_gt=sed_gt,
                                       doa_frame_gt=azi_gt, 
                                       doa_frame_output=doa_output, 
                                       n_classes=n_classes)
    
    seld_loss = weights[0] * sed_loss + weights[1] * doa_loss
    
    return seld_loss


def weighted_loss(target_dict, pred_dict, n_classes=12, loss_weights=[0.3, 0.7]):

    """
    Use this function in the case when the output is in a single dictionary of 
    pred_dict = {
        'event_frame_logit': event_frame_logit,
        'doa_frame_output': doa_output,
    }

    Accordingly, the ground truth has also to be the same dictionary of 
    target_dict = {
        'event_frame_gt' : event_frame_gt,
        'doa_frame_gt' : doa_frame_gt
    }
    """

    sed_weight = loss_weights[0]
    doa_weight = loss_weights[1]

    event_frame_logit = pred_dict['event_frame_logit']
    event_frame_gt = target_dict['event_frame_gt']
    doa_frame_output = pred_dict['doa_frame_output']
    doa_frame_gt = target_dict['doa_frame_gt']

    doa_loss = compute_doa_reg_loss(doa_frame_gt, doa_frame_output, n_classes)
    sed_loss = binary_crossentropy(event_frame_gt, event_frame_logit)

    total_loss = sed_weight*sed_loss + doa_weight*doa_loss

    return total_loss

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
    x_loss = compute_masked_reg_loss(input=doa_frame_output[:, :, : n_classes],
                                     target=doa_frame_gt[:, :, : n_classes],
                                     mask=event_frame_gt)
    y_loss = compute_masked_reg_loss(input=doa_frame_output[:, :, n_classes:2*n_classes],
                                     target=doa_frame_gt[:, :, n_classes:2*n_classes],
                                     mask=event_frame_gt)
    azi_loss = x_loss + y_loss
    return azi_loss

def compute_doa_reg_loss(target_dict, pred_dict, n_classes):
    x_loss = compute_masked_reg_loss(input=pred_dict['doa_frame_output'][:, :, : n_classes],
                                        target=target_dict['doa_frame_gt'][:, :, : n_classes],
                                        mask=target_dict['event_frame_gt'])
    y_loss = compute_masked_reg_loss(input=pred_dict['doa_frame_output'][:, :, n_classes:2*n_classes],
                                        target=target_dict['doa_frame_gt'][:, :, n_classes:2*n_classes],
                                        mask=target_dict['event_frame_gt'])
    z_loss = compute_masked_reg_loss(input=pred_dict['doa_frame_output'][:, :, 2 * n_classes:],
                                        target=target_dict['doa_frame_gt'][:, :, 2 * n_classes:],
                                        mask=target_dict['event_frame_gt'])
    doa_loss = x_loss + y_loss + z_loss

    return doa_loss

def compute_masked_reg_loss(input, target, mask, loss_type="MAE"):
    """
    Compute masked mean loss.
    :param input: batch_size, n_timesteps, n_classes
    :param target: batch_size, n_timestpes, n_classes
    :param mask: batch_size, n_timestpes, n_classes
    :param loss_type: choice: MSE or MAE. MAE is better for SMN
    """
    # Align the time_steps of output and target
    N = min(input.shape[1], target.shape[1])

    input = input[:, 0: N, :]
    target = target[:, 0: N, :]
    mask = mask[:, 0: N, :]

    normalize_value = np.sum(mask)

    if loss_type == 'MAE':
        reg_loss = np.sum(np.abs(input - target) * mask) / normalize_value
    elif loss_type == 'MSE':
        reg_loss = np.sum((input - target) ** 2 * mask) / normalize_value
    else:
        raise ValueError('Unknown reg loss type: {}'.format(loss_type))

    return reg_loss