"""
Implementing custom loss and metrics functions for Tensorflow Training/Inference


SED Loss --> Binary Cross Entropy (can just use tf implementation)

DOA Loss --> Masked Regression Loss (need to self implement)

SELD Metrics --> DCASE metrics (need to figure out how to implement)
"""

import keras.backend as K
import numpy as np
import tensorflow as tf
import pandas as pd

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

def custom_metric(y_true, y_pred):

    pass

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