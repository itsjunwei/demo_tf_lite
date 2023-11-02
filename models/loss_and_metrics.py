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
from tqdm import tqdm
import os
import math

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

def sed_loss(y_true, y_pred):
    """Similar to the weighted SELD loss, except in this case, we only calculate
    the SED loss. One loss returns one value in TensorFlow so using this plus 
    DOA loss can help us see and track the loss values better

    Returns:
        sed_loss : binary crossentropy loss for SED classification
    """
    n_classes = 3
    sed_pred = y_pred[:, : , :n_classes]
    sed_gt   = y_true[:, : , :n_classes]

    sed_loss = binary_crossentropy(y_true=sed_gt,
                                   y_pred=sed_pred,
                                   from_logits=False)
    
    return sed_loss 

def doa_loss(y_true, y_pred):
    """Similar to the weighted SELD loss, except in this case, we only calculate
    the DOA loss. One loss returns one value in TensorFlow so using this plus 
    SED loss can help us see and track the loss values better

    Returns:
        doa_loss : masked mean absolute error for the loss for DOA regression
    """
    n_classes = 3
    doa_pred = y_pred[:, : , n_classes:]
    
    sed_gt   = y_true[:, : , :n_classes]
    doa_gt   = y_true[:, : , n_classes:]
    
    doa_loss = masked_reg_loss_azimuth(event_frame_gt=sed_gt,
                                       doa_frame_gt=doa_gt,
                                       doa_frame_output=doa_pred,
                                       n_classes=n_classes)
    
    return doa_loss

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

def compute_masked_reg_loss(input, target, mask, loss_type = "MAE"):
    """
    Compute masked regression loss. 
    MAE - mean absolute error (MAE is better for SMN)
    MSE - mean squared error
    
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
    normalize_value = tf.reduce_sum(mask)
    eps = np.finfo(np.float).eps
    # we use keras backend functions to do math
    
    # Formula:
    # sum{ |input - target| * mask } 
    # ______________________________
    # sum{          mask           }

    if loss_type == "MAE":
        reg_loss = tf.reduce_sum(tf.keras.backend.abs(input-target)*mask) / (normalize_value + eps)
    elif loss_type == "MSE":
        reg_loss = tf.reduce_sum((input - target) ** 2 * mask) / (normalize_value + eps)
    else:
        raise ValueError('Unknown reg loss type: {}'.format(loss_type))

    return reg_loss

# not used
def location_dependent_error_rate(y_true, y_pred):
    """Calculate the location dependent error rate for the predictions.
    
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

def remove_batch_dim(tens):
    """Remove the batch dimension from an input tensor or 3D array
    Assumes that the input is of shape (batch_size x frames_per_batch x n_classes)
    
    Combines the batches and returns (frames_total x n_classes)
    """
    # tens : (batch size, frames, n_classes)
    full_frames = tens.shape[0] * tens.shape[1] # combine all batches
    tens = tens.reshape(full_frames, tens.shape[2]) # should be (n_frames_total, n_classes) final
    return tens

def convert_xy_to_azimuth(array, 
                          n_classes=3):
    """Converting an array of X,Y predictions into an array of azimuths.
    [x1, x2, ... , xn, y1, y2, ... , yn] into [azi1, azi2, ... , azin]
    
    Inputs:
        array       : (np.ndarray) An array of X,Y predictions
        n_classes   : (int) `n` or number of possible active classes. Code will
                       manually set n_classes if it is incorrect.
                       
    Returns:
        azimuth_deg : (np.ndarray) Array of azimuths in the range [-180, 180)"""
        
    if not array.shape[-1] == 2*n_classes:
        print("Check  ", array.shape)
        n_classes = array.shape[-1]//2
        print("Manually setting n_classes to be half of last dim, ", n_classes)
    
    x_coords = array[: , :n_classes]
    y_coords = array[: , n_classes:]
    azimuths = np.arctan2(y_coords, x_coords)
    azimuth_deg = np.degrees(azimuths)
    azimuth_deg[azimuth_deg >= 180] -= 360 # Crop the azimuths to be [-180, 180)

    return azimuth_deg    

def get_angular_distance(azimuth_difference):
        """For an input absolute azimuth difference, returns the angular distance
        between the two azimuth predictions
        
        Inputs
            azimuth_difference : Absolute difference between two azimuth values in degrees
        
        Returns
            distance : the calculated angular distance between the two points
        """
        return 180 - abs(azimuth_difference - 180)

class SELDMetrics(object):
    def __init__(self, 
                 model, 
                 val_dataset, 
                 epoch_count, 
                 doa_threshold = 20, 
                 n_classes = 3,
                 sed_threshold = 0.5,
                 n_val_iter = 1000):
        
        # Define self variables 
        self.model = model
        self.val_dataset = val_dataset
        self.epoch_count = epoch_count + 1
        self.n_classes = n_classes
        self.doa_threshold = doa_threshold
        self.sed_threshold = sed_threshold
        self.n_val_iter = n_val_iter
        
        # For SED metrics (F1 score)
        self._TP = 0
        self._FP = 0
        self._FN = 0

        # Substitutions, Deletion and Insertion errors (ER_CD)
        self._S = 0
        self._D = 0
        self._I = 0
        self._Nref = 0
        
        # For DOA metrics
        self.doa_err    = 0 # accumulated doa error for correct SED predictions
        self._TP_count  = 0 # no. of correct SED predictions
        self._DE_FN     = 0 # correct SED predictions but wrong DOA estimate


    def update_seld_metrics(self):
        """Essentialy, this will act as the validation epoch. 
        It will cycle through the dataset generator, compute predictions for each batch
        and update the SELD scores. 
        """
        csv_metrics = []
        # This is for a dataset created using the .from_generator() function
        for x_val, y_val in tqdm(self.val_dataset, total = self.n_val_iter): 
            
            predictions = self.model.predict(x_val, 
                                             verbose = 0)

            # Extract the SED values from the single array
            SED_pred = remove_batch_dim(np.array(predictions[:, :, :self.n_classes]))
            SED_gt   = remove_batch_dim(np.array(y_val[:, :, :self.n_classes])) 
            # If the probability exceeds the threshold --> considered active (set to 1, else 0)
            SED_pred = (SED_pred > self.sed_threshold).astype(int)
                         
            # Extract the DOA values (X,Y) and convert them into azimuth
            azi_gt   = convert_xy_to_azimuth(remove_batch_dim(np.array(y_val[:, : , self.n_classes:])), n_classes = self.n_classes)
            azi_pred = convert_xy_to_azimuth(remove_batch_dim(np.array(predictions[:, : , self.n_classes:])), n_classes = self.n_classes)

            for i in range(len(SED_pred)):
                output = np.concatenate([SED_pred[i], 
                                         SED_gt[i],
                                         azi_pred[i],
                                         azi_gt[i]],
                                        axis = -1)
                csv_metrics.append(output.flatten())
            
            # compute False Negatives (FN), False Positives (FP) and True Positives (TP)
            loc_FN = np.logical_and(SED_gt == 1, SED_pred == 0).sum(1)
            loc_FP = np.logical_and(SED_gt == 0, SED_pred == 1).sum(1)
            TP_sed = np.logical_and(SED_gt == 1, SED_pred == 1)
            # to be considered correct prediction, the DOA difference must be within threshold
            TP_doa = np.abs(azi_gt - azi_pred) < self.doa_threshold 
            loc_TP = np.logical_and(TP_sed, TP_doa).sum() # num of correct class + correct DOA
            
            # Update substitution, deletion and insertion errors
            self._S     += np.minimum(loc_FP, loc_FN).sum()
            self._D     += np.maximum(0, loc_FN - loc_FP).sum()
            self._I     += np.maximum(0, loc_FP - loc_FN).sum()
            self._Nref  += SED_gt.sum() # total number of active classes per batch
            
            # Similarly, update TP, FN and FP 
            self._TP += loc_TP
            self._FN += loc_FN.sum()
            self._FP += loc_FP.sum()
            
            # Class Dependent Localization Error
            correct_cls_doa     = np.multiply(TP_sed, np.abs(azi_gt - azi_pred)) # get DOAs of correct active class predictions
            vectorized_ang_dist = np.vectorize(get_angular_distance) # get the vectorized function to apply to every value in array
            self.doa_err    += vectorized_ang_dist(correct_cls_doa).sum()
            self._TP_count  += TP_sed.sum() # count of correct active class event predictions
            
            # For class-dependent localization recall
            FN_doa = np.abs(azi_gt - azi_pred) > self.doa_threshold # azimuth diff is outside threshold
            loc_FN = np.logical_and(TP_sed, FN_doa).sum() # num of correct SED, wrong DOA
            self._DE_FN += loc_FN # total count of correct SED, wrong DOA
        df = pd.DataFrame(csv_metrics)
        os.makedirs('./temp_metrics', exist_ok = True)
        df.to_csv('./temp_metrics/temp_data.csv', index=False, header=False)
        
    def calculate_seld_metrics(self):
        """Generate the SELD metrics from the calculated values
        Differs from the code provided by DCASE as this is demo-specific
        
        Returns:
            _ER     : (float) Error Rate
            _F1     : (float) F1 Score
            LE_CD   : (float) Localization Error
            LR_CD   : (float) Localization Recall 
        """
        eps = np.finfo(np.float).eps
        
        # ER (localization dependent error rate)
        _ER = (self._S + self._D + self._I) / (self._Nref + eps)
        
        # F1 (localization dependent F1 score)
        _F1 = self._TP / (eps + self._TP + 0.5 * (self._FP + self._FN))

        # LE_CD (class dependent localization error)
        LE_CD = self.doa_err / self._TP_count
        
        # LE_F1 (class dependent localization recall)
        LR_CD = self._TP_count / (eps + self._TP_count + self._DE_FN)

        return _ER, _F1, LE_CD, LR_CD
    
    def calc_csv_metrics(self):
        
        # Read the predictions/gt data file
        data = pd.read_csv('./temp_metrics/temp_data.csv', header=None)
        
        # SED predictions (first n_classes) and gt (second n_classes)
        sed_pred = data.iloc[:, :self.n_classes*2]
        sed = sed_pred.values
        # Mask is essentially just see if predictions == ground truth
        mask = (sed_pred.iloc[: , :self.n_classes].values == sed_pred.iloc[: , self.n_classes:self.n_classes*2].values).all(axis=1)
        # Number of rows/frames where predictions == ground truth
        correct_sed = mask.sum()
        # Raw accuracy for SED
        sed_accuracy = correct_sed / sed_pred.shape[0]
        
        # Extract DOA predictions (n_classes) , ground truths (n_classes)
        doa = data.iloc[: , self.n_classes*2 : ]
        doa = doa.values
        
        c_sed_c_doa = 0 # Num of correct SED & DOA
        c_sed_c_doa_total_doa_error = 0 # DOA error for correct SED & DOA
        lecd_doa_error = 0 # total DOA error for correct SED preds
        total_S = 0
        total_D = 0
        total_I = 0
        total_Nref = 0
        sed_FN = 0
        sed_FP = 0
        
        
        for idx, is_sed in enumerate(mask):
            sed_p = sed[idx][:self.n_classes]
            sed_g = sed[idx][self.n_classes:]
            loc_FN = np.logical_and(sed_g == 1, sed_p == 0).sum()
            loc_FP = np.logical_and(sed_g == 0, sed_p == 1).sum()
            total_S += np.minimum(loc_FP, loc_FN).sum()
            total_D += np.maximum(0, loc_FN - loc_FP).sum()
            total_I += np.maximum(0, loc_FP - loc_FN).sum()
            total_Nref += sed_g.sum()
            sed_FP += loc_FP.sum()
            sed_FN += loc_FN.sum()
            if is_sed: 
                for class_idx, one_hot_class in enumerate(sed_pred[idx]):
                    if one_hot_class != 0:
                        doa_diff = np.abs(doa[idx][class_idx] - doa[idx][class_idx + self.n_classes])
                        lecd_doa_error += doa_diff
                        if doa_diff <= self.doa_threshold:
                            c_sed_c_doa += 1
                            c_sed_c_doa_total_doa_error += doa_diff
                            
        error_rate = (total_S + total_D + total_I) / (total_Nref)
        f_score    = c_sed_c_doa / (c_sed_c_doa + 0.5 * (sed_FN + sed_FP))
        le_cd      = lecd_doa_error / correct_sed
        lr_cd      = c_sed_c_doa/total_Nref
        seld_error = 0.25 * (error_rate + (1-f_score) + le_cd/180 + (1-lr_cd))
        print("Raw Accuracy : {:.4f}".format(sed_accuracy))
        print("SELD Error : {:.3f} , ER : {:.3f} , F1 : {:.3f}, LE : {:.3f}, LR : {:.3f}\n".format(seld_error, error_rate, f_score, le_cd, lr_cd))
     
        