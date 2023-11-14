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
        - y_pred/true = [sed .... doa]
        - sed -> n_classes
        - doa -> n_classes * 2 (x,y coordinates)
        
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

    n_classes = 3 # hard-coded for the demo
    sed_pred = y_pred[:, : , :n_classes]
    doa_pred = y_pred[:, : , n_classes:]
    
    sed_gt   = y_true[:, : , :n_classes]
    doa_gt   = y_true[:, : , n_classes:]
    
    sed_loss = binary_crossentropy(y_true=sed_gt,
                                   y_pred=sed_pred,
                                   from_logits=False)

    doa_loss = masked_reg_loss_azimuth(event_frame_gt   = sed_gt,
                                       doa_frame_gt     = doa_gt,
                                       doa_frame_output = doa_pred,
                                       n_classes        = n_classes)
    
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
    azimuths = np.around(np.arctan2(y_coords, x_coords) * 180.0 / np.pi)
    azimuths[azimuths == 180] = -180
    
    return azimuths
  

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
        self.n_val_iter = n_val_iter # number of validation iterations (just for tqdm)
        
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
    
    def calc_csv_metrics(self, filepath = None):
        """Generate SELD Metrics from the predictions / ground truth that is calculated from the
        model that is used in this metrics class. Can pass through csv filepath if needed as well.
        
        The metrics are hard coded for the demo itself. Also note that the way that the data is calculated,
        it is not possible for the model to identify multiple instances of the same class in the same 
        timeframe.
        
        Returns
            seld_error : Aggregated SELD Error
            error_rate : Average number of errors per timeframe with an active class
            f_score    : Location-dependent F score
            le_cd      : Class dependent localization error
            lr_cd      : Class dependent localization recall
        """
        
        # To prevent dividing by zero
        eps = np.finfo(np.float).eps
        
        if filepath is not None:
            data = pd.read_csv(filepath, header=None)
        else:
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
                    output = np.concatenate([SED_pred[i], SED_gt[i], azi_pred[i], azi_gt[i]],
                                            axis = -1)
                    csv_metrics.append(output.flatten())
            
            # Create and compile the csv_metrics array    
            data = pd.DataFrame(csv_metrics)

        # SED predictions (first n_classes) and gt (second n_classes)
        sed_pred = data.iloc[:, :self.n_classes*2]
        sed = sed_pred.values # convert to np.ndarry
        # Mask is essentially just see if predictions == ground truth
        mask = (sed_pred.iloc[: , :self.n_classes].values == sed_pred.iloc[: , self.n_classes:self.n_classes*2].values).all(axis=1)
        # Number of rows/frames where predictions == ground truth
        correct_sed = mask.sum()
        # Raw accuracy for SED
        sed_accuracy = correct_sed / sed_pred.iloc[: , self.n_classes:self.n_classes*2].sum()

        # Extract DOA predictions (n_classes) , ground truths (n_classes)
        doa = data.iloc[: , self.n_classes*2 : ]
        doa = doa.values # convert to np.ndarry

        c_sed_c_doa = 0 # Num of correct SED & DOA
        c_sed_c_doa_total_doa_error = 0 # DOA error for correct SED & DOA
        lecd_doa_error = 0 # total DOA error for correct SED preds
        gt_postive_doa_err = []
        gt_negative_doa_err = []
        total_S = 0 # Subtitutions
        total_D = 0 # Deletions
        total_I = 0 # Insertions
        total_Nref = 0 # Total number of frames with active classes
        sed_FN = 0 # Total SED False Negatives
        sed_FP = 0 # Total SED False Positives
        sed_TP = 0 # Total SED True Positivse
        c_sed_w_doa = 0

        # Loop through all predictions, while is_sed == True if it correct prediction for all classes
        for idx in range(len(sed)):
            sed_p   = sed[idx][:self.n_classes] # SED Prediction for a timeframe
            sed_g   = sed[idx][self.n_classes:] # SED Ground Truth for that timeframe
            loc_FN  = np.logical_and(sed_g == 1, sed_p == 0).sum() # False Negatives for the timeframe
            loc_FP  = np.logical_and(sed_g == 0, sed_p == 1).sum() # False Positives for the timeframe
            total_S += np.minimum(loc_FP, loc_FN).sum() # Substitution Error
            total_D += np.maximum(0, loc_FN - loc_FP).sum() # Deletion Error
            total_I += np.maximum(0, loc_FP - loc_FN).sum() # Insertion Error
            total_Nref += sed_g.sum() # Num of active classes per timeframe
            sed_FP  += loc_FP.sum() 
            sed_FN  += loc_FN.sum()
            tp_sed = np.logical_and(sed_p == 1 , sed_g == 1) # e.g. [True, False , True etc]
            sed_TP += tp_sed.sum()
            for class_idx, is_sed in enumerate(tp_sed):
                if is_sed: # correct SED prediction
                    doa_diff = doa[idx][class_idx] - doa[idx][class_idx + self.n_classes] # DOA difference
                    while doa_diff < -180 : doa_diff += 360
                    while doa_diff >= 180 : doa_diff -= 360 # Limit to [-180, 180)
                    doa_diff = np.abs(doa_diff) # Convert to absolute DOA difference
                    lecd_doa_error += doa_diff # sum the total DOA errors for correct SED predictions
                    if doa[idx][class_idx + self.n_classes] >= 0:
                        gt_postive_doa_err.append(doa_diff)
                    elif doa[idx][class_idx + self.n_classes] < 0:
                        gt_negative_doa_err.append(doa_diff)
                    if doa_diff <= self.doa_threshold: # Correct DOA prediction
                        c_sed_c_doa += 1 # number of correct DOA and SED
                        c_sed_c_doa_total_doa_error += np.abs(doa_diff) # DOA diff total for correct DOA and SED
                    else:
                        c_sed_w_doa += 1
                            
        
        # SELD Metrics of Error Rate, F-Score, Localization Error and Recall                    
        error_rate = (total_S + total_D + total_I) / (total_Nref + eps)
        f_score    = c_sed_c_doa / (c_sed_c_doa + 0.5 * (sed_FN + sed_FP) + eps)
        le_cd      = lecd_doa_error / (sed_TP + eps)
        lr_cd      = c_sed_c_doa/(c_sed_c_doa + c_sed_w_doa + eps)
        seld_error = 0.25 * (error_rate + (1-f_score) + le_cd/180 + (1-lr_cd))
        # Raw Accuracy --> Ignore DOA Threshold
        print("Raw Accuracy (ignoring DOA threshold) : {:.2f}%".format(sed_TP*100/(total_Nref + eps)))
        print("DOA Error for Correct Predictions (SED + DOA) : {:.2f}".format(c_sed_c_doa_total_doa_error/(c_sed_c_doa + eps)))
        print("DOA Errors for Positive GT DOA : {:.2f} , Negative GT DOA : {:.2f}".format( sum(gt_postive_doa_err) / (len(gt_postive_doa_err) + eps), sum(gt_negative_doa_err) / (len(gt_negative_doa_err) + eps) ) ) 
        print("SELD Error : {:.4f} , ER : {:.4f} , F1 : {:.4f}, LE : {:.4f}, LR : {:.4f}".format(seld_error, error_rate, f_score, le_cd, lr_cd))
        return seld_error, error_rate, f_score, le_cd, lr_cd
        