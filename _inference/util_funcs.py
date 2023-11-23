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

def local_scaling(x):
    """Scaling an array to fit between -1 and 1"""
    x_min = np.min(x)
    x_max = np.max(x)
    x_normed = (x - x_min) / (x_max - x_min)
    x_scaled = 2 * x_normed - 1
    
    return x_scaled

def apply_sigmoid(x):
    """Sigmoid function to be applied to each element of the array

    Inputs
        x (np.ndarray) : Input array

    Returns
        x (np.ndarray) : Output array where the sigmoid function is applied"""

    return 1 / (1 + np.exp(-x))

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
                          n_classes=4):
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
                 n_classes = 4,
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
        