"""
Create feature dataset and corresponding ground truth labels for training/offline usage

Just run the main the file and it will generate the demo dataset in "../dataset/demo_dataset/"
    - demo_salsalite_features.npy
    - demo_gt_labels.npy
"""

import h5py
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def load_file(filepath):
    """
    Extract SALSA-Lite features and ground truth from h5 file
    
    The label rate of the model should be 5fps, we match that accordingly by only producing one label
    per 200ms SALSA-Lite feature (200ms is hardcoded for now)

    Input
    -----
    filepath (str) : filepath to the h5 file

    Returns
    -------
    features    (np.ndarray)  : SALSA-Lite Features
    gt_class    (np.ndarray)  : Active class ground truth
    gt_doa      (np.ndarray)  : Active DOA ground truth [x-directions ... y-directions]
    """
    
    # Read the h5 file
    hf = h5py.File(filepath, 'r')
    features = hf['feature'][:] # features --> n_channels , n_timebins, n_freqbins (7,17,191)

    n_frames_out = int(np.floor(len(features[0])//16))

    # Converting to one-hot encoding
    class_labels = ['dog',
                    'impact',
                    'speech']

    # Extract ground truth from the filename
    filename = filepath.split(os.sep)[-1] # filename  =  class_azimuth_idx.h5
    gts = filename.split('_') # [class, azimuth, ... ]

    # One-hot class encoding
    class_idx = class_labels.index(gts[0])
    gt_class = np.zeros(len(class_labels), dtype=np.float32)
    gt_class[class_idx] = 1
    
    # Converting Azimuth into radians and assigning to active class
    gt_azi = int(gts[1])
    gt_doa = np.zeros(len(class_labels)*2, dtype=np.float32)
    azi_rad = np.deg2rad(gt_azi)
    gt_doa[class_idx] = np.cos(azi_rad)
    gt_doa[len(class_labels) + class_idx] = np.sin(azi_rad)
    
    # Ground Truth should be in the form of 
    # class : onehot encoding
    # doa : azimuths in radians
    gt_class = gt_class.reshape((1,3))
    gt_doa = gt_doa.reshape((1,6))
    
    # Expand it such that it meets the required n_frames output
    gt_class = np.concatenate([gt_class]*n_frames_out, axis=0)
    gt_doa   = np.concatenate([gt_doa]*n_frames_out, axis=0)
    return features , gt_class , gt_doa


def create_dataset():
    """Create dataset

    Returns:
        data            : dataset of features 
        class_labels    : dataset of active class ground truth labels
        doa_labels      : dataset of doa ground truth labels
    """
    data = []
    class_labels = []
    doa_labels = []
    classes = ['dog', 'impact', 'speech']
    for cls in classes:
        feature_dir = '../dataset/features/{}'.format(cls)
        
        # Get the class-wise mean and std.dev to normalize features
        scaler_filepath = '../dataset/features/scalers/{}_feature_scaler.h5'.format(cls)
        with h5py.File(scaler_filepath, 'r') as shf:
            mean = shf['mean'][:]
            std = shf['std'][:]
            
        # Loop through features, add to database
        print("Generating dataset for : ", feature_dir)
        for file in tqdm(os.listdir(feature_dir)):
            if file.endswith('.h5'):
                full_filepath = os.path.join(feature_dir, file)
                salsa_features , cls_label, doa_label = load_file(filepath=full_filepath)
                salsa_features[:4] = (salsa_features[:4]-mean)/std
                data.append(salsa_features)
                
                # Because the n_frames_out per input may not be 1, need to find a way to calculate
                # number of ground truth frames needed per input. For now, hardcoded
                # TO-DO
                class_labels.append(cls_label)
                doa_labels.append(doa_label)
                
    return data, class_labels, doa_labels
        
        
        
if __name__ == "__main__":
    # Ensure that script working directory is same directory as the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    print("Changing directory to : ", dname)
    os.chdir(dname)
    
    # Create arrays for feature, ground truth labels dataset
    d , sed, doa = create_dataset()
    
    # Create directories for storage
    os.makedirs('../dataset/demo_dataset/', exist_ok=True)
    try:
        feature_fp = "../dataset/demo_dataset/demo_salsalite_features.npy"
        if os.path.exists(feature_fp): os.remove(feature_fp)
        np.save(feature_fp, d, allow_pickle=True)
        print("Features saved!")
        
        gt_fp = '../dataset/demo_dataset/demo_class_labels.npy'
        if os.path.exists(gt_fp) : os.remove(gt_fp)
        np.save(gt_fp, sed, allow_pickle=True)
        print("Active class ground truth saved!")
        
        doa_fp = '../dataset/demo_dataset/demo_doa_labels.npy'
        if os.path.exists(doa_fp) : os.remove(doa_fp)
        np.save(doa_fp, doa, allow_pickle=True)
        print("DOA ground truth saved!")
    except:
        print("Error please debug")