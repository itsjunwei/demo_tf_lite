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
import librosa
import soundfile as sf
from demo_extract_salsalite import *
import gc

def segment_concat_audio(concat_data_dir = "./data/Dataset_concatenated_tracks/",
                         fs = 24000,
                         window_duration = 0.2,
                         hop_duration = 0.1,
                         add_wgn = False,
                         snr_db = None):
    """Convert the concatenated audio files into segmented audio segments. This is meant to be the step
    prior to conversion to SALSA-Lite Features
    
    Input
        concat_data_dir : (filepath) Where all the concatenated tracks are stored, seperated by subclass folders
        fs              : (int) Sampling Frequency in Hertz (Hz)
        window_duration : (float) Input window duration in seconds
        hop_duration    : (float) Hop/Overlap duration between consecutive windows in seconds
        add_wgn         : (boolean) True if wish to add white Gaussian noise
        snr_db          : (int) Desired level of SNR if WGN is to be added, defaults to 20dB
        
    Returns
        higher_level_dir : (filepath) Directory where all the segmented data will be stored (and seperated
                            by class subfolders)"""

    class_types = ['Dog', 'Impact' , 'Speech']
    # Manual settings 
    frame_len = int(window_duration*fs) 
    hop_len = int(hop_duration*fs)   

    for ct in class_types:
        
        # raw data directory
        full_data_dir = os.path.join(concat_data_dir, ct)
        
        # cleaned, output data directory
        output_data_dir = './_audio/cleaned_data_{}s_{}s/{}'.format(window_duration, hop_duration, ct.lower())
        higher_level_dir = './_audio/cleaned_data_{}s_{}s/'.format(window_duration, hop_duration)
        # create dirs
        os.makedirs(output_data_dir, exist_ok=True)
        print("Storing {} audio in :  {}".format(ct, output_data_dir))
        
        # loop through raw data dir
        for file in os.listdir(full_data_dir):
            if file.endswith('.wav'):
                fullfn = os.path.join(full_data_dir, file)
                
                # extract azimuth from gt
                vars = file.split('_')
                azimuth = vars[2]
                
                # load audio
                audio , _ = librosa.load(fullfn, sr=fs, mono=False, dtype=np.float32)
                
                if add_wgn:
                    signal_power = np.mean(np.abs(audio) ** 2)
                    if snr_db is None:
                        snr_db = 20
                    desired_SNR_dB = snr_db
                    # Calculate the standard deviation of the noise
                    noise_std_dev = np.sqrt(signal_power / (10 ** (desired_SNR_dB / 10)))
                    # Generate the noise
                    noise = np.random.normal(0, noise_std_dev, size=audio.shape)
                    audio += noise
                
                # Segment the audio input into overlapping frames
                frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
                
                # Transpose into (n_segments, timebins, channels)
                frames = frames.T
                for idx, frame in enumerate(tqdm(frames)):
                    final_fn = "{}_{}_{}.wav".format(ct.lower(), azimuth, idx+1)
                    final_fp = os.path.join(output_data_dir, final_fn)
                    sf.write(final_fp, frame, samplerate=fs)
    return higher_level_dir


def load_file(filepath):
    """
    Extract SALSA-Lite features and ground truth from h5 file
    
    The label rate of the model should be 5fps, we match that accordingly by only producing one label
    per 200ms SALSA-Lite feature (200ms is hardcoded for now)

    Input
    -----
    filepath (str) : filepath to the h5 file. Assuming the filename is [class]_[azimuth]_[index].h5

    Returns
    -------
    features    (np.ndarray)  : SALSA-Lite Features
    gt_class    (np.ndarray)  : Active class ground truth
    gt_doa      (np.ndarray)  : Active DOA ground truth [x-directions ... y-directions]
    """
    
    # Read the h5 file
    hf = h5py.File(filepath, 'r')
    features = hf['feature'][:] # features --> n_channels , n_timebins, n_freqbins 

    n_frames_out = int(np.floor(len(features[0])//16)) # Number of timeframes model will output

    # Converting to one-hot encoding
    class_labels = ['dog',
                    'impact',
                    'speech']

    # Extract ground truth from the filename
    filename = filepath.split(os.sep)[-1] # filename  =  class_azimuth_idx.h5
    gts = filename.split('_') # [class, azimuth, ... ]
    
    if gts[0] == "noise": # Everything is zero
        full_gt  = np.zeros(len(class_labels) * 3, dtype = np.float32)
        frame_gt = np.concatenate([full_gt] * n_frames_out, axis = 0)
        return features , frame_gt

    full_gt = np.zeros(len(class_labels) * 3, dtype=np.float32)
    # One-hot class encoding
    class_idx = class_labels.index(gts[0])
    full_gt[class_idx] = 1
    
    # Converting Azimuth into radians and assigning to active class
    gt_azi = int(gts[1])
    # Convert the azimuth from [0, 360) to [-180, 180), taking (0 == 0) and (180 == -180)
    if gt_azi == 330:
        gt_azi = -30
    elif gt_azi == 270:
        gt_azi = -90
    elif gt_azi == 210:
        gt_azi = -150
    azi_rad = np.deg2rad(gt_azi)
    full_gt[class_idx + len(class_labels)] = np.cos(azi_rad) # X-coordinate
    full_gt[class_idx + 2 * len(class_labels)] = np.sin(azi_rad) # Y-coordinate
    
    # Expand it such that it meets the required n_frames output
    frame_gt = np.concatenate([full_gt] * n_frames_out, axis=0)

    return features , frame_gt


def create_dataset(feature_path_dir):
    """Create dataset

    Returns:
        data            : dataset of features 
        class_labels    : dataset of active class ground truth labels
        doa_labels      : dataset of doa ground truth labels
    """
    data = []
    gt_labels = []
    classes = ['dog', 'impact', 'speech']
    for cls in classes:
        feature_dir = os.path.join(feature_path_dir, cls)
        
        # Get the class-wise mean and std.dev to normalize features
        scaler_dir = os.path.join(feature_path_dir, 'scalers')
        scaler_filepath = os.path.join(scaler_dir, '{}_feature_scaler.h5'.format(cls))
        with h5py.File(scaler_filepath, 'r') as shf:
            mean = shf['mean'][:]
            std = shf['std'][:]
            
        # Loop through features, add to database
        print("Generating dataset for : ", feature_dir)
        for file in tqdm(os.listdir(feature_dir)):
            if file.endswith('.h5'):
                full_filepath = os.path.join(feature_dir, file)
                salsa_features , gt_label = load_file(filepath=full_filepath)
                salsa_features[:4] = (salsa_features[:4]-mean)/std
                data.append(salsa_features)
                gt_labels.append(gt_label)
                
    return data, gt_labels
        
        
        
if __name__ == "__main__":
    # Ensure that script working directory is same directory as the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    print("Changing directory to : ", dname)
    os.chdir(dname)
    gc.enable()
    gc.collect()
    
    # Window, Hop duration in seconds 
    ws = 0.2
    hs = 0.1
    
    # # Segment the audio first 
    # audio_upper_dir = segment_concat_audio(window_duration=ws,
    #                                        hop_duration=hs) # './_audio/cleaned_data_{}s_{}s/'.format(window_duration, hop_duration)
    audio_upper_dir = './_audio/cleaned_data_{}s_{}s/'.format(ws, hs)

    # Next, we extract the features for the segmented audio clips
    classes = ['dog', 'impact', 'speech']
    feature_upper_dir = os.path.join('.' , '_features', 'features_{}s_{}s'.format(ws, hs))
    for cls in classes:
        audio_dir = os.path.join(audio_upper_dir, cls)
        feature_dir = os.path.join(feature_upper_dir, cls)
        os.makedirs(os.path.join(feature_upper_dir, 'scalers'), exist_ok=True)
        extract_features(audio_dir, feature_dir)
        compute_scaler(feature_dir, upper_feat_dir=feature_upper_dir)

    # Create arrays for feature, ground truth labels dataset
    data , gt = create_dataset(feature_upper_dir)

    # Create directories for storage
    dataset_dir = "./training_datasets/demo_dataset_{}s_{}s/".format(ws,hs)
    os.makedirs(dataset_dir, exist_ok=True)

    feature_fp = os.path.join(dataset_dir, "demo_salsalite_features.npy")
    np.save(feature_fp, data, allow_pickle=True)
    print("Features saved at {}!".format(feature_fp))
    
    gt_fp = os.path.join(dataset_dir, 'demo_gt_labels.npy')
    np.save(gt_fp, gt, allow_pickle=True)
    print("Active class ground truth saved at {}!".format(gt_fp))
