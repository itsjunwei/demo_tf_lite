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
import gc
import yaml
from sklearn import preprocessing

def full_feature_with_norm(audio_dir,
                           feature_dir,
                           cfg = None,
                           data_config: str = './configs/salsa_lite_demo_3class.yml'
                           ):
    """
    Extract SALSA-Lite features from a segment of audio 
    SALSA-Lite consists of:
        - 4 Log-linear spectrograms 
        - 3 Normalized Interchannel Phase Differences 

    We also perform the scaling (normalization) of the 4 Log-Power linear spectrograms locally,
    without calculating the global mean or standard deviation

    Arguments
    -----------
    audio_dir (directory)   : folder where all the audio data is stored
    feature_dir (directory) : folder where all the features should be stored
    cfg (array)             : array of configuration parameters (from .yml file)
    data_config (.yml file) : filepath to the .yml config file used for SALSA-Lite

    Returns
    --------
    None 

    To-do
    ------
    """
    if cfg is None:
        # Load data config files
        with open(data_config, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    # Create Feature Directory
    os.makedirs(feature_dir, exist_ok=True)
    
    # Parse config file
    fs = cfg['data']['fs']
    n_fft = cfg['data']['n_fft']
    hop_length = cfg['data']['hop_len']
    win_length = cfg['data']['win_len']

    # Doa info
    n_mics = 4
    fmin_doa = cfg['data']['fmin_doa']
    fmax_doa = cfg['data']['fmax_doa']
    
    """
    For the demo, fmax_doa = 4kHz, fs = 48kHz, n_fft = 512, hop = 300
    This results in the following:
        n_bins      = 257
        lower_bin   = 1
        upper_bin   = 42
        cutoff_bin  = 96 
        logspecs -> 95 bins total
        phasespecs -> 41 bins total
        
    Since these are all fixed, can we just put them into the config.yml instead
    and just read them from there and avoid these calculations
    """
    fmax_doa = np.min((fmax_doa, fs // 2))
    n_bins = n_fft // 2 + 1
    lower_bin = int(np.floor(fmin_doa * n_fft / float(fs)))
    upper_bin = int(np.floor(fmax_doa * n_fft / float(fs)))
    lower_bin = np.max((1, lower_bin))

    # Cutoff frequency for spectrograms
    fmax = 9000  # Hz, meant to reduce feature dimensions
    cutoff_bin = int(np.floor(fmax * n_fft / float(fs)))  # 9000 Hz, 512 nfft: cutoff_bin = 192
    assert upper_bin <= cutoff_bin, 'Upper bin for spatial feature is higher than cutoff bin for spectrogram!'

    # Normalization factor for salsa_lite --> 2*pi*f/c
    c = 343
    delta = 2 * np.pi * fs / (n_fft * c)
    freq_vector = np.arange(n_bins)
    freq_vector[0] = 1
    freq_vector = freq_vector[:, None, None]  # n_bins x 1 x 1
    
    # Checking parameters
    print("lower_bin    : ", lower_bin)
    print("upper_bin    : ", upper_bin)
    print("cutoff_bin   : ", cutoff_bin)
    
    # Extract features
    audio_fn_list = sorted(os.listdir(audio_dir))
    
    for count, file in enumerate(tqdm(audio_fn_list)):
        if file.endswith('.wav'):
            audio_file = os.path.join(audio_dir, file)
            audio_input, _ = librosa.load(audio_file, sr=fs, mono=False, dtype=np.float32)

            # Extract Log-Linear Spectrograms 
            log_specs = []
            for imic in np.arange(n_mics):
                stft = librosa.stft(y=np.asfortranarray(audio_input[imic, :]), n_fft=n_fft, hop_length=hop_length,
                                    center=True, window='hann', pad_mode='reflect')
                if imic == 0:
                    n_frames = stft.shape[1]
                    X = np.zeros((n_bins, n_frames, n_mics), dtype='complex')  # (n_bins, n_frames, n_mics)
                X[:, :, imic] = stft
                # Compute log linear power spectrum
                spec = (np.abs(stft) ** 2).T
                log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
                log_spec = np.expand_dims(log_spec, axis=0)
                log_specs.append(log_spec)
            log_specs = np.concatenate(log_specs, axis=0)  # (n_mics, n_frames, n_bins)

            # Compute spatial feature
            # X : (n_bins, n_frames, n_mics) , NIPD formula : -(c / (2pi x f)) x arg[X1*(t,f) . X2:M(t,f)]
            phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))
            phase_vector = phase_vector / (delta * freq_vector)
            phase_vector = np.transpose(phase_vector, (2, 1, 0))  # (n_mics, n_frames, n_bins)
            
            # Crop frequency
            log_specs = log_specs[:, :, lower_bin:cutoff_bin]
            phase_vector = phase_vector[:, :, lower_bin:cutoff_bin]
            phase_vector[:, :, upper_bin:] = 0
            
            # Stack features
            audio_feature = np.concatenate((log_specs, phase_vector), axis=0)
            audio_feature = audio_feature.astype(np.float32)
    
            # Now we normalize the first 4 log power spectrogram channels of SALSALITE
            n_feature_channels = 4
            scaler_dict = {}
            for i_chan in np.arange(n_feature_channels):
                scaler_dict[i_chan] = preprocessing.StandardScaler()
                scaler_dict[i_chan].partial_fit(audio_feature[i_chan, : , : ]) # (n_timesteps, n_features)
                
            # Extract mean and std
            feature_mean = []
            feature_std = []
            for i_chan in range(n_feature_channels):
                feature_mean.append(scaler_dict[i_chan].mean_)
                feature_std.append(np.sqrt(scaler_dict[i_chan].var_))

            feature_mean = np.array(feature_mean)
            feature_std = np.array(feature_std)

            # Expand dims for timesteps: (n_chanels, n_timesteps, n_features)
            feature_mean = np.expand_dims(feature_mean, axis=1)
            feature_std = np.expand_dims(feature_std, axis=1)
            audio_feature[:4] = (audio_feature[:4] - feature_mean)/feature_std
                
            feature_fn = os.path.join(feature_dir, file.replace('wav', 'h5'))
            with h5py.File(feature_fn, 'w') as hf:
                hf.create_dataset('feature', data=audio_feature, dtype=np.float32)


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

    class_types = ['Noise', 'Dog', 'Impact' , 'Speech']
    # Manual settings 
    frame_len = int(window_duration*fs) 
    hop_len = int(hop_duration*fs)   

    for ct in class_types:
        
        # raw data directory
        full_data_dir = os.path.join(concat_data_dir, ct)
        
        # cleaned, output data directory
        output_data_dir = './_audio/cleaned_data_{}s_{}s/{}'.format(window_duration, hop_duration, ct.lower())
        higher_level_dir = './_audio/cleaned_data_{}s_{}s/'.format(window_duration, hop_duration)
        if add_wgn: 
            higher_level_dir = higher_level_dir.replace("cleaned_data" , "add_wgn")
            output_data_dir = output_data_dir.replace('cleaned_data' , 'add_wgn')
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
    features    (np.ndarray)  : SALSA-Lite Features (7, n_frames_in , freq_bins)
    frame_gt    (np.ndarray)  : Ground truth labels for the audio file, (n_frames_out, n_classes * 3)
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
    gts = filename.split('_') # [class, azimuth, ... ] assumed the same for all frames in audio
    
    if gts[0] == "noise": # Everything is zero
        full_gt  = np.zeros(len(class_labels) * 3, dtype = np.float32)
        full_gt = np.reshape(full_gt, (1, len(full_gt)))
        frame_gt = np.concatenate([full_gt] * n_frames_out, axis = 0)
        return features , frame_gt

    full_gt = np.zeros( len(class_labels) * 3, dtype=np.float32)
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
    # azi_rad = np.deg2rad(gt_azi)
    azi_rad = gt_azi * np.pi / 180.0 # Convert to radian unit this way
    full_gt[class_idx + len(class_labels)] = np.cos(azi_rad) # X-coordinate
    full_gt[class_idx + 2 * len(class_labels)] = np.sin(azi_rad) # Y-coordinate
    # Produce the ground truth labels for a single frame, expand the frame dimensions later
    full_gt = np.reshape(full_gt, (1, len(full_gt)))
    
    # Expand it such that it meets the required n_frames output
    frame_gt = np.concatenate([full_gt] * n_frames_out, axis=0)

    return features , frame_gt


def create_dataset(feature_path_dir,
                   for_cpu = False):
    """Create dataset arrays to be stored in .npy format later

    Input:
        feature_path_dir : filepath to where the features are stored in
        for_cpu          : boolean. True if generating dataset for CPU usage in form of NHWC

    Returns:
        data      : dataset of features 
        gt_labels : dataset of ground truth labels
    """
    data = []
    gt_labels = []
    classes = ['dog', 'impact', 'speech', 'noise']
    for cls in classes:
        feature_dir = os.path.join(feature_path_dir, cls)
        
        # # Get the class-wise mean and std.dev to normalize features
        # scaler_dir = os.path.join(feature_path_dir, 'scalers')
        # scaler_filepath = os.path.join(scaler_dir, '{}_feature_scaler.h5'.format(cls))
        # with h5py.File(scaler_filepath, 'r') as shf:
        #     mean = shf['mean'][:]
        #     std = shf['std'][:]
            
        # Loop through features, add to database
        print("Generating dataset for : ", feature_dir)
        for file in tqdm(sorted(os.listdir(feature_dir))):
            if file.endswith('.h5'):
                full_filepath = os.path.join(feature_dir, file)
                salsa_features , gt_label = load_file(filepath=full_filepath)
                # salsa_features[:4] = (salsa_features[:4]-mean)/std
                # Because for CPU, Tensorflow only operates on height (freq) , width (time) , channel shape
                if for_cpu : salsa_features = np.transpose(salsa_features, [2,1,0])
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
    
    # General Configs 
    # Window, Hop duration in seconds 
    ws = 0.4
    hs = 0.2
    seperate_audio = True
    create_features = True
    generate_dataset = True
    # dataset_dir = "./training_datasets/demo_dataset_{}s_{}s_NHWC/".format(ws,hs)
    dataset_dir = "./training_datasets/demo_dataset_{}s_{}s_wgn20/".format(ws,hs)
    # concat_audio_dir = ".\data\Dataset_concatenated_tracks"
    concat_audio_dir = ".\data\scaled_audio"
    
    # Segment the audio first 
    if seperate_audio: 
        audio_upper_dir = segment_concat_audio(concat_data_dir = concat_audio_dir,
                                               window_duration=ws,
                                               hop_duration=hs,
                                               add_wgn=True,
                                               snr_db=20) 
    else:
        audio_upper_dir = './_audio/cleaned_data_{}s_{}s/'.format(ws, hs)

    # Next, we extract the features for the segmented audio clips

    feature_upper_dir = os.path.join('.' , '_features', 'features_{}s_{}s'.format(ws, hs))
    
    if create_features:
        classes = ['dog', 'impact', 'speech', 'noise']
        for cls in classes:
            audio_dir = os.path.join(audio_upper_dir, cls)
            feature_dir = os.path.join(feature_upper_dir, cls)
            full_feature_with_norm(audio_dir, feature_dir)

    # Now we generate the entire dataset (features, labels) and store them in .npy files so that training
    # data loading is alot easier.
    if generate_dataset:
        # Create arrays for feature, ground truth labels dataset
        data , gt = create_dataset(feature_upper_dir, for_cpu = True)

        # Create directories for storage
        os.makedirs(dataset_dir, exist_ok=True)

        feature_fp = os.path.join(dataset_dir, "demo_salsalite_features.npy")
        np.save(feature_fp, data, allow_pickle=True)
        print("Features saved at {}!".format(feature_fp))
        
        gt_fp = os.path.join(dataset_dir, 'demo_gt_labels.npy')
        np.save(gt_fp, gt, allow_pickle=True)
        print("Active class ground truth saved at {}!".format(gt_fp))