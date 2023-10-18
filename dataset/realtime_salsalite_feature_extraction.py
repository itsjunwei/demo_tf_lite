"""
Consists of function(s) needed to extract salsa-lite features for microphone array format

This file should be used
"""

import librosa
import numpy as np
import yaml
from timeit import default_timer as timer
from sklearn import preprocessing
import h5py

def extract_features(audio_file,
                     is_globnorm=True,
                     data_config: str = './configs/salsa_lite_demo_3class.yml'
                    ) -> None:
    """
    Extract SALSA-Lite features from a segment of audio 
    
    SALSA-Lite consists of 4 Log-linear spectrograms + 3 Normalized Interchannel Phase Differences 

    [This needs to be revised for the demo]
    The frequency range of log-linear spectrogram is 0 to 9kHz.
    

    Arguments
    -----------
    audio_file  (filepath)  : .wav audio file to be read in (or whichever audio file format the mic input will be stored as)
    is_globnorm (boolean)   : True if use pre-calculated global mean/std.dev to normalize features, False if use sample mean/std.dev
    data_config (.yml file) : filepath to the .yml config file used for SALSA-Lite

    Returns
    --------
    audio_feature : np.ndarray of SALSA-Lite features with shape of (7, t, f) --> t depends on length of audio, f depends on frequency range 

    To-do
    ------
    Remove config parameters to outside function. Call once and input to a dictionary and read it from there instead 

    Check the audio input format --> adjust the librosa stft function inputs from there
    """
    
    # Load data config files
    with open(data_config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

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
    For SALSA-Lite, fmax_doa = 2kHz, fs = 24kHz, n_fft = 512
    This results in the following:
        n_bins      = 257
        lower_bin   = 1
        upper_bin   = 42
        cutoff_bin  = 192 
    """
    fmax_doa = np.min((fmax_doa, fs // 2))
    n_bins = n_fft // 2 + 1
    lower_bin = int(np.floor(fmin_doa * n_fft / float(fs)))  # 512: 1; 256: 0
    upper_bin = int(np.floor(fmax_doa * n_fft / float(fs)))  # 9000Hz: 512: 192, 256: 96
    lower_bin = np.max((1, lower_bin))

    # Cutoff frequency for spectrograms
    fmax = 9000  # Hz
    cutoff_bin = int(np.floor(fmax * n_fft / float(fs)))  # 9000 Hz, 512 nfft: cutoff_bin = 192
    assert upper_bin <= cutoff_bin, 'Upper bin for spatial feature is higher than cutoff bin for spectrogram!'

    # Normalization factor for salsa_lite --> 2*pi*f/c
    c = 343
    delta = 2 * np.pi * fs / (n_fft * c)
    freq_vector = np.arange(n_bins)
    freq_vector[0] = 1
    freq_vector = freq_vector[:, None, None]  # n_bins x 1 x 1

    # Extract features
    start_time = timer()
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


    if is_globnorm:
        with h5py.File('./configs/mic_feature_scaler.h5', 'r') as hf:
            mean = hf['mean'][:]
            std = hf['std'][:]
        log_specs = (log_specs-mean)/std
    else:
        # Normalize features per file (sample normalization)

        # Initialize 
        scaler_dict = {}
        for i_chan in np.arange(4):
            scaler_dict[i_chan] = preprocessing.StandardScaler()

        # Iterate through data
        for i_chan in range(4):
            scaler_dict[i_chan].partial_fit(log_specs[i_chan, : , :])  # (n_timesteps, n_features)

        # Extract mean and std
        feature_mean = []
        feature_std = []
        for i_chan in range(4):
            feature_mean.append(scaler_dict[i_chan].mean_)
            feature_std.append(np.sqrt(scaler_dict[i_chan].var_))

        feature_mean = np.array(feature_mean)
        feature_std = np.array(feature_std)

        # Expand dims for timesteps: (n_chanels, n_timesteps, n_features)
        feature_mean = np.expand_dims(feature_mean, axis=1)
        feature_std = np.expand_dims(feature_std, axis=1)
        # Normalize the Log-Linear Spectrograms
        log_specs =(log_specs-feature_mean)/feature_std

    # Stack features
    audio_feature = np.concatenate((log_specs, phase_vector), axis=0)
    audio_feature = audio_feature.astype(np.float32)

    # Feature Extraction time (for tracking)
    end_time = timer() - start_time

    return audio_feature

