"""
Consists of function(s) needed to extract salsa-lite features for microphone array format

This file should be used
"""
import librosa
import numpy as np
import yaml
from sklearn import preprocessing
import os
import time

def extract_features(audio_data,
                     cfg = None,
                     data_config: str = './configs/salsa_lite_demo_3class.yml'
                    ) -> None:
    """
    Extract SALSA-Lite features from a segment of audio 
    SALSA-Lite consists of:
        - 4 Log-linear spectrograms 
        - 3 Normalized Interchannel Phase Differences 

    [This needs to be revised for the demo]
    The frequency range of log-linear spectrogram is 0 to 9kHz.

    Arguments
    -----------
    audio_data (np.ndarray) : audio data from mic streaming (assuming it is 4 channels)
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
    
    # Extract the features from the audio data
    """
    The audio data is from the ambimik, assuming that it is a four-channel array
    The shape should be (4 , x) for (n_channels, time*fs)
    """
    log_specs = []
    for imic in np.arange(n_mics):
        stft = librosa.stft(y=np.asfortranarray(audio_data[imic, :]), 
                            n_fft=n_fft, 
                            hop_length=hop_length,
                            center=True, 
                            window='hann', 
                            pad_mode='reflect')
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
        
    return audio_feature

if __name__ == "__main__":
    # Ensure that script working directory is same directory as the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    print("Changing directory to : ", dname)
    os.chdir(dname)
        
    data_config = "./configs/salsa_lite_demo_3class.yml"
    with open(data_config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    test_audio = np.random.rand(4,48000)
    iterations = 10
    start_time = time.time()
    for i in range(iterations):
        test_feat = extract_features(test_audio)
    end_time = time.time()
    print("Time taken  : {}".format((end_time-start_time)/iterations))

    start_time = time.time()
    for i in range(iterations):
        test_feat = extract_features(test_audio,cfg)
    end_time = time.time()
    print("Time taken  : {}".format((end_time-start_time)/iterations))
    