import wave
import os
import librosa
import numpy as np
import soundfile as sf

def normalize_array(array):
    """
    Normalize the array, each row locally. Used after we segment the audio into segments to mimic the way
    that we will normalize the audio input during demo conditions. Normalizing (instead of scaling) helps 
    to preserve the relative distance between data points, which could be better representations of the data
    for the machine to learn.
    
    Arguments
    --------
    array (np.ndarray) : In this case, consider this the signal we wish to normalize
    
    Returns
    ------
    array_normed (np.ndarray) : Normalized array
    """
    array_normed = []
    for i in range(len(array)):
        x = array[i]
        x = x - np.mean(x)
        x = x / np.max(np.abs(x))
        array_normed.append(x)
    array_normed = np.array(array_normed)
    
    return array_normed

# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)


# Directory containing .wav files
directory = './hybrid_jwtest'

# Output file
os.makedirs(os.path.join(directory, 'concat'))
output_file = os.path.join(directory, 'concat', 'hybrid_concat.wav')

# Get a list of .wav files in the directory
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

# Open the output file
with wave.open(output_file, 'wb') as outfile:

    # For each .wav file
    for i, wav_file in enumerate(wav_files):

        # Open the .wav file
        with wave.open(os.path.join(directory, wav_file), 'rb') as infile:

            # If this is the first file, set output parameters to match input parameters
            if i == 0:
                outfile.setparams(infile.getparams())

            # Write audio frames to output file
            outfile.writeframes(infile.readframes(infile.getnframes()))
            
audio_data , _ = librosa.load(output_file, sr=48000,
                              mono=False, dtype=np.float32)
norm_audio = normalize_array(audio_data)
print(norm_audio.shape)
outfile = output_file.replace('concat', 'scaled_concat')
sf.write(output_file, data=norm_audio.T, samplerate=48000)