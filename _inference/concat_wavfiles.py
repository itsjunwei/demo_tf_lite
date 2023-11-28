import wave
import os
import librosa
import numpy as np
# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)


# Directory containing .wav files
directory = './hybrid'

# Output file
output_file = 'hybrid_concat.wav'

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