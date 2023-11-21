import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

def display_spectrogram(spectrogram, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    
    # Assuming spectrogram is of shape (n_mel, n_time, n_channels)
    # and you want to display the first channel
    channel = 4
    ax.imshow(spectrogram[:, :, channel])
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel-frequency')
    plt.show()

def freq_mask(spec, F=3, num_masks=1, replace_with_zero=True):
    """
    Apply a frequency mask to the spectrogram. Starting point of the mask is f0, while the max. width
    of the mask is defined by F. Seeing as how the log-power spectrogram is (1,41) bins and the NIPD
    spectrograms are (1,21), setting F=2 will allow for maximum 5%/10% of spectrogram to be masked for 
    the log-power/NIPD spectrograms respectively.
    
    The masks will replace the original values with zero. If `replace_with_zero` is False, the mask will
    replace the original values with the mean of the spectrogram values instead. 
    
    Usage:
        Just pass the function to the dataset via a .map() function
        For example, 
            augmented_dataset = dataset.map(lambda x, y : (freq_mask(x), y))
    """
    cloned = tf.identity(spec)
    num_mel_channels = cloned.shape[0]
    
    for i in range(0, num_masks):        
        f = tf.random.uniform([], minval=0, maxval=F, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=num_mel_channels - f, dtype=tf.int32)

        if num_mel_channels == f: 
            pass
        else:
            mask = tf.concat((tf.ones(shape=(f0,)), tf.zeros(shape=(f,)), tf.ones(shape=(num_mel_channels-f0-f,))), axis=0)
            mask = tf.reshape(mask, (num_mel_channels, 1, 1))
            mask = tf.tile(mask, [1, cloned.shape[1], cloned.shape[2]])
            
            if replace_with_zero:
                cloned = cloned * mask
            else:
                cloned = cloned * mask + tf.math.reduce_mean(cloned, axis=0) * (1-mask)
    return cloned

def random_shift_updown(spec):
    
    n_mel, n_time, n_channels = spec.shape
    freq_shift_range = int(n_mel * 0.08)
    shift_len = np.random.randint(1, freq_shift_range, 1)[0]
    direction = np.random.choice(['up', 'down'], 1)[0]

    new_spec = tf.identity(spec)
    
    if direction == 'up':
        paddings = tf.constant([[shift_len, 0], [0, 0], [0, 0]])
        new_spec = tf.pad(new_spec, paddings, "REFLECT")
        new_spec = new_spec[0:n_mel, :, :]
        
    else:
        paddings = tf.constant([[0, shift_len], [0, 0], [0, 0]])
        new_spec = tf.pad(new_spec, paddings, "REFLECT")
        new_spec = new_spec[shift_len:, :, :]
        
    return new_spec

def random_bool(percent_chance=0.7):
    # Returns True with a 70% chance, and False with a 30% chance
    return random.random() < percent_chance

if __name__ == "__main__":
    
    # Define the shape of your spectrogram
    n_mel = 95
    n_time = 81
    n_channels = 7

    # Create a 1D tensor with increasing values from 0 to 1
    mel_values = tf.linspace(0.0, 1.0, n_mel)

    # Expand dimensions to match the shape of the spectrogram
    mel_values = tf.expand_dims(mel_values, axis=-1)  # shape becomes (n_mel, 1)
    mel_values = tf.expand_dims(mel_values, axis=-1)  # shape becomes (n_mel, 1, 1)

    # Tile the values to fill the time and channel dimensions
    spectrogram = tf.tile(mel_values, [1, n_time, n_channels])  # shape becomes (n_mel, n_time, n_channels)
    # rt = tf.random.uniform(shape=[95, 81, 7])
    # print(rt)
    # print(freq_mask(rt))
    # print(random_shift_updown(rt))
    
    display_spectrogram(spectrogram)
    display_spectrogram(freq_mask(rt))
    display_spectrogram(random_shift_updown(spectrogram))