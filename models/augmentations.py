import tensorflow as tf
import random


def freq_mask(spec, F=2, num_masks=1, replace_with_zero=True):
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

def random_bool(percent_chance=0.7):
    # Returns True with a 70% chance, and False with a 30% chance
    return random.random() < percent_chance

if __name__ == "__main__":
    rt = tf.random.uniform(shape=[1, 95, 81, 7])
    print(rt)
    print(freq_mask(rt))