import tensorflow as tf
import random


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=True):
    """
    Apply masking to a spectrogram in the frequency domain.
    Args:
    spec: The input spectrogram. input_shape = (95, 161, 7) ; (n_melbins, n_timeframes, n_channels)
    F: Parameter to determine maximum width of each mask.
    num_masks: Number of masks to apply.
    replace_with_zero: Whether to replace the masked parts with zero or their mean.
    """
    cloned = tf.identity(spec)
    num_mel_channels = cloned.shape[-3]
    
    for i in range(0, num_masks):        
        f = tf.random.uniform([], minval=0, maxval=F, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=num_mel_channels - f, dtype=tf.int32)

        # Determines the final shape for broadcast
        mask_shape = [1, num_mel_channels, 1, 1]

        mask = tf.pad(tf.ones([1, f, 1, 1]), ((0, 0), (f0, num_mel_channels - f - f0), (0, 0), (0, 0)))
        mask = 1 - mask

        if replace_with_zero: 
            cloned = cloned * mask
        else: 
            cloned = cloned * mask + tf.reduce_mean(cloned, axis=-3, keepdims=True) * (1 - mask)
    return cloned

def random_bool(percent_chance=0.7):
    # Returns True with a 70% chance, and False with a 30% chance
    return random.random() < percent_chance


