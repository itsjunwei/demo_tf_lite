"""
File to contain all the utility functions that may be useful throughout the model

Current functions:
    - Learning Rate scheduler 
"""
import tensorflow as tf
from keras.backend import get_value

def custom_scheduler(epoch, 
                     initial_lr = 3e-4, 
                     final_lr = 1e-4, 
                     change_epoch=0.7):
    
    total_epochs = get_value(model.optimizer.iterations) // get_value(model.optimizer.decay_steps)
    if epoch < change_epoch * total_epochs:
        # Return the initial learning rate
        return initial_lr
    else:
        # Return the final learning rate
        return final_lr