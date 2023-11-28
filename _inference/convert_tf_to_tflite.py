import tensorflow as tf
import os 
from inference_model import get_model

# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)
os.system('cls')
os.makedirs('./tflite_models', exist_ok=True)

# Hard coded settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
window_duration_s = 1
feature_len = int(window_duration_s * 10 * 16 + 1) # General FFT formula
resnet_style = 'bottleneck'
n_classes = 4


input_shape = (95, feature_len, 7) # Height, Width , Channels shape
print("Input shape : ", input_shape)
# Get the salsa-lite model
salsa_lite_model = get_model(input_shape    = input_shape, 
                             resnet_style   = resnet_style, 
                             n_classes      = n_classes,
                             azi_only       = True,
                             batch_size     = 1)


for model_file in os.listdir('./saved_models'):
    if model_file.endswith('.h5'):
        trained_model_filepath = os.path.join('./saved_models', model_file)
        model_name = model_file.replace('.h5', "")
        
        if not os.path.exists("./tflite_models/{}".format(model_name)):

            """Load the pre-trained model"""
            print("Loading model from : ", trained_model_filepath)
            salsa_lite_model.load_weights(trained_model_filepath)

            
            """Converting and saving as TFLITE, only need to do this once"""
            salsa_lite_model.save("./tflite_models/{}".format(model_name))
            converter = tf.lite.TFLiteConverter.from_saved_model("./tflite_models/{}".format(model_name))
            tflite_model = converter.convert()

            with open('./tflite_models/{}/tflite_model.tflite'.format(model_name) , 'wb') as f:
                f.write(tflite_model)