# demo_tf_lite
SALSA-Lite demo code implemented in TF-Lite

### Audio Segmentation 
Navigate to the `analysis` folder. The `audio_segment.ipynb` handles the splitting of the concatenated tracks into segments of specified window size and hop size. The data will be stored in `dataset/cleaned_data/` and sub-folders for each class.

### Feature Extraction 
Data to be stored in the `cleaned_data` folder, with sub-folders for each sub-class. Run the `demo_extract_salsalite.py` script to convert the data into salsa-lite features. The features will then be stored in `./dataset/features` 

### Dataset Creation
In order to feed the data into the model, we will use generators and the Tensorflow `.from_generator()` function. We will convert the features and their corresponding ground truth labels to `.npy` files and store them in `./dataset/demo_dataset`. The ground truths are in the form of (n_timeframes x 3*n_classes), the first n_classes representing the SED ground truth, and the next two n_classes the X,Y Cartesian coordinate directions.

### Model Training
In the `models` folder, just run `demo_training.py` and adjust the global settings accordingly. 

### Inference
TODO

