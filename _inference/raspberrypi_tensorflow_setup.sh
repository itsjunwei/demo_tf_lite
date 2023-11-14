#!/bin/bash

# fresh start
sudo apt-get update
sudo apt-get upgrade

# install the dependencies
sudo apt-get install git python3-pip
sudo -H pip3 install --upgrade protobuf==3.20.0

# Setting up virutal environment
sudo pip3 install virtualenv
virtualenv -p python3 SALSA_ENV
source SALSA_ENV/bin/activate

# install the following version of Cython for compatibility issues
git clone https://github.com/Qengineering/Tensorflow-io.git
cd Tensorflow-io
sudo -H pip3 install tensorflow_io_gcs_filesystem-0.23.1-cp39-cp39-linux_aarch64.whl

# install gdown to download from Google drive
sudo -H pip3 install gdown
gdown https://drive.google.com/uc?id=1G2P-FaHAXJ-UuQAQn_0SYjNwBu0aShpd
sudo -H pip3 install tensorflow-2.10.0-cp39-cp39-linux_aarch64.whl

# Create and download the Tensorflow code needed
git clone https://github.com/itsjunwei/demo_tf_lite.git

# Now we install all the packages 

# installing h5py
sudo apt install libhdf5-dev
pip3 install h5py==2.10.0

# installing sklearn
pip3 install scikit-learn==0.23.2 --no-build-isolation

# install the remaining packages
pip3 install -r  requirements_for_rpi.txt

echo "All packages installed!"
pip3 list
