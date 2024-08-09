import subprocess
import sys
import os

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
required_packages = [
    "matplotlib", "Pillow", "gdown", "numpy", "keras", 
    "opencv-python", "tensorflow", "streamlit", "pyngrok"
]

for package in required_packages:
    install(package)

# Now import all the required libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gdown
import cv2
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, concatenate, add
from keras.models import Model
import tensorflow as tf
from copy import deepcopy
import colorsys
from pyngrok import ngrok

# Ensure all necessary data is downloaded
DATA_ROOT = 'data'
os.makedirs(DATA_ROOT, exist_ok=True)

# Download necessary files using wget
subprocess.run(["wget", "-O", os.path.join(DATA_ROOT, "image.jpg"), "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/image.jpg"])
subprocess.run(["wget", "-O", os.path.join(DATA_ROOT, "image2.jpg"), "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/image2.jpg"])
subprocess.run(["wget", "-O", os.path.join(DATA_ROOT, "video1.mp4"), "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/6.mp4"])
subprocess.run(["wget", "-O", os.path.join(DATA_ROOT, "yolo_weights.h5"), "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/yolo.h5"])

# Authenticate ngrok (required for running on local with Streamlit sharing, not needed on Streamlit Cloud)
subprocess.run(["ngrok", "authtoken", "2kQ0MRi11P8t2mp4tPVzJjB4XnD_4Ze9SyY1ZPfiVkgr4KtE6"])

# Example placeholder functionality for Streamlit app
st.title('Object Detection App')
st.write("This is a placeholder for the object detection functionality.")

# You can add more Streamlit components to build out your app
# For example, adding an image uploader, processing the image, and displaying results

