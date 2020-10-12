import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

#create dataset from text files
path = Path(__file__).parent / 'test data'
dataset = keras.preprocessing.text_dataset_from_directory(path)

for data, labels in dataset:
   print(data.shape)  # (64,)
   print(data.dtype)  # string
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
"""
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
print(model.layers) """