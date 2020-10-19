import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

def getDataset(dSet):
    for tensor in dSet:
        return tensor[0]

#create dataset from text files
path = Path(__file__).parent / 'test data'
dataset = keras.preprocessing.text_dataset_from_directory(path)

#create text vectorizer to create mapping between datasets
text_vectorizer = layers.experimental.preprocessing.TextVectorization(output_mode ="int")

print(dataset)

initalDataSet = getDataset(dataset)
print(initalDataSet)
#adapt the vectorizer and appliy it to the data to create mapping from en to fr
text_vectorizer.adapt(initalDataSet)

dataset= dataset.map(
    lambda x, y: (text_vectorizer(x), y)
)
#get mapped data set
mappedDataSet = getDataset(dataset)
print(mappedDataSet)




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
print(model.layers) 

for dSet in dataset:
    text_vectorizer.adapt(dSet[0])
    dataset= dataset.map(
        lambda x, y: (text_vectorizer(x), y)
    )
for dSet in dataset:
    print(dSet[0])

"""