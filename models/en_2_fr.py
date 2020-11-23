
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from .format_data import data_formatting

#get the vectors for the model

encoder_input_vector, decoder_input_vector, decoder_output_vector = data_formatting(
    Path(__file__).parent  / 'test data'  / 'europarl-v7-EN.txt', 
    Path(__file__).parent  / 'test data'  / 'europarl-v7-FR.txt')

print(encoder_input_vector[1,12])



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