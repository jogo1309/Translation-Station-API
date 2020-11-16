import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path


from .read_data import pre_process_data


#english dataset
path = Path(__file__).parent  / 'test data'  / 'europarl-v7-EN.txt'
input_sentances, input_words, num_encoder_tokens, max_input_length = pre_process_data(path, False)

#print(input_words)
#print(num_input_tokens)
#print(max_input_length)

#french dataset
path = Path(__file__).parent  / 'test data'  / 'europarl-v7-FR.txt'
output_sentances, output_words, num_decoder_tokens, max_output_length = pre_process_data(path, True)

#print(output_words)
#print(max_output_length)
#print(max_output_length)

#word dictionaries
input_word_index = dict([(word, i) for i, word in enumerate(input_words)])
output_word_index = dict([(word , i) for i,word in enumerate(output_words)])

#one hot encoding array's
#each array has 3 dimentions 1: amount of sentances, 2: max length of a sentance, 3: amount of unique words in the dataset

#empty vectors of correct length
encoder_input_vectors = np.zeros((len(input_sentances), max_input_length, num_encoder_tokens), dtype='float32')
decoder_input_vectors = np.zeros((len(output_sentances), max_output_length, num_decoder_tokens), dtype='float32')
decoder_output_vectors = np.zeros((len(output_sentances), max_output_length, num_decoder_tokens), dtype='float32')

#create one hot encoding

print(encoder_input_vectors)

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